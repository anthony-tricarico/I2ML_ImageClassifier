"""
CLIP-based Image Retrieval System with Supervised Contrastive Learning

This script implements a complete pipeline for training and evaluating
a CLIP-based image retrieval system using supervised contrastive learning.
"""

import os
import json
import time
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.cuda.amp import autocast, GradScaler

from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor
from sklearn.metrics.pairwise import cosine_similarity


class CLIPImageDatasetTrain(Dataset):
    """Dataset for training images organized in class folders."""
    
    def __init__(self, root_dir, processor):
        """
        Args:
            root_dir: Main folder with subfolders for each class
            processor: CLIPProcessor instance from Hugging Face
        """
        self.processor = processor
        self.image_paths = []
        self.labels = []

        # Create mapping: folder_name → numeric label
        class_names = sorted(os.listdir(root_dir))
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        for class_name in class_names:
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_path, fname))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        return pixel_values, label


class CLIPImageDatasetTest(Dataset):
    """Dataset for test images without labels."""
    
    def __init__(self, image_dir, processor):
        """
        Args:
            image_dir: Directory with images
            processor: CLIPProcessor instance from Hugging Face
        """
        self.image_dir = image_dir
        self.image_paths = [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Use CLIP processor to get pixel_values
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # Remove batch dim

        return pixel_values, img_path  # Return tensor and path for tracking


class BalancedBatchSampler(Sampler):
    """Sampler that creates balanced batches with equal samples per class."""
    
    def __init__(self, labels, n_classes, n_samples_per_class):
        """
        Args:
            labels: List of labels (same length as dataset)
            n_classes: How many different classes per batch
            n_samples_per_class: How many images per class per batch
        """
        self.labels = labels
        self.n_classes = n_classes
        self.n_samples_per_class = n_samples_per_class

        # Build mapping: label → list of indices
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

        # Filter classes that DON'T have enough images
        self.label_to_indices = {
            label: idxs for label, idxs in self.label_to_indices.items()
            if len(idxs) >= self.n_samples_per_class
        }

        self.labels_set = list(self.label_to_indices.keys())

        if len(self.labels_set) < self.n_classes:
            raise ValueError(f"Number of valid classes ({len(self.labels_set)}) "
                             f"lower than required n_classes per batch ({self.n_classes})")

        # Maximum number of estimated batches
        self.num_batches = len(labels) // (n_classes * n_samples_per_class)

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            selected_classes = random.sample(self.labels_set, self.n_classes)

            for cls in selected_classes:
                indices = self.label_to_indices[cls]
                sampled_indices = random.sample(indices, self.n_samples_per_class)
                batch.extend(sampled_indices)

            yield batch

    def __len__(self):
        return self.num_batches


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss."""
    
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: tensor (B, D), already normalized (L2-normalized embeddings)
            labels: tensor (B,) with labels
        """
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # (B, B)

        # Calculate similarity: normalized dot product
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask: exclude comparison with self
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(features.shape[0]).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        # Calculate loss
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Average over anchors (only if there are positives)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1.0)

        loss = -mean_log_prob_pos.mean()
        return loss


class CLIPEncoder(nn.Module):
    """CLIP vision encoder wrapper."""
    
    def __init__(self, model_name="openai/clip-vit-large-patch14-336"):
        super(CLIPEncoder, self).__init__()
        # Load only the visual part of CLIP
        self.vision_encoder = CLIPModel.from_pretrained(model_name).vision_model

    def forward(self, pixel_values):
        outputs = self.vision_encoder(pixel_values=pixel_values)
        embeddings = outputs.pooler_output  # shape (B, D)
        embeddings = F.normalize(embeddings, p=2, dim=1)  # L2-normalization
        return embeddings


class ImageRetrievalTrainer:
    """Main trainer class for the image retrieval system."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPImageProcessor.from_pretrained(config['model_name'])
        
        # Initialize model
        self.encoder = CLIPEncoder(config['model_name']).to(self.device)
        
        # Initialize optimizer and loss
        self.optimizer = optim.Adam(
            self.encoder.parameters(), 
            lr=config['learning_rate'], 
            weight_decay=config['weight_decay']
        )
        self.criterion = SupConLoss(temperature=config['temperature'])
        self.scaler = GradScaler()
        
        # Create directories
        os.makedirs(config['results_dir'], exist_ok=True)
        os.makedirs(config['weights_dir'], exist_ok=True)
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    def create_dataloaders(self):
        """Create training and evaluation dataloaders."""
        # Training dataset
        train_dataset = CLIPImageDatasetTrain(self.config['train_dir'], self.processor)
        
        # Balanced batch sampler
        sampler = BalancedBatchSampler(
            labels=train_dataset.labels,
            n_classes=self.config['n_classes_per_batch'],
            n_samples_per_class=self.config['n_samples_per_class']
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            persistent_workers=True
        )
        
        # Evaluation datasets
        gallery_dataset = CLIPImageDatasetTest(self.config['gallery_dir'], self.processor)
        query_dataset = CLIPImageDatasetTest(self.config['query_dir'], self.processor)
        
        self.gallery_loader = DataLoader(
            gallery_dataset, 
            batch_size=self.config['eval_batch_size'], 
            shuffle=False
        )
        self.query_loader = DataLoader(
            query_dataset, 
            batch_size=self.config['eval_batch_size'], 
            shuffle=False
        )
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.encoder.train()
        total_loss = 0
        num_batches = 0
        
        train_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}/{self.config['num_epochs']}")
        
        for pixel_values, labels in train_bar:
            pixel_values = pixel_values.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            with autocast():
                embeddings = self.encoder(pixel_values)
                loss = self.criterion(embeddings, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            train_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate(self, epoch):
        """Evaluate the model and compute retrieval results."""
        self.encoder.eval()
        print(f"\nEvaluating after epoch {epoch+1}...")
        
        gallery_embeddings = []
        query_embeddings = []
        gallery_paths = []
        query_paths = []
        
        with torch.no_grad():
            # Extract gallery embeddings
            print("Extracting gallery embeddings...")
            for pixel_values, paths in tqdm(self.gallery_loader, desc="Gallery"):
                pixel_values = pixel_values.to(self.device, non_blocking=True)
                with autocast():
                    emb = self.encoder(pixel_values).cpu()
                gallery_embeddings.append(emb.numpy())
                gallery_paths.extend(paths)
            
            # Extract query embeddings
            print("Extracting query embeddings...")
            for pixel_values, paths in tqdm(self.query_loader, desc="Query"):
                pixel_values = pixel_values.to(self.device, non_blocking=True)
                with autocast():
                    emb = self.encoder(pixel_values).cpu()
                query_embeddings.append(emb.numpy())
                query_paths.extend(paths)
        
        # Stack embeddings
        gallery_embeddings = np.vstack(gallery_embeddings)
        query_embeddings = np.vstack(query_embeddings)
        
        # Compute cosine similarity
        print("Computing cosine similarities...")
        similarity_matrix = cosine_similarity(query_embeddings, gallery_embeddings)
        
        # Get top-k results
        top_k_indices = np.argsort(similarity_matrix, axis=1)[:, -self.config['top_k']:][:, ::-1]
        
        # Build results dictionary
        results = {
            os.path.basename(query_paths[i]):
            [os.path.basename(gallery_paths[idx]) for idx in indices]
            for i, indices in enumerate(top_k_indices)
        }
        
        return results
    
    def save_results(self, results, epoch):
        """Save retrieval results to file."""
        output_file = os.path.join(self.config['results_dir'], f"retrieval_results_epoch_{epoch+1:02d}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
        return output_file
    
    def save_model(self, epoch, avg_loss):
        """Save model weights."""
        weights_file = os.path.join(self.config['weights_dir'], f"encoder_epoch_{epoch+1:02d}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': avg_loss,
        }, weights_file)
        print(f"Model weights saved to {weights_file}")
    
    def preview_results(self, results, epoch, num_preview=3):
        """Preview first few results."""
        print(f"\nFirst {num_preview} results for epoch {epoch+1}:")
        for i, (query, retrieved) in enumerate(results.items()):
            if i >= num_preview:
                break
            print(f"Query: {query}")
            print(f"Top-3 Retrieved: {retrieved[:3]}")
            print("-" * 30)
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*50}")
        print("STARTING TRAINING")
        print(f"{'='*50}")
        
        self.create_dataloaders()
        
        for epoch in range(self.config['num_epochs']):
            print(f"\n{'='*50}")
            print(f"EPOCH {epoch+1}/{self.config['num_epochs']}")
            print(f"{'='*50}")
            
            # Training
            avg_loss = self.train_epoch(epoch)
            print(f"Epoch [{epoch+1}/{self.config['num_epochs']}] - Avg Training Loss: {avg_loss:.4f}")
            
            # Evaluation
            results = self.evaluate(epoch)
            
            # Save results and model
            self.save_results(results, epoch)
            self.save_model(epoch, avg_loss)
            
            # Preview results
            self.preview_results(results, epoch)
        
        print(f"\n{'='*50}")
        print("TRAINING COMPLETED!")
        print(f"All results saved in '{self.config['results_dir']}'")
        print(f"All model weights saved in '{self.config['weights_dir']}'")
        print(f"{'='*50}")


def get_default_config():
    """Get default configuration parameters."""
    return {
        # Model parameters
        'model_name': "openai/clip-vit-large-patch14-336",
        'temperature': 0.07,
        
        # Training parameters
        'num_epochs': 1,
        'learning_rate': 1e-7,
        'weight_decay': 0.01,
        'n_classes_per_batch': 4,
        'n_samples_per_class': 2,
        
        # Data parameters
        'train_dir': "train",
        'gallery_dir': "test-2/gallery",
        'query_dir': "test-2/query",
        'eval_batch_size': 8,
        'num_workers': 6,
        'top_k': 10,
        
        # Output parameters
        'results_dir': "results",
        'weights_dir': "weights",
    }


def main():
    """Main function to run the training pipeline."""
    # Get configuration
    config = get_default_config()
    
    # Print configuration
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Check CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Create trainer and start training
    trainer = ImageRetrievalTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
    
    