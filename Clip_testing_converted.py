
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import transforms
import torch.optim as optim
from transformers import  CLIPModel
import torch.optim as optim
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

train_dir = 'data/Data_example/training'
gallery_dir = 'data/Data_example/test/gallery'
query_dir = 'data/Data_example/test/query'

from torch.utils.data import Dataset
from PIL import Image
import os

class CLIPImageDataset(Dataset):
    def __init__(self, image_dir, processor):
        """
        image_dir: directory con immagini
        processor: istanza di CLIPProcessor da Hugging Face
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
        
        # Usa il processor CLIP per ottenere pixel_values
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # Remove batch dim

        return pixel_values, img_path  # Ritorna tensor e percorso per tracciamento



# Define the transformations: resize, normalize, and convert to tensor
transform = transforms.Compose([
    # perform data augmentation: flip the image horizontally
    transforms.RandomHorizontalFlip(),
    # rotate the image by 45 degrees
    transforms.RandomRotation(45),
    # convert the image to a tensor
    transforms.ToTensor(),
    # reshape the tensor to have two dimensions
    transforms.Resize((224, 224)),  # Adjust to your image size
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pretrained weights normalization
])

from transformers import CLIPProcessor
from torch.utils.data import DataLoader

# Istanzia il processor di CLIP
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Crea istanze del dataset aggiornato
gallery_dataset = CLIPImageDataset(gallery_dir, processor=processor)
query_dataset = CLIPImageDataset(query_dir, processor=processor)

# Crea i DataLoader per caricare immagini in batch
gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False)
query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False)

class CLIPTripletDataset(Dataset):
    def __init__(self, root_dir, processor):
        """
        root_dir: directory con sottocartelle per ogni classe
        processor: CLIPProcessor da Hugging Face
        """
        self.dataset = datasets.ImageFolder(root_dir)
        self.processor = processor
        self.class_to_idx = self.dataset.class_to_idx
        self.imgs = self.dataset.imgs
        self.class_indices = {class_name: [] for class_name in self.class_to_idx}

        for idx, (img_path, class_idx) in enumerate(self.imgs):
            class_name = list(self.class_to_idx.keys())[list(self.class_to_idx.values()).index(class_idx)]
            self.class_indices[class_name].append(idx)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        def process_image(img_path):
            image = Image.open(img_path).convert("RGB")
            return self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        anchor_img_path, anchor_label = self.imgs[idx]
        anchor_image = process_image(anchor_img_path)

        # Positive: altra immagine della stessa classe
        positive_idx = random.choice(self.class_indices[
            list(self.class_to_idx.keys())[anchor_label]
        ])
        positive_img_path, _ = self.imgs[positive_idx]
        positive_image = process_image(positive_img_path)

        # Negative: immagine da classe diversa
        negative_class = random.choice(list(self.class_to_idx.keys()))
        while negative_class == list(self.class_to_idx.keys())[anchor_label]:
            negative_class = random.choice(list(self.class_to_idx.keys()))
        negative_idx = random.choice(self.class_indices[negative_class])
        negative_img_path, _ = self.imgs[negative_idx]
        negative_image = process_image(negative_img_path)

        return anchor_image, positive_image, negative_image

torch.seed()

# Define the transformation (resize, normalization, etc.)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Resize((224, 224)),  # Adjust to your image size
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pretrained weights normalization
])

# Init modello e processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model.to(device)  # solo parte visiva

# Dataset e DataLoader
train_dataset = CLIPTripletDataset(root_dir=train_dir, processor=processor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the TripletMarginLoss (you can adjust the margin parameter)
triplet_loss = nn.TripletMarginLoss(margin=0.000001, p=2)

# Set up the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.000001)
# optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for anchor, positive, negative in train_loader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()

        # Estrai embedding usando vision model (usa pooler_output)
        anchor_emb = model(pixel_values=anchor).pooler_output
        positive_emb = model(pixel_values=positive).pooler_output
        negative_emb = model(pixel_values=negative).pooler_output

        loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}")

model.eval()
with torch.no_grad():
    gallery_embeddings = []
    query_embeddings = []
    gallery_paths = []
    query_paths = []

    # Extract gallery embeddings
    for pixel_values, paths in gallery_loader:
        pixel_values = pixel_values.to(device)
        outputs = model(pixel_values=pixel_values)  # model can be CLIPModel or CLIPVisionModel
        emb = outputs.pooler_output  # (batch_size, hidden_dim)
        gallery_embeddings.append(emb.cpu().numpy())
        gallery_paths.extend(paths)

    # Extract query embeddings
    for pixel_values, paths in query_loader:
        pixel_values = pixel_values.to(device)
        outputs = model(pixel_values=pixel_values)
        emb = outputs.pooler_output
        query_embeddings.append(emb.cpu().numpy())
        query_paths.extend(paths)

# Stack all embedding batches into single numpy arrays
gallery_embeddings = np.vstack(gallery_embeddings)  # shape: (N_gallery, D)
query_embeddings = np.vstack(query_embeddings)      # shape: (N_query, D)

# Compute cosine similarity between each query and all gallery embeddings
similarity_matrix = cosine_similarity(query_embeddings, gallery_embeddings)

# For each query, find the index of the most similar gallery image
retrieved_indices = np.argmax(similarity_matrix, axis=1)

# Print top-1 retrieval results
print("Top-1 Retrieval Results:\n")
for i, idx in enumerate(retrieved_indices):
    print(f"Query image:    {query_paths[i]}")
    print(f"Retrieved image: {gallery_paths[idx]}")
    print("-" * 50)

top_k = 3
top_k_indices = np.argsort(similarity_matrix, axis=1)[:, -top_k:][:, ::-1]

for i, indices in enumerate(top_k_indices):
    print(f"Query image: {query_paths[i]}")
    print("Top-3 Retrieved gallery images:")
    for rank, idx in enumerate(indices, start=1):
        print(f"  {rank}. {gallery_paths[idx]}")
    print("-" * 50)

