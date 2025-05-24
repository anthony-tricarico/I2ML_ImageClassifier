import torch
import torch.nn as nn
import torch.optim as optim


def prepare_model_for_training(model, finetune=False):
    """
    Prepare the ResNet model for training by either freezing or unfreezing layers.
    
    Args:
        model: The ResNet model to prepare
        finetune (bool): If True, all layers will be trainable. If False, only the final layer will be trainable.
    
    Returns:
        The prepared model
    """
    if not finetune:
        # Freeze all layers except the final fully connected layer
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the final fully connected layer
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        # Unfreeze all layers for finetuning
        for param in model.parameters():
            param.requires_grad = True
    
    return model