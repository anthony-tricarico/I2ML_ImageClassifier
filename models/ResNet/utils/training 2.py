import torch
import torch.nn as nn
import torch.optim as optim


def prepare_model_for_training(model, finetune=False, trainable_layers=None):
    """
    Prepare a model for training by either freezing or unfreezing layers.
    
    Args:
        model: The model to prepare
        finetune (bool): If True, all layers will be trainable. If False, only specified layers will be trainable.
        trainable_layers (list): List of layer names to keep trainable when finetune=False.
                                If None, will try to automatically detect the final classification layer.
    
    Returns:
        The prepared model

    Examples:
        model = prepare_model_for_training(model, finetune=False)  # Will automatically detect and unfreeze the 'fc' layer

        For CLIP-like models, use:
        model = prepare_model_for_training(model, finetune=False, trainable_layers=['visual.proj', 'text.proj'])  # Specify the projection layers
        
        For full finetuning, use:
        model = prepare_model_for_training(model, finetune=True)  # Unfreezes all layers


    """
    if not finetune:
        # First freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
            
        if trainable_layers is None:
            # Try to automatically detect the final classification layer
            # Common names for classification layers in different architectures
            possible_layer_names = ['fc', 'classifier', 'head', 'proj', 'projection']
            
            for layer_name in possible_layer_names:
                if hasattr(model, layer_name):
                    # Unfreeze the detected layer
                    for param in getattr(model, layer_name).parameters():
                        param.requires_grad = True
                    break
        else:
            # Unfreeze only specified layers
            for layer_name in trainable_layers:
                if hasattr(model, layer_name):
                    for param in getattr(model, layer_name).parameters():
                        param.requires_grad = True
    else:
        # Unfreeze all layers for finetuning
        for param in model.parameters():
            param.requires_grad = True
    
    return model