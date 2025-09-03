from typing import Dict, Optional
import torch
from tqdm import tqdm

from kvmm.utils.custom_exception import WeightMappingError, WeightShapeMismatchError
from kvmm.utils.weight_split_torch_and_keras import split_model_weights
from kvmm.utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_weights,
)


def transfer_torch_to_keras_weights(
    torch_model: torch.nn.Module,
    keras_model,  # Keras model type
    weight_name_mapping: Optional[Dict[str, str]] = None,
    save_weights: bool = True,
    output_filename: Optional[str] = None,
    show_progress: bool = True
) -> None:
    """
    Transfer weights from a PyTorch model to a Keras model.
    
    Args:
        torch_model: The source PyTorch model
        keras_model: The target Keras model
        weight_name_mapping: Dictionary mapping Keras weight names to PyTorch weight names
        save_weights: Whether to save the transferred weights to file
        output_filename: Custom filename for saved weights (if None, uses model name)
        show_progress: Whether to show progress bar during transfer
    
    Raises:
        WeightMappingError: If a Keras weight name cannot be mapped to a PyTorch weight
        WeightShapeMismatchError: If weight shapes don't match between models
    """
    
    if weight_name_mapping is None:
        weight_name_mapping = {
            "_": ".",
            "c2f.block": "model.model",
            "conv.block": "model.model",
            "sppf.block": "model.model",
            "c3.block": "model.model",
            "detect.head": "model.model",
            "cv2.head.conv": "cv2",
            "cv3.head.conv": "cv3",
            "batchnorm": "bn",
            "bias": "bias",
            "kernel": "weight",
            "gamma": "weight",
            "beta": "bias",
            "moving.mean": "running_mean",
            "moving.variance": "running_var",
        }
    
    torch_model.eval()
    
    trainable_torch_weights, non_trainable_torch_weights, _ = split_model_weights(
        torch_model
    )
    trainable_keras_weights, non_trainable_keras_weights = split_model_weights(
        keras_model
    )
    
    torch_weights_dict: Dict[str, torch.Tensor] = {
        **trainable_torch_weights,
        **non_trainable_torch_weights,
    }
    
    all_keras_weights = trainable_keras_weights + non_trainable_keras_weights
    
    iterator = tqdm(
        all_keras_weights,
        total=len(all_keras_weights),
        desc="Transferring weights",
        disable=not show_progress
    )
    
    for keras_weight, keras_weight_name in iterator:
        torch_weight_name: str = keras_weight_name
        for keras_name_part, torch_name_part in weight_name_mapping.items():
            torch_weight_name = torch_weight_name.replace(keras_name_part, torch_name_part)
        
        if torch_weight_name not in torch_weights_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)
        
        torch_weight: torch.Tensor = torch_weights_dict[torch_weight_name]
        
        if not compare_keras_torch_names(
            keras_weight_name, keras_weight, torch_weight_name, torch_weight
        ):
            raise WeightShapeMismatchError(
                keras_weight_name, keras_weight.shape, 
                torch_weight_name, torch_weight.shape
            )
        
        transfer_weights(keras_weight_name, keras_weight, torch_weight)
    
    if save_weights:
        if output_filename is None:
            model_name = getattr(torch_model, 'model_name', 'model')
            if hasattr(model_name, 'split'):
                model_name = model_name.split(".")[0]
            model_filename = f"{model_name}.weights.h5"
        else:
            model_filename = output_filename
            
        keras_model.save_weights(model_filename)
        print(f"Model saved successfully as {model_filename}")