"""
Eigenvalue and Eigenvector Analysis for Neural Networks
=======================================================

Computes and tracks the spectral properties of weight matrices
during training, including SVD, condition numbers, effective rank, 
and 1D loss landscape slices.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, List


def compute_weight_svd(model) -> Dict[str, dict]:
    """
    Compute SVD for all 2D weight matrices in the model.
    
    Returns a dict mapping layer name -> {
        'singular_values': np.ndarray,
        'U': np.ndarray,
        'Vt': np.ndarray,
        'condition_number': float,
        'effective_rank': float,
        'shape': tuple,
    }
    """
    results = {}
    for name, param in model.named_parameters():
        if param.dim() >= 2 and "weight" in name and param.requires_grad:
            W = param.detach().cpu().numpy()
            
            # Flatten Conv1D/Conv2D weights to 2D for SVD
            if W.ndim > 2:
                W = W.reshape(W.shape[0], -1)
                
            if W.shape[0] < 2 or W.shape[1] < 2:
                continue
                
            try:
                U, S, Vt = np.linalg.svd(W, full_matrices=False)
                results[name] = {
                    'singular_values': S.astype(np.float32),
                    'U': U.astype(np.float32),
                    'Vt': Vt.astype(np.float32),
                    'condition_number': float(S[0] / max(S[-1], 1e-10)),
                    'effective_rank': float(np.sum(S > S[0] * 0.01)),
                    'shape': W.shape,
                }
            except np.linalg.LinAlgError:
                continue
                
    return results


def compute_eigen_arrows(svd_results: Dict[str, dict], input_dim: int = 2) -> List[dict]:
    """
    Extract 2D arrows from right singular vectors for visualization.
    
    For the first layer with 2D input, the right singular vectors
    represent the most important directions in input space.
    """
    arrows = []
    for name, data in svd_results.items():
        Vt = data['Vt']
        S = data['singular_values']
        shape = data['shape']
        
        # Check if this layer maps from a 2D input
        if shape[1] == input_dim and len(Vt) > 0:
            k = min(3, len(Vt))
            vectors = Vt[:k] * S[:k, None]  # Scale by singular value
            arrows.append({
                'name': name,
                'vectors': vectors,
                'singular_values': S[:k],
            })
            
    return arrows


def compute_loss_landscape_slice(
    model, 
    criterion, 
    X: torch.Tensor, 
    y: torch.Tensor,
    layer_name: str, 
    direction: np.ndarray, 
    n_points: int = 31, 
    scale: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute loss when perturbing a weight matrix along a direction.
    
    Parameters
    ----------
    model : nn.Module
        The neural network model.
    criterion : loss function
        The loss criterion.
    X : torch.Tensor
        Input data (small batch for speed).
    y : torch.Tensor
        Target data.
    layer_name : str
        Name of the parameter to perturb.
    direction : np.ndarray
        Direction in weight space (same shape as the parameter).
    n_points : int
        Number of points along the perturbation axis.
    scale : float
        Perturbation range in units of the direction norm.

    Returns
    -------
    offsets : np.ndarray of shape (n_points,)
        Perturbation magnitudes (alpha values).
    losses : np.ndarray of shape (n_points,)
        Loss values at each perturbation.
    """
    target_param = None
    for name, param in model.named_parameters():
        if name == layer_name:
            target_param = param
            break

    if target_param is None:
        return np.array([]), np.array([])

    original = target_param.data.clone()
    direction_t = torch.FloatTensor(direction.copy())

    # Normalize direction so that ||direction|| = ||W|| * scale
    w_norm = float(original.norm())
    d_norm = float(direction_t.norm())
    if d_norm > 1e-10:
        direction_t = direction_t * (w_norm / d_norm)

    offsets = np.linspace(-scale, scale, n_points)
    losses = []

    was_training = model.training
    model.eval()

    with torch.no_grad():
        for alpha in offsets:
            target_param.data = original + alpha * direction_t
            try:
                loss = criterion(model(X), y)
                losses.append(float(loss))
            except Exception:
                losses.append(float('nan'))

    target_param.data = original  # Restore original weights
    if was_training:
        model.train()

    return offsets, np.array(losses)