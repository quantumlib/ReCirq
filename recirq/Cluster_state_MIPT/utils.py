"""Utility functions for quantum information theory calculations.

This module provides functions for calculating various quantum information theory
measures such as negativity, squashed concurrence, and von Neumann entropy.
"""

from typing import Tuple
import torch


# Global constants
PAULI_MATRICES = None  # Will be initialized in setup_pauli_matrices()
EPSILON = 1e-5  # Small constant for numerical stability


def setup_pauli_matrices(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Initialize Pauli matrices and compute measurement basis.
    
    Args:
        device: Device to store tensors on.
        dtype: Data type for tensors.
        
    Returns:
        Tensor of shape (3, 2, 2) containing the measurement basis.
    """
    global PAULI_MATRICES
    pauli = torch.tensor([[[1,0],[0,1]],[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]], 
                        device=device, dtype=dtype)
    PAULI_MATRICES = pauli
    return torch.linalg.eig(pauli)[1][1:].mT  # (3, 2, 2)


def eps(rho: torch.Tensor, e: float = 0.1) -> torch.Tensor:
    """Add noise to a density matrix.
    
    Args:
        rho: Input density matrix of shape (batch_size, dim, dim).
        e: Noise parameter between 0 and 1.
        
    Returns:
        Noisy density matrix of same shape as input.
        
    Raises:
        ValueError: If e is not between 0 and 1.
        ValueError: If rho is not a valid density matrix.
    """
    if not 0 <= e <= 1:
        raise ValueError(f"Noise parameter e must be between 0 and 1, got {e}")
    
    # Check if rho is a valid density matrix
    if not torch.allclose(torch.trace(rho[0]), torch.tensor(1.0, dtype=rho.dtype, device=rho.device)):
        raise ValueError("Input matrix must have trace 1")
    if not torch.allclose(rho[0].conj().T, rho[0]):
        raise ValueError("Input matrix must be Hermitian")
    
    I = torch.eye(rho.shape[-1], rho.shape[-1], dtype=rho.dtype, device=rho.device)[None,...].expand(rho.shape[0], -1, -1)/rho.shape[-1]
    return (1-e)*rho + e*I


def blogm(A: torch.Tensor) -> torch.Tensor:
    """Calculate the matrix logarithm of a density matrix.
    
    Args:
        A: Input matrix of shape (batch_size, dim, dim).
        
    Returns:
        Matrix logarithm of same shape as input.
        
    Raises:
        ValueError: If A is not a valid density matrix.
    """
    # Check if A is a valid density matrix
    if not torch.allclose(torch.trace(A[0]), torch.tensor(1.0, dtype=A.dtype, device=A.device)):
        raise ValueError("Input matrix must have trace 1")
    if not torch.allclose(A[0].conj().T, A[0]):
        raise ValueError("Input matrix must be Hermitian")
    
    E, U = torch.linalg.eig(A)
    E = E + EPSILON  # Add small constant for numerical stability
    logE = torch.log(E.abs()).to(U.dtype)
    logA = torch.bmm(torch.bmm(U, torch.diag_embed(logE, offset=0, dim1=-2, dim2=-1)), U.conj().mT)
    return logA


def bSqc(rhoQ: torch.Tensor, rhoC: torch.Tensor) -> torch.Tensor:
    """Calculate the squashed concurrence between two states.
    
    Args:
        rhoQ: First density matrix of shape (batch_size, dim, dim).
        rhoC: Second density matrix of shape (batch_size, dim, dim).
        
    Returns:
        Squashed concurrence value of shape (batch_size,).
        
    Raises:
        ValueError: If input matrices have different devices or dtypes.
        ValueError: If input matrices are not valid density matrices.
    """
    if rhoQ.device != rhoC.device:
        raise ValueError(f"Input matrices must be on same device, got {rhoQ.device} and {rhoC.device}")
    if rhoQ.dtype != rhoC.dtype:
        raise ValueError(f"Input matrices must have same dtype, got {rhoQ.dtype} and {rhoC.dtype}")
    
    return -torch.vmap(torch.trace)(rhoQ@blogm(rhoC)).real


def Neg(rhoS: torch.Tensor, rhoC: torch.Tensor) -> torch.Tensor:
    """Calculate the negativity between two states.
    
    Args:
        rhoS: First density matrix of shape (batch_size, 4, 4).
        rhoC: Second density matrix of shape (batch_size, 4, 4).
        
    Returns:
        Negativity value of shape (batch_size,).
        
    Raises:
        ValueError: If input matrices are not 4x4.
        ValueError: If input matrices have different devices or dtypes.
    """
    if rhoS.shape[-2:] != (4, 4) or rhoC.shape[-2:] != (4, 4):
        raise ValueError("Input matrices must be 4x4")
    if rhoS.device != rhoC.device:
        raise ValueError(f"Input matrices must be on same device, got {rhoS.device} and {rhoC.device}")
    if rhoS.dtype != rhoC.dtype:
        raise ValueError(f"Input matrices must have same dtype, got {rhoS.dtype} and {rhoC.dtype}")
    
    rhoC_pt = rhoC.view(-1,2,2,2,2).permute(0,1,4,3,2).reshape(-1,4,4)
    rhoS_pt = rhoS.view(-1,2,2,2,2).permute(0,1,4,3,2).reshape(-1,4,4)
    e, v = torch.linalg.eig(rhoC_pt)
    e = e + EPSILON  # Add small constant for numerical stability
    mask = e.real < 0
    negative_v = v * mask.unsqueeze(1)
    P = torch.bmm(negative_v, negative_v.mT.conj())  # projection matrix
    return -torch.vmap(torch.trace)(torch.bmm(P, rhoS_pt)).real


def Sa(rhoS: torch.Tensor, rhoC: torch.Tensor) -> torch.Tensor:
    """Calculate the von Neumann entropy of a state.
    
    Args:
        rhoS: First density matrix of shape (batch_size, 4, 4).
        rhoC: Second density matrix of shape (batch_size, 4, 4).
        
    Returns:
        Von Neumann entropy value of shape (batch_size,).
        
    Raises:
        ValueError: If input matrices are not 4x4.
        ValueError: If input matrices have different devices or dtypes.
    """
    if rhoS.shape[-2:] != (4, 4) or rhoC.shape[-2:] != (4, 4):
        raise ValueError("Input matrices must be 4x4")
    if rhoS.device != rhoC.device:
        raise ValueError(f"Input matrices must be on same device, got {rhoS.device} and {rhoC.device}")
    if rhoS.dtype != rhoC.dtype:
        raise ValueError(f"Input matrices must have same dtype, got {rhoS.dtype} and {rhoC.dtype}")
    
    rhoCa = torch.einsum('bijkj->bik', rhoC.view(-1,2,2,2,2))
    rhoSa = torch.einsum('bijkj->bik', rhoS.view(-1,2,2,2,2))
    return bSqc(rhoSa, rhoCa) 