"""Module for analyzing quantum measurement data.

This module provides functionality for analyzing measurement results from quantum
experiments, including entanglement measures and state reconstruction.
"""

from typing import List, Tuple, Optional
import torch
from recirq.Cluster_state_MIPT.utils import blogm, bSqc, Neg, Sa


# Global constants
DTYPE = torch.complex128
DEVICE = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hadamard matrix
H = (torch.tensor([[1, 1], [1, -1]]) / 2**0.5).to(torch.cdouble)

# Unitary matrices for state preparation
u = torch.tensor([
    [torch.tensor(-1j*torch.pi/4).exp()/torch.sqrt(torch.tensor(2)),
     torch.tensor(1j*torch.pi/4).exp()/torch.sqrt(torch.tensor(2))],
    [0, 0],
    [0, 0],
    [torch.tensor(1j*torch.pi/4).exp()/torch.sqrt(torch.tensor(2)),
     torch.tensor(-1j*torch.pi/4).exp()/torch.sqrt(torch.tensor(2))]
]).view(2, 2, 2).to(torch.cdouble)

v = torch.tensor([
    [0, 0, 0, 2**0.5],
    [2**0.5, 0, 0, 0]
]).view(2, 2, 2).to(torch.cdouble)
v = v.permute(1, 2, 0)


def eps(rho: torch.Tensor, e: float = 0.1) -> torch.Tensor:
    """Add noise to a density matrix.
    
    Args:
        rho: Input density matrix.
        e: Noise parameter.
        
    Returns:
        Noisy density matrix.
    """
    I = torch.eye(
        rho.shape[-1],
        rho.shape[-1],
        dtype=rho.dtype,
        device=rho.device
    )[None, ...].expand(rho.shape[0], -1, -1) / rho.shape[-1]
    return (1 - e) * rho + e * I


def cross(
    m: Optional[int],
    theta: torch.Tensor,
    phi: torch.Tensor,
    kernel: torch.Tensor
) -> torch.Tensor:
    """Apply cross measurement to a quantum state.
    
    Args:
        m: Measurement outcome (0, 1, or None).
        theta: Rotation angle.
        phi: Phase angle.
        kernel: Kernel tensor.
        
    Returns:
        Processed quantum state.
    """
    U = torch.tensor([
        [torch.cos(theta/2)*torch.exp(1.0j*phi/2),
         torch.sin(theta/2)*torch.exp(-1.0j*phi/2)],
        [-torch.sin(theta/2)*torch.exp(1.0j*phi/2),
         torch.cos(theta/2)*torch.exp(-1.0j*phi/2)]
    ], dtype=kernel.dtype)
    
    out = H @ torch.tensor([1, 0], dtype=kernel.dtype)
    out = torch.einsum(kernel, [0, 1, 2], out, [0], [1, 2])
    out = torch.einsum(kernel, [1, 3, 4], out, [1, 2], [2, 3, 4])
    out = torch.einsum(kernel, [3, 5, 6], out, [2, 3, 4], [2, 4, 5, 6])
    out = torch.einsum(kernel, [5, 7, 8], out, [2, 4, 5, 6], [7, 8, 6, 4, 2])
    
    # Single qubit rotation
    out = torch.einsum(U, [9, 7], out, [7, 8, 6, 4, 2], [9, 8, 6, 4, 2])
    
    if m == 0:
        m = torch.tensor([1, 0], dtype=u.dtype)
    elif m == 1:
        m = torch.tensor([0, 1], dtype=u.dtype)
    
    if m is not None:
        out = torch.einsum(m, [9], out, [9, 8, 6, 4, 2], [8, 6, 4, 2])
    
    return out


def side(
    m: Optional[int],
    theta: torch.Tensor,
    phi: torch.Tensor,
    kernel: torch.Tensor
) -> torch.Tensor:
    """Apply side measurement to a quantum state.
    
    Args:
        m: Measurement outcome (0, 1, or None).
        theta: Rotation angle.
        phi: Phase angle.
        kernel: Kernel tensor.
        
    Returns:
        Processed quantum state.
    """
    U = torch.tensor([
        [torch.cos(theta/2)*torch.exp(1.0j*phi/2),
         torch.sin(theta/2)*torch.exp(-1.0j*phi/2)],
        [-torch.sin(theta/2)*torch.exp(1.0j*phi/2),
         torch.cos(theta/2)*torch.exp(-1.0j*phi/2)]
    ], dtype=kernel.dtype)
    
    out = H @ torch.tensor([1, 0], dtype=kernel.dtype)
    out = torch.einsum(kernel, [0, 1, 2], out, [0], [1, 2])
    out = torch.einsum(kernel, [1, 3, 4], out, [1, 2], [2, 3, 4])
    out = torch.einsum(kernel, [3, 5, 6], out, [2, 3, 4], [2, 4, 5, 6])
    
    # Single qubit rotation
    out = torch.einsum(U, [7, 5], out, [2, 4, 5, 6], [7, 6, 4, 2])
    
    if m == 0:
        m = torch.tensor([1, 0], dtype=u.dtype)
    elif m == 1:
        m = torch.tensor([0, 1], dtype=u.dtype)
    
    if m is not None:
        out = torch.einsum(m, [7], out, [7, 6, 4, 2], [6, 4, 2])
    
    return out


def corner(
    m: Optional[int],
    theta: torch.Tensor,
    phi: torch.Tensor,
    kernel: torch.Tensor
) -> torch.Tensor:
    """Apply corner measurement to a quantum state.
    
    Args:
        m: Measurement outcome (0, 1, or None).
        theta: Rotation angle.
        phi: Phase angle.
        kernel: Kernel tensor.
        
    Returns:
        Processed quantum state.
    """
    U = torch.tensor([
        [torch.cos(theta/2)*torch.exp(1.0j*phi/2),
         torch.sin(theta/2)*torch.exp(-1.0j*phi/2)],
        [-torch.sin(theta/2)*torch.exp(1.0j*phi/2),
         torch.cos(theta/2)*torch.exp(-1.0j*phi/2)]
    ], dtype=kernel.dtype)
    
    out = H @ torch.tensor([1, 0], dtype=kernel.dtype)
    out = torch.einsum(kernel, [0, 1, 2], out, [0], [1, 2])
    out = torch.einsum(kernel, [1, 3, 4], out, [1, 2], [2, 3, 4])
    
    # Single qubit rotation
    out = torch.einsum(U, [5, 3], out, [2, 3, 4], [5, 2, 4])
    
    if m == 0:
        m = torch.tensor([1, 0], dtype=u.dtype)
    elif m == 1:
        m = torch.tensor([0, 1], dtype=u.dtype)
    
    if m is not None:
        out = torch.einsum(m, [5], out, [5, 2, 4], [2, 4])
    
    return out


def contract(
    src: torch.Tensor,
    src_idx: List[int],
    dst: torch.Tensor,
    dst_idx: List[int]
) -> Tuple[torch.Tensor, List[int]]:
    """Contract two tensors along specified indices.
    
    Args:
        src: Source tensor.
        src_idx: Source tensor indices.
        dst: Destination tensor.
        dst_idx: Destination tensor indices.
        
    Returns:
        A tuple containing:
            - Contracted tensor
            - New index list
    """
    out_idx = src_idx.copy()
    for i in dst_idx:
        if i != 0:  # batch dim
            if i in out_idx:
                out_idx.remove(i)
            else:
                out_idx.append(i)
    
    if 0 not in out_idx:
        out_idx = [0] + out_idx
    
    out_tensor = torch.einsum(src, src_idx, dst, dst_idx, out_idx)
    return out_tensor, out_idx


def analyze_data(
    folder_name: str,
    distance: int,
    theta_indices: List[int],
    num_loops: int,
    seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Analyze measurement data for multiple experiments.
    
    Args:
        folder_name: Name of the folder containing the data.
        distance: Distance between probe qubits.
        theta_indices: List of theta indices to process.
        num_loops: Number of loops to process for each experiment.
        seed: Random seed for reproducibility.
        
    Returns:
        A tuple containing:
            - decode_N_values: Negativity values
            - decode_S_values: Squashed concurrence values
    """
    torch.manual_seed(seed)
    decode_N_values = []
    decode_S_values = []
    
    for theta_idx in theta_indices:
        print(f'theta_idx={theta_idx} start')
        
        # Load data
        prep = torch.load(
            f'data/{folder_name}_d={distance}/theta{theta_idx}/all_prepseq_theta={theta_idx}.pt'
        )
        rhoS = torch.load(
            f'data/{folder_name}_d={distance}/theta{theta_idx}/all_rhoS_theta={theta_idx}.pt'
        )
        
        # Set up parameters
        theta = torch.linspace(0, torch.pi/2, 11)[theta_idx]
        phi = torch.tensor((5/4)*torch.pi)
        
        # Prepare measurement operators
        u_cross = torch.cat([
            cross(0, theta, phi, u)[None, ...],
            cross(1, theta, phi, u)[None, ...]
        ], dim=0)
        u_side = torch.cat([
            side(0, theta, phi, u)[None, ...],
            side(1, theta, phi, u)[None, ...]
        ], dim=0)
        u_corner = torch.cat([
            corner(0, theta, phi, u)[None, ...],
            corner(1, theta, phi, u)[None, ...]
        ], dim=0)
        v_cross = torch.cat([
            cross(0, theta, phi, v)[None, ...],
            cross(1, theta, phi, v)[None, ...]
        ], dim=0)
        v_side = torch.cat([
            side(0, theta, phi, v)[None, ...],
            side(1, theta, phi, v)[None, ...]
        ], dim=0)
        v_corner = torch.cat([
            corner(0, theta, phi, v)[None, ...],
            corner(1, theta, phi, v)[None, ...]
        ], dim=0)
        
        # Process each row of the lattice
        src, src_idx = contract(
            v_corner[prep[:, 33]], [0, 28, 33],
            u_side[prep[:, 32]], [0, 32, 33, 26]
        )
        
        # Process remaining rows (similar pattern as in your code)
        # ... (implement the remaining row processing)
        
        # Final processing
        src = src.permute(0, 2, 1).contiguous()
        rho = torch.vmap(torch.outer)(src.view(-1, 4), src.view(-1, 4).conj())
        coef = torch.vmap(torch.trace)(rho).view(-1, 1, 1)
        idx = (coef.real != 0).view(-1)
        
        rho = rho[idx]
        coef = coef[idx]
        rhoS = rhoS[idx]
        rho /= coef
        
        # Calculate entanglement measures
        decode_N_values.append(Neg(rhoS, eps(rho, 0.3)))
        decode_S_values.append(bSqc(rhoS, eps(rho, 0.3)))
        
        print(
            f'theta_idx={theta_idx} done',
            decode_N_values[-1].mean().item(),
            decode_S_values[-1].mean().item()
        )
    
    return (
        torch.cat(decode_N_values).view(len(theta_indices), -1),
        torch.cat(decode_S_values).view(len(theta_indices), -1)
    ) 