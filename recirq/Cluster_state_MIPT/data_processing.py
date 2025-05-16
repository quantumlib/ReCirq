"""Module for post-processing quantum measurement data.

This module provides functionality for processing measurement results from quantum
experiments, including shadow state reconstruction and data shuffling.
"""

from typing import Dict, List, Tuple
import os
import torch


# Global constants
DTYPE = torch.complex128
DEVICE = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pauli matrices and measurement bases
PAULI = torch.tensor([
    [[1, 0], [0, 1]],
    [[0, 1], [1, 0]],
    [[0, -1j], [1j, 0]],
    [[1, 0], [0, -1]]
], device=DEVICE, dtype=DTYPE)
BASIS = torch.linalg.eig(PAULI)[1][1:].mT  # (3, 2, 2)


def process_measurements(
    filename: str,
    distance: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Process measurement results from a quantum experiment.
    
    Args:
        filename: Base filename for the measurement data.
        distance: Distance between probe qubits.
        
    Returns:
        A tuple containing:
            - prepseq: Preparation sequence tensor
            - shadow_state: Shadow state tensor
            - rhoS: Density matrix tensor
    """
    data: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
    
    # Process measurements for each basis combination
    for i in range(3):
        for j in range(3):
            # Load measurements
            m = torch.load(f"{filename}_({i},{j}).pt")
            msk = torch.ones(m.shape[0], device=DEVICE, dtype=torch.bool)
            
            # Post-select on 2-qubit mitigation
            anc_phy_pairs = [
                (0, 7), (1, 8), (2, 9), (3, 10), (4, 11), (5, 12),
                (54, 47), (55, 48), (56, 49), (57, 50), (58, 51), (59, 52),
                (6, 7), (14, 15), (22, 23), (30, 31), (38, 39), (46, 47),
                (13, 12), (21, 20), (29, 28), (37, 36), (45, 44), (53, 52)
            ]
            for anc, phy in anc_phy_pairs:
                msk = msk & (m[:, anc] == m[:, phy])
            
            # Get preparation indices
            prep_idx = [
                7, 8, 9, 10, 11, 12,
                15, 16, 17, 18, 19, 20,
                23, 24, 25, 26, 27, 28,
                31, 32, 33, 34, 35, 36,
                39, 40, 41, 42, 43, 44,
                47, 48, 49, 50, 51, 52
            ]
            
            # Get probe indices based on distance
            if distance == 6:
                probe_idx = [7, 12]
            elif distance == 5:
                probe_idx = [7, 11]
            elif distance == 4:
                probe_idx = [8, 11]
            elif distance == 3:
                probe_idx = [8, 10]
            else:
                raise ValueError(f"Invalid distance: {distance}")
            
            # Remove probe indices from preparation indices
            prep_idx = [p for p in prep_idx if p not in probe_idx]
            
            # Process measurements
            m = m[msk]  # (batch, num_qubits)
            probe = torch.cat([
                m[:, probe_idx[0]].view(-1, 1),
                m[:, probe_idx[1]].view(-1, 1)
            ], 1)
            prep = m[:, prep_idx]
            data[(i, j)] = (prep, probe)
    
    # Construct shadow states and density matrices
    prepseq, shadow_state, rhoS = [], [], []
    for k in data.keys():
        # Get measurement outcomes
        probseq = data[k][1].to(dtype=torch.int64).to(device=DEVICE)
        
        # Construct post-measurement states
        obs_basis0 = BASIS[k[0]].unsqueeze(0).expand(probseq.shape[0], -1, -1)
        shadow_state0 = obs_basis0.gather(
            1, probseq[:, 0].view(-1, 1, 1).expand(-1, -1, 2)
        ).squeeze(1)
        
        obs_basis1 = BASIS[k[1]].unsqueeze(0).expand(probseq.shape[0], -1, -1)
        shadow_state1 = obs_basis1.gather(
            1, probseq[:, 1].view(-1, 1, 1).expand(-1, -1, 2)
        ).squeeze(1)
        
        shadow_state01 = torch.vmap(torch.kron)(shadow_state0, shadow_state1)
        
        # Construct density matrices
        I = torch.eye(2, 2, device=DEVICE)[None, ...].expand(
            shadow_state01.shape[0], -1, -1
        )
        rhoS0 = 3 * torch.vmap(torch.outer)(
            shadow_state0, shadow_state0.conj()
        ) - I
        rhoS1 = 3 * torch.vmap(torch.outer)(
            shadow_state1, shadow_state1.conj()
        ) - I
        rhoS01 = torch.vmap(torch.kron)(rhoS0, rhoS1)
        
        # Collect results
        prepseq.append(data[k][0].to(dtype=torch.int64).to(device=DEVICE))
        shadow_state.append(shadow_state01)
        rhoS.append(rhoS01)
    
    # Concatenate results
    prepseq = torch.cat(prepseq, 0).to(torch.int64)
    shadow_state = torch.cat(shadow_state, 0)
    rhoS = torch.cat(rhoS, 0)
    
    return prepseq, shadow_state, rhoS


def shuffle_data(
    prepseq: torch.Tensor,
    shadow_state: torch.Tensor,
    rhoS: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shuffle the processed data.
    
    Args:
        prepseq: Preparation sequence tensor.
        shadow_state: Shadow state tensor.
        rhoS: Density matrix tensor.
        
    Returns:
        A tuple containing the shuffled tensors in the same order.
    """
    indices = torch.randperm(prepseq.shape[0])
    return (
        prepseq[indices],
        shadow_state[indices],
        rhoS[indices]
    )


def process_all_data(
    folder_name: str,
    distances: List[int],
    theta_indices: List[int],
    num_loops: int,
    seed: int = 0
) -> None:
    """Process all measurement data for multiple experiments.
    
    Args:
        folder_name: Name of the folder containing the data.
        distances: List of distances to process.
        theta_indices: List of theta indices to process.
        num_loops: Number of loops to process for each experiment.
        seed: Random seed for shuffling.
    """
    torch.manual_seed(seed)
    
    for theta_idx in theta_indices:
        for d in distances:
            all_prepseq = []
            all_shadow_state = []
            all_rhoS = []
            
            for loop in range(num_loops):
                filename = f'data/{folder_name}_d={d}/theta{theta_idx}/loop{loop}/theta={theta_idx}'
                prepseq, shadow_state, rhoS = process_measurements(filename, d)
                prepseq, shadow_state, rhoS = shuffle_data(prepseq, shadow_state, rhoS)
                
                all_prepseq.append(prepseq)
                all_shadow_state.append(shadow_state)
                all_rhoS.append(rhoS)
                
                print(
                    f'distance={d}, loop={loop}, theta_idx={theta_idx}, '
                    f'portion to keep={((prepseq.shape[0]/9000000)):.4f}'
                )
            
            # Concatenate and save results
            all_prepseq = torch.cat(all_prepseq, 0)
            all_shadow_state = torch.cat(all_shadow_state, 0)
            all_rhoS = torch.cat(all_rhoS, 0)
            
            # Create output directory if it doesn't exist
            output_dir = f'data/{folder_name}_d={d}/theta{theta_idx}'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save processed data
            torch.save(
                all_prepseq,
                f'{output_dir}/all_prepseq_theta={theta_idx}.pt'
            )
            torch.save(
                all_shadow_state,
                f'{output_dir}/all_shadow_state_theta={theta_idx}.pt'
            )
            torch.save(
                all_rhoS,
                f'{output_dir}/all_rhoS_theta={theta_idx}.pt'
            )
            
            print(f"Processed data shapes for theta_idx={theta_idx}:")
            print(f"prepseq: {all_prepseq.shape}")
            print(f"shadow_state: {all_shadow_state.shape}")
            print(f"rhoS: {all_rhoS.shape}") 