"""Configuration parameters for measurement induced entanglement experiments."""

from typing import Dict, Any
import torch


# Device configuration
DEVICE_CONFIG = {
    'project_id': 'your-project-id',
    'processor_id': 'your-processor-id',
    'device_config_name': 'your-config-name',
    'snapshot_id': 'your-snapshot-id',
    'run_name': 'your-run-name'
}

# Experiment parameters
EXPERIMENT_CONFIG = {
    'folder_name': '6x6',
    'distances': [3, 4, 5],
    'theta_indices': list(range(11)),
    'num_loops': 17,
    'seed': 0
}

# Data processing parameters
PROCESSING_CONFIG = {
    'dtype': torch.complex128,
    'device': torch.device("cpu"),  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    'noise_parameter': 0.3
}

# Analysis parameters
ANALYSIS_CONFIG = {
    'phi': (5/4) * torch.pi,
    'theta_range': torch.linspace(0, torch.pi/2, 11)
}

# File paths
PATH_CONFIG = {
    'data_dir': 'data',
    'results_dir': 'results',
    'plots_dir': 'plots'
}


def get_config() -> Dict[str, Any]:
    """Get the complete configuration.
    
    Returns:
        Dictionary containing all configuration parameters.
    """
    return {
        'device': DEVICE_CONFIG,
        'experiment': EXPERIMENT_CONFIG,
        'processing': PROCESSING_CONFIG,
        'analysis': ANALYSIS_CONFIG,
        'paths': PATH_CONFIG
    } 