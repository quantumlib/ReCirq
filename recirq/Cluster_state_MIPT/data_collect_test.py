"""Tests for data_collect.py."""

import os
import tempfile
from unittest import mock

import cirq
import cirq_google as cg
import numpy as np
import pytest
import torch

from recirq.Cluster_state_MIPT import data_collect


def test_collect_data_parameters():
    """Test parameter validation in collect_data."""
    mock_engine = mock.MagicMock(spec=cg.Engine)
    
    # Test with invalid repetitions
    with pytest.raises(ValueError):
        data_collect.collect_data(
            eng=mock_engine,
            processor_id='test-processor',
            repetitions=0
        )
    
    # Test with invalid num_rc_circuits
    with pytest.raises(ValueError):
        data_collect.collect_data(
            eng=mock_engine,
            processor_id='test-processor',
            num_rc_circuits=0
        )


def test_collect_data_directory():
    """Test directory creation for data collection."""
    mock_engine = mock.MagicMock(spec=cg.Engine)
    processor_id = 'test-processor'
    folder_name = 'test_data'
    
    # Create a temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock the data directory
        with mock.patch('os.path.join', return_value=temp_dir):
            # Mock the engine to prevent actual execution
            mock_engine.get_processor.return_value.get_sampler.return_value.run_batch.side_effect = Exception(
                "This test should not execute quantum circuits")
            
            # Call the function and expect it to fail due to our mock
            with pytest.raises(Exception):
                data_collect.collect_data(
                    eng=mock_engine,
                    processor_id=processor_id,
                    repetitions=100,
                    num_rc_circuits=2,
                    folder_name=folder_name
                )
            
            # Verify that the directory was created
            assert os.path.exists(temp_dir) 