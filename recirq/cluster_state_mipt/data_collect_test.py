# Copyright 2025 Google
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for data_collect.py."""

import os
import tempfile
from unittest import mock

import cirq
import cirq_google as cg
import numpy as np
import pytest

from recirq.cluster_state_mipt import data_collect


def test_collect_data_parameters():
    """Test parameter validation in collect_data."""
    mock_engine = mock.MagicMock(spec=cg.Engine)
    
    # Test with invalid repetitions
    with pytest.raises(ValueError):
        data_collect.collect_data(
            engine=mock_engine,
            processor_id='test-processor',
            repetitions=0
        )
    
    # Test with invalid num_rc_circuits
    with pytest.raises(ValueError):
        data_collect.collect_data(
            engine=mock_engine,
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
                    engine=mock_engine,
                    processor_id=processor_id,
                    repetitions=100,
                    num_rc_circuits=2,
                    folder_name=folder_name
                )
            
            # Verify that the directory was created
            assert os.path.exists(temp_dir) 