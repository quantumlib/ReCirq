# Copyright 2020 Google
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
import os
import io
import tarfile
from unittest import mock

import pytest

import recirq
from recirq.readout_scan.tasks import ReadoutScanTask


def test_display_markdown_docstring():
    md = recirq.display_markdown_docstring(ReadoutScanTask)
    assert md.data == """### ReadoutScanTask
Scan over Ry(theta) angles from -pi/2 to 3pi/2 tracing out a sinusoid
which is primarily affected by readout error.

#### See Also
`run_readout_scan` 

#### Attributes
 - `dataset_id`: A unique identifier for this dataset.
 - `device_name`: The device to run on, by name.
 - `n_shots`: The number of repetitions for each theta value.
 - `qubit`: The qubit to benchmark.
 - `resolution_factor`: We select the number of points in the linspace so that the special points: (-1/2, 0, 1/2, 1, 3/2) * pi are always included. The total number of theta evaluations is resolution_factor * 4 + 1.
"""


@mock.patch('urllib.request.urlopen')
def test_fetch_guide_data_collection_data_traversal(mock_urlopen, tmpdir):
    # Create a malicious tarball in memory.
    malicious_tar_stream = io.BytesIO()
    with tarfile.open(fileobj=malicious_tar_stream, mode='w:xz') as tf:
        # Add a file that tries to write outside the target directory
        malicious_info = tarfile.TarInfo(name="../../tmp/pwned")
        tf.addfile(malicious_info, io.BytesIO(b"pwned"))
    malicious_tar_stream.seek(0)

    # Read the stream into a BytesIO object so that the mock should return a
    # response object whose read() method returns the tarball content.
    mock_response = mock.Mock()
    mock_response.read.return_value = malicious_tar_stream.getvalue()
    mock_urlopen.return_value = mock_response

    with pytest.raises(ValueError, match="Encountered untrusted path"):
        recirq.fetch_guide_data_collection_data(base_dir=tmpdir)

    assert not os.path.exists('/tmp/pwned')
