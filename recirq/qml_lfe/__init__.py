# Insert qml_lfe on PYTHONPATH. This is to avoid the requirement
# of needing to do from recirq.qml_lfe.xxx style imports inside of the
# qml_lfe directory.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
