"""Make the in-repo ``pyarty`` importable when running pytest without install.

Keeps `pytest` working straight from a clone, as the README documents.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
