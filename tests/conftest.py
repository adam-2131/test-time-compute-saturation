
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

for directory in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
