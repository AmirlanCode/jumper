import sys
from pathlib import Path

# make `import crab_rescue` work when pytest runs from the repository root or from here
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
