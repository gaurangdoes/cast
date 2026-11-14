"""Where things live. Importing this also makes the Vexo Open kit importable: the kit is a
folder cloned next to the code (vexo-open-kit/), not an installed package."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # the repository
DATA = ROOT / "data"                               # recordings, models, saved tasks: never committed
KIT = ROOT / "vexo-open-kit"
WEB = ROOT / "web"

for _p in (KIT / "python", KIT / "examples" / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
