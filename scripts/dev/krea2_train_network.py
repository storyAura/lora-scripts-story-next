from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "sd-scripts"
vendor_str = str(VENDOR)
if vendor_str in sys.path:
    sys.path.remove(vendor_str)
sys.path.insert(0, vendor_str)

runpy.run_path(str(VENDOR / "krea2_train_network.py"), run_name="__main__")
