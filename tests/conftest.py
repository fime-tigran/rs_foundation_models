import sys
from pathlib import Path

rs_finetune = Path(__file__).resolve().parent.parent / "rs_finetune"
if str(rs_finetune) not in sys.path:
    sys.path.insert(0, str(rs_finetune))
