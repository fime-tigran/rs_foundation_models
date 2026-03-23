import sys
from pathlib import Path

import pytest

rs_finetune = Path(__file__).resolve().parent.parent
if str(rs_finetune) not in sys.path:
    sys.path.insert(0, str(rs_finetune))


@pytest.fixture(autouse=True)
def _seed_torch():
    import torch
    torch.manual_seed(42)
