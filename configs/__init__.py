"""
Model-specific configurations for unified training.

Each model config defines:
- add_model_args(): Add model-specific arguments to parser
- create_model(): Create model from args and unified config
- compute_loss(): Compute loss from model output
"""

from configs.fsq import FSQConfig
from configs.ifsq import iFSQConfig
from configs.vq import VQConfig
from configs.lfq import LFQConfig
from configs.sim_vq import SimVQConfig
from configs.lgq import LGQConfig
from configs.rot_vq import RotVQConfig
from configs.softvq import SoftVQConfig
from configs.ibq import IBQConfig
from configs.bsq import BSQConfig

MODEL_CONFIGS = {
    "fsq": FSQConfig,
    "ifsq": iFSQConfig,
    "vq": VQConfig,
    "lfq": LFQConfig,
    "sim_vq": SimVQConfig,
    "lgq": LGQConfig,
    "lmb": LGQConfig,  # backward-compat alias for the former name
    "rot_vq": RotVQConfig,
    "softvq": SoftVQConfig,
    "ibq": IBQConfig,
    "bsq": BSQConfig,
}

# Backward-compatible export (LMB was renamed to LGQ).
LMBConfig = LGQConfig

__all__ = [
    "MODEL_CONFIGS",
    "FSQConfig",
    "iFSQConfig",
    "VQConfig",
    "LFQConfig",
    "SimVQConfig",
    "LGQConfig",
    "LMBConfig",
    "RotVQConfig",
    "SoftVQConfig",
    "IBQConfig",
    "BSQConfig",
]
