"""Backward-compatibility shim.

The method formerly named LMB (Learnable Multi-Bin) is now LGQ (Learnable
Geometric Quantization). This module re-exports the config under its old name so
existing scripts that do ``from configs.lmb import LMBConfig`` keep working.
"""

from configs.lgq import LGQConfig as LMBConfig  # noqa: F401
