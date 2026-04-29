"""
utils.py — Shared utilities for RBM plankton project
==================================================
"""

import os
import numpy as np
import torch


def get_device():
    """Return CUDA device if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"[Device]  {name}")
    return device


def save_weights(out_dir, weights_dict):
    """Save weights dictionary to npz file."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "weights.npz")
    np.savez(path, **weights_dict)
    print(f"[Saved]  weights → {path}")


def load_weights(path):
    """Load weights from npz file."""
    return np.load(path, allow_pickle=True)
