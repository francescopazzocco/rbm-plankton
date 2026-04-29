"""
models/__init__.py — Expose RBM model classes
============================================
"""

from .bernoulli_rbm import BernoulliRBM
from .nb_rbm import NBRBM

__all__ = ["BernoulliRBM", "NBRBM"]
