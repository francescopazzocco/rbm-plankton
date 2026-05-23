# models/ - RBM family

## Files

| File | Contents |
|------|----------|
| `base_rbm.py` | `BaseRBM` - shared init (`W`, `a`, `b`), `reconstruction_mse()` |
| `bernoulli_rbm.py` | `BernoulliRBM` - Bernoulli visible + Bernoulli hidden (via CD) |
| `_constants.py` | numeric constants for all RBM model classes |
| `nb_rbm.py` | `NB_RBM`, `NB_ReLU_RBM`, `NBSigmoidRBM`, `NBSoftmaxRBM` - NB visible family |
| `zinb_rbm.py` | `ZINB_RBM`, `ZINB_ReLU_RBM` - ZINB visible family |
| `_hidden_monitors.py` | Mixins: `BernoulliHiddenMonitor`, `ReLUHiddenMonitor`, `SigmoidHiddenMonitor`, `SoftmaxHiddenMonitor` |
| `io.py` | Load/save model parameters, `METRIC_COL`, `best_seed_dir` |
| `utils.py` | Helper functions (data prep, evaluation) |
| `visualization.py` | Plotting helpers |
| `__init__.py` | Exports all model classes |

## Class hierarchy

```
BaseRBM                         base_rbm.py
├── BernoulliRBM                bernoulli_rbm.py
│
├── NB branch                   nb_rbm.py
│   └── NB_RBM(BernoulliHiddenMonitor, BaseRBM)
│       ├── NB_ReLU_RBM(ReLUHiddenMonitor, NB_RBM)     unstable
│       ├── NBSigmoidRBM(SigmoidHiddenMonitor, NB_RBM) stable
│       └── NBSoftmaxRBM(SoftmaxHiddenMonitor, NB_RBM) stable
│
└── ZINB branch                 zinb_rbm.py
    └── ZINB_RBM(BernoulliHiddenMonitor, BaseRBM)
        ├── ZINB_ReLU_RBM(ReLUHiddenMonitor, ZINB_RBM)     unstable
        ├── ZINBSigmoidRBM(SigmoidHiddenMonitor, ZINB_RBM) stable
        └── ZINBSoftmaxRBM(SoftmaxHiddenMonitor, ZINB_RBM) stable
```

Each visible branch implements a different count distribution (NB or ZINB).  
Within each branch, the hidden type is swapped via monitoring mixin:

| Mixin | Hidden type | Monitor |
|-------|-------------|---------|
| `BernoulliHiddenMonitor` | Bernoulli | sat_lo/sat_hi/sat_mid |
| `ReLUHiddenMonitor` | Rectified Gaussian (ReLU) | h_mean, h_sparsity |
| `SigmoidHiddenMonitor` | Sigmoid → Bernoulli | h_mean |
| `SoftmaxHiddenMonitor` | Softmax → Multinomial one-hot | h_entropy |

