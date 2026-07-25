import pytest
import torch

from models.base_rbm import BaseRBM


def test_init_creates_correctly_shaped_parameters():
    rbm = BaseRBM(n_visible=10, n_hidden=4, device=torch.device("cpu"))

    assert rbm.W.shape == (10, 4)
    assert rbm.b.shape == (4,)
    assert rbm.a.shape == (10,)
    assert rbm.D == 10
    assert rbm.L == 4


def test_init_with_scale_init_false_leaves_parameters_unset():
    rbm = BaseRBM(n_visible=10, n_hidden=4, device=torch.device("cpu"), scale_init=False)

    assert rbm.W is None
    assert rbm.b is None
    assert rbm.a is None


def test_hidden_probs_and_reconstruct_are_not_implemented_on_base_class():
    rbm = BaseRBM(n_visible=5, n_hidden=3, device=torch.device("cpu"))
    V = torch.zeros(2, 5)

    with pytest.raises(NotImplementedError):
        rbm.hidden_probs(V)

    with pytest.raises(NotImplementedError):
        rbm.reconstruct(V)
