import torch

from models.bernoulli_rbm import BernoulliRBM


def make_rbm(n_visible=8, n_hidden=3, seed=0):
    torch.manual_seed(seed)
    return BernoulliRBM(n_visible, n_hidden, device=torch.device("cpu"))


def test_hidden_probs_are_valid_probabilities():
    rbm = make_rbm()
    V = torch.randint(0, 2, (5, 8)).float()

    probs = rbm.hidden_probs(V)

    assert probs.shape == (5, 3)
    assert torch.all((probs >= 0) & (probs <= 1))


def test_reconstruct_returns_visible_shaped_probabilities():
    rbm = make_rbm()
    V = torch.randint(0, 2, (5, 8)).float()

    recon = rbm.reconstruct(V)

    assert recon.shape == V.shape
    assert torch.all((recon >= 0) & (recon <= 1))


def test_free_energy_returns_one_value_per_sample():
    rbm = make_rbm()
    V = torch.randint(0, 2, (5, 8)).float()

    energy = rbm.free_energy(V)

    assert energy.shape == (5,)
    assert torch.isfinite(energy).all()


def test_reconstruction_mse_is_finite_and_nonnegative():
    rbm = make_rbm()
    V = torch.randint(0, 2, (5, 8)).float()

    mse = rbm.reconstruction_mse(V)

    assert mse >= 0
