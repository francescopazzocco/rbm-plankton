import numpy as np
from models.utils import load_weights, save_weights


def test_save_and_load_weights_round_trip(tmp_path):
    weights = {
        "W": np.random.randn(4, 3).astype(np.float32),
        "a": np.zeros(4, dtype=np.float32),
        "b": np.zeros(3, dtype=np.float32),
    }

    save_weights(tmp_path, weights)
    loaded = load_weights(tmp_path / "weights.npz")

    for key, value in weights.items():
        assert np.allclose(loaded[key], value)


def test_save_weights_creates_output_directory(tmp_path):
    out_dir = tmp_path / "nested" / "run"

    save_weights(out_dir, {"W": np.zeros((2, 2), dtype=np.float32)})

    assert (out_dir / "weights.npz").exists()
