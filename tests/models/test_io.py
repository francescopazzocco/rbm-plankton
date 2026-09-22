import pandas as pd
from models.io import (
    CHRONO,
    METRIC_COL,
    SHUFFLED,
    best_seed_dir,
    discover_run_dirs,
    load_hidden_activations,
)


def _make_seed_dir(base, family_l_name, seed, val_pll):
    seed_dir = base / family_l_name / f"seed_{seed}"
    seed_dir.mkdir(parents=True)
    pd.DataFrame({"val_pll": [val_pll * 2, val_pll]}).to_csv(
        seed_dir / "rbm_training_curves.csv", index=False
    )
    return seed_dir


def test_discover_run_dirs_groups_by_family_and_l(tmp_path):
    _make_seed_dir(tmp_path, "bernoulli_median_L4", seed=0, val_pll=0.5)
    _make_seed_dir(tmp_path, "bernoulli_median_L6", seed=0, val_pll=0.4)
    _make_seed_dir(tmp_path, "nb_L4", seed=0, val_pll=0.3)
    (tmp_path / "not_a_run_dir").mkdir()

    runs = discover_run_dirs(tmp_path, CHRONO)

    assert set(runs.keys()) == {"bernoulli_median", "nb"}
    assert set(runs["bernoulli_median"].keys()) == {4, 6}
    assert len(runs["nb"][4]) == 1


def test_discover_run_dirs_selects_only_the_requested_split(tmp_path):
    _make_seed_dir(tmp_path, "nb_L4_shuffled", seed=0, val_pll=0.3)
    _make_seed_dir(tmp_path, "nb_L6", seed=0, val_pll=0.3)

    shuffled = discover_run_dirs(tmp_path, SHUFFLED)
    chrono   = discover_run_dirs(tmp_path, CHRONO)

    assert list(shuffled["nb"].keys()) == [4]
    assert list(chrono["nb"].keys()) == [6]


def test_best_seed_dir_picks_lowest_final_metric(tmp_path):
    family_l_dir = tmp_path / "bernoulli_median_L4"
    _make_seed_dir(tmp_path, "bernoulli_median_L4", seed=0, val_pll=0.5)
    _make_seed_dir(tmp_path, "bernoulli_median_L4", seed=1, val_pll=0.2)

    best = best_seed_dir(family_l_dir, METRIC_COL["bernoulli_median"])

    assert best.name == "seed_1"


def test_best_seed_dir_returns_none_when_no_seeds_present(tmp_path):
    family_l_dir = tmp_path / "empty_L4"
    family_l_dir.mkdir()

    assert best_seed_dir(family_l_dir, METRIC_COL["nb"]) is None


def test_load_hidden_activations_keeps_only_hidden_columns(tmp_path):
    csv_path = tmp_path / "activations.csv"
    pd.DataFrame({
        "date": ["2020-01-01", "2020-01-02"],
        "h0": [0.1, 0.2],
        "h1": [0.3, 0.4],
        "other": ["x", "y"],
    }).to_csv(csv_path, index=False)

    df = load_hidden_activations(csv_path)

    assert list(df.columns) == ["h0", "h1"]
    assert df.index.name == "date"
