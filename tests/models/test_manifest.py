from models.manifest import load, publish


def test_publish_copies_file_and_records_entry(tmp_path):
    src = tmp_path / "staging" / "fig.png"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"fake png")
    results_root = tmp_path / "results"

    published = publish(src, "02_model_analysis", "some_script.py", results_root=results_root)

    assert published == ["02_model_analysis/fig.png"]
    assert (results_root / "02_model_analysis" / "fig.png").read_bytes() == b"fake png"
    entry = load(results_root)["02_model_analysis/fig.png"]
    assert entry["produced_by"] == "some_script.py"
    assert "commit" in entry and "published_at" in entry


def test_publish_directory_filters_by_pattern(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "a.csv").write_text("x")
    (staging / "b.png").write_bytes(b"x")
    results_root = tmp_path / "results"

    published = publish(staging, "tables", "some_script.py",
                         pattern="*.csv", results_root=results_root)

    assert published == ["tables/a.csv"]
    assert not (results_root / "tables" / "b.png").exists()


def test_publish_does_not_clobber_a_sibling_producers_entries(tmp_path):
    results_root = tmp_path / "results"
    src_a = tmp_path / "staging_a" / "one.png"
    src_a.parent.mkdir(parents=True)
    src_a.write_bytes(b"a")
    src_b = tmp_path / "staging_b" / "two.png"
    src_b.parent.mkdir(parents=True)
    src_b.write_bytes(b"b")

    publish(src_a, "02_model_analysis", "script_a.py", results_root=results_root)
    publish(src_b, "02_model_analysis", "script_b.py", results_root=results_root)

    entries = load(results_root)
    assert entries["02_model_analysis/one.png"]["produced_by"] == "script_a.py"
    assert entries["02_model_analysis/two.png"]["produced_by"] == "script_b.py"


def test_publish_raises_when_nothing_matches(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    results_root = tmp_path / "results"

    try:
        publish(staging, "tables", "some_script.py", pattern="*.csv", results_root=results_root)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
