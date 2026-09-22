"""
manifest.py - Provenance ledger for the tracked figures/tables in results/.
=============================================================================
results/ is a *published* tier: no script writes there directly any more.
Every script under code/scripts/ writes into the matching subtree of
diagnostic_outputs/ (the staging tier, gitignored); code/scripts/publish_results.py
is the only path that copies staged files into results/, and it always goes
through publish() below, which records what was published, by which script,
from which commit, and when.

results/MANIFEST.json is the ledger: one entry per tracked file, keyed by its
path relative to RESULTS_ROOT. tests/test_results_manifest.py fails CI if a
tracked file under results/ has no entry — the check that stops a plain
`git add results/...` from bypassing this file silently. See LOG-029.
"""

from __future__ import annotations

import datetime
import json
import shutil
import subprocess
from pathlib import Path

from models.paths import PROJECT_ROOT, RESULTS_ROOT

MANIFEST_PATH = RESULTS_ROOT / "MANIFEST.json"


def git_commit() -> str:
    """Current HEAD short hash, or 'unknown' outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def load(results_root: Path = RESULTS_ROOT) -> dict:
    """The current ledger, as {relative_path: {produced_by, commit, published_at}}."""
    manifest_path = Path(results_root) / "MANIFEST.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text())


def _save(entries: dict, results_root: Path) -> None:
    manifest_path = Path(results_root) / "MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {k: entries[k] for k in sorted(entries)}
    manifest_path.write_text(json.dumps(ordered, indent=2) + "\n")


def publish(src: Path, dst_subdir: str, produced_by: str,
            pattern: str = "**/*", results_root: Path = RESULTS_ROOT) -> list[str]:
    """Copy src (a file, or every file under it matching pattern) into
    results_root/dst_subdir, and record each copied file in
    results_root/MANIFEST.json.

    Only files actually copied this call get a manifest entry, and nothing
    under dst_subdir is deleted first — two producers can publish into the
    same results/ folder (e.g. 02_model_analysis) without one clobbering the
    other's manifest entries.

    Returns the list of published paths, relative to results_root.
    """
    src = Path(src)
    results_root = Path(results_root)
    dst_root = results_root / dst_subdir
    dst_root.mkdir(parents=True, exist_ok=True)

    if src.is_file():
        files, rel_base = [src], src.parent
    else:
        files = sorted(p for p in src.glob(pattern) if p.is_file())
        rel_base = src

    if not files:
        raise FileNotFoundError(f"Nothing to publish: {src} (pattern={pattern!r})")

    commit = git_commit()
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    entries = load(results_root)
    published = []

    for f in files:
        dst = dst_root / f.relative_to(rel_base)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        key = dst.relative_to(results_root).as_posix()
        entries[key] = {"produced_by": produced_by, "commit": commit, "published_at": timestamp}
        published.append(key)

    _save(entries, results_root)
    return published
