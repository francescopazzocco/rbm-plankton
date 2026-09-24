"""Every tracked file under results/ (besides the docs and the manifest
itself) must have a provenance entry in results/MANIFEST.json.

A file that fails this check reached results/ some way other than
code/scripts/publish_results.py -- i.e. a script wrote there directly, or
someone `git add`-ed a file into it by hand. See ARCHITECTURE.md and
DECISION_LOG LOG-025.
"""

import subprocess

from models.manifest import load
from models.paths import PROJECT_ROOT, RESULTS_ROOT

EXEMPT = {"README.md", "MANIFEST.json"}


def _tracked_results_paths():
    out = subprocess.run(
        ["git", "ls-files", "results"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    )
    prefix_len = len("results/")
    return [line[prefix_len:] for line in out.stdout.splitlines() if line.strip()]


def test_every_tracked_result_is_in_the_manifest():
    manifest = load(RESULTS_ROOT)

    missing = [rel for rel in _tracked_results_paths()
               if rel not in EXEMPT and rel not in manifest]

    assert not missing, (
        "Tracked under results/ but missing from MANIFEST.json (published "
        "without code/scripts/publish_results.py?): " + ", ".join(missing)
    )
