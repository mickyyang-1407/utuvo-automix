"""The page's JavaScript harnesses run with the Python suite (R118).

They exercise code taken out of web/index.html with a fake DOM.  They used to
be run by hand, so a change to the page could pass the suite while one of
them failed (R118: test_project_storage_ui.mjs, after every mix request
started carrying the group faders).  Needs Node; without it they are skipped,
loudly.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
HARNESSES = sorted([*HERE.glob("test_*.js"), *HERE.glob("test_*.mjs")])
NODE = shutil.which("node")


def test_there_are_harnesses():
    names = {p.name for p in HARNESSES}
    assert {"test_ab_playback.js", "test_group_edit_invalidation.js",
            "test_project_storage_ui.mjs", "test_group_faders_ui.js"} <= names


@pytest.mark.skipif(NODE is None, reason="node is not installed: the page's JS harnesses did not run")
@pytest.mark.parametrize("harness", HARNESSES, ids=lambda p: p.name)
def test_the_harness_passes(harness):
    p = subprocess.run([NODE, str(harness)], cwd=HERE.parent, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, (p.stdout[-2000:] + p.stderr[-2000:])
