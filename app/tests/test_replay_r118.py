"""A plan the published 0.118.0 App rendered, faders and all, renders bit-identically (R119).

The plan was written by the notarized 0.118.0 App on
the demo stems (pop_2020s), with three group faders set in it (vocal +2,
drums -1, FX & space -1.5 -- what the "almost there?" panel writes), and
rendered by that same App.  Whatever R119 adds must leave it exactly as
0.118.0 played it.
"""
import hashlib
import json
from pathlib import Path

from automix import cli

HERE = Path(__file__).resolve().parent
PLAN = HERE / "r25_reference" / "R118-pop-faders-plan.json"
DEMO = HERE.parent / "demo_stems"
WAV_SHA256 = "cf5da1f4669b4fea43314c1aed4cca1847d5493192e6ac5a94764948bf9da62f"


def test_the_plan_is_an_r118_plan_with_faders():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    trims = {k: b["user_trim_db"] for k, b in plan["buses"].items() if "user_trim_db" in b}
    assert trims == {"vocal": 2.0, "drum": -1.0, "fx": -1.5}


def test_a_saved_0_118_plan_renders_the_same_wav(tmp_path):
    code = cli.main(["--render-plan", str(PLAN), "--stems", str(DEMO), "--out", str(tmp_path / "out"), "--quiet"])
    assert code in (0, None)
    with (tmp_path / "out" / "mix.wav").open("rb") as fh:
        assert hashlib.file_digest(fh, "sha256").hexdigest() == WAV_SHA256
