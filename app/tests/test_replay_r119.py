"""A plan the published 0.119.0 App wrote and rendered renders bit-identically (R120).

The plan was written by the notarized 0.119.0 App on the demo stems
(pop_2020s: the past-only compressor detector, the ducked vocal reverb) and
rendered by that same App.  Whatever
R120 adds -- drum mic alignment, the vocal double, resonance smoothing --
lives in new plan fields, so this plan must play exactly as 0.119.0 played it.
"""
import hashlib
import json
from pathlib import Path

from automix import cli

HERE = Path(__file__).resolve().parent
PLAN = HERE / "r25_reference" / "R119-pop-plan.json"
DEMO = HERE.parent / "demo_stems"
WAV_SHA256 = "e46fcfd2180d1e0e06395e36e4102d6075162ed0b9435eb0692a5e8b606b446f"


def test_the_plan_is_an_r119_plan():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["compressor_law"] == "trailing_box_v1" and plan["reverb"]["duck"]["law"] == "voice_duck_v1"
    assert "drum_align" not in plan
    assert not any(k in t for t in plan["tracks"] for k in ("align", "double", "resonance"))


def test_a_saved_0_119_plan_renders_the_same_wav(tmp_path):
    code = cli.main(["--render-plan", str(PLAN), "--stems", str(DEMO), "--out", str(tmp_path / "out"), "--quiet"])
    assert code in (0, None)
    with (tmp_path / "out" / "mix.wav").open("rb") as fh:
        assert hashlib.file_digest(fh, "sha256").hexdigest() == WAV_SHA256
