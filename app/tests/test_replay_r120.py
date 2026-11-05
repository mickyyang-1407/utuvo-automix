"""A plan the published 0.120.0 App wrote and rendered renders bit-identically (R121).

The plan was written by the notarized 0.120.0 App on the demo stems
(pop_2020s: the lead doubled and its resonances smoothed, the bass ducked
under the kick as a whole -- kick_duck_v1; one mic per drum, so nothing to
line up) and rendered by
that same App.  R121 changes how new
plans duck the bass (kick_duck_band_v1); this plan keeps kick_duck_v1 and
must play exactly as 0.120.0 played it.
"""
import hashlib
import json
from pathlib import Path

from automix import cli

HERE = Path(__file__).resolve().parent
PLAN = HERE / "r25_reference" / "R120-pop-plan.json"
DEMO = HERE.parent / "demo_stems"
WAV_SHA256 = "0014c79771973eed5abe8bcef8f378d317c756916aa479c601314bfee9d1b0dd"


def test_the_plan_is_an_r120_plan_with_the_whole_bass_ducked():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["buses"]["bass"]["duck"]["law"] == "kick_duck_v1"
    assert "crossover_hz" not in plan["buses"]["bass"]["duck"]
    lead = next(t for t in plan["tracks"] if t["role"] == "lead_vocal")
    assert "double" in lead and "resonance" in lead


def test_a_saved_0_120_plan_renders_the_same_wav(tmp_path):
    code = cli.main(["--render-plan", str(PLAN), "--stems", str(DEMO), "--out", str(tmp_path / "out"), "--quiet"])
    assert code in (0, None)
    with (tmp_path / "out" / "mix.wav").open("rb") as fh:
        assert hashlib.file_digest(fh, "sha256").hexdigest() == WAV_SHA256
