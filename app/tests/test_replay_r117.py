"""A plan the published 0.117.0 App wrote renders bit-identically (R118).

R118 added the user's group faders (buses.<group>.user_trim_db); a saved
0.117.0 plan has none and must render exactly the WAV 0.117.0 rendered from
it.
"""
import hashlib
import json
from pathlib import Path

from automix import cli

HERE = Path(__file__).resolve().parent
PLAN = HERE / "r25_reference" / "R117-pop-plan.json"
DEMO = HERE.parent / "demo_stems"
WAV_SHA256 = "e91832023a63d81447bc84f6bdf01df90d0844874ffec63676b0c47d0bf56903"


def test_the_plan_is_an_r117_plan_without_faders():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["saturation_law"] == "env_tanh_os4_v1"
    assert plan["buses"]["vocal"]["balance"]["sends"] == "post_fader_v1"
    assert all("user_trim_db" not in b for b in plan["buses"].values())


def test_a_saved_0_117_plan_renders_the_same_wav(tmp_path):
    code = cli.main(["--render-plan", str(PLAN), "--stems", str(DEMO), "--out", str(tmp_path / "out"), "--quiet"])
    assert code in (0, None)
    with (tmp_path / "out" / "mix.wav").open("rb") as fh:
        assert hashlib.file_digest(fh, "sha256").hexdigest() == WAV_SHA256
