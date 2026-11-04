"""A plan the published 0.115.0 App wrote renders bit-identically (R116).

R116 changed what new plans ask for (reverb.timing, balance.meter, a 12 dB
drum/bass cut, half the drum parallel blend); a saved 0.115.0 plan carries
none of that and must render exactly the WAV 0.115.0 rendered from it.
"""
import hashlib
import json
from pathlib import Path

from automix import cli

HERE = Path(__file__).resolve().parent
PLAN = HERE / "r25_reference" / "R115-pop-plan.json"
DEMO = HERE.parent / "demo_stems"
WAV_SHA256 = "4a23d2759811056518353a06cd0356cf4766f8c20e8a39e2b5726c07b9653eea"


def test_the_plan_predates_every_r116_field():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert "timing" not in plan["reverb"]
    for name in ("vocal", "drum", "bass"):
        assert "meter" not in plan["buses"][name]["balance"]
    assert plan["buses"]["drum"]["balance"]["max_cut_db"] == 6.0
    assert plan["buses"]["drum"]["parallel"]["blend"] == 0.35


def test_a_saved_0_115_plan_renders_the_same_wav(tmp_path):
    code = cli.main(["--render-plan", str(PLAN), "--stems", str(DEMO), "--out", str(tmp_path / "out"), "--quiet"])
    assert code in (0, None)
    with (tmp_path / "out" / "mix.wav").open("rb") as fh:
        assert hashlib.file_digest(fh, "sha256").hexdigest() == WAV_SHA256
