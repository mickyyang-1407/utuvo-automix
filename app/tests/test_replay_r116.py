"""A plan the published 0.116.0 App wrote renders bit-identically (R117).

R117 changed what new plans ask for (balance.sends = post_fader_v1 on the
vocal, drum and bass buses, saturation_law, gentler drum compression); a saved
0.116.0 plan carries none of that and must render exactly the WAV 0.116.0
rendered from it.
"""
import hashlib
import json
from pathlib import Path

from automix import cli

HERE = Path(__file__).resolve().parent
PLAN = HERE / "r25_reference" / "R116-pop-plan.json"
DEMO = HERE.parent / "demo_stems"
WAV_SHA256 = "2c51549dac0009042ef8567489d946897128c9b725363c97f9d379e17bb01ce5"


def test_the_plan_predates_every_r117_field():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert "saturation_law" not in plan
    for name in ("vocal", "drum", "bass"):
        assert "sends" not in plan["buses"][name]["balance"]
    # it is an R116 plan: it has the R116 fields
    assert plan["reverb"]["timing"]["law"] == "tempo_space_v1"
    assert plan["buses"]["drum"]["balance"]["meter"] == "master_eq_v1"


def test_a_saved_0_116_plan_renders_the_same_wav(tmp_path):
    code = cli.main(["--render-plan", str(PLAN), "--stems", str(DEMO), "--out", str(tmp_path / "out"), "--quiet"])
    assert code in (0, None)
    with (tmp_path / "out" / "mix.wav").open("rb") as fh:
        assert hashlib.file_digest(fh, "sha256").hexdigest() == WAV_SHA256
