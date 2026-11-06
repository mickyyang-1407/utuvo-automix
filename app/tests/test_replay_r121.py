"""A plan made by notarized 0.121.0 replays to the same 24-bit WAV."""
import hashlib
from pathlib import Path

from automix import cli

HERE = Path(__file__).resolve().parent
PLAN = HERE / "r25_reference" / "R121-pop-plan.json"
WAV_SHA256 = "54b3f0b5d9647431a3a7c29c52f8162f161875a46081d85a9abea146fe39b38f"


def test_notarized_0121_plan_replays_byte_for_byte(tmp_path):
    out = tmp_path / "out"
    code = cli.main(["--render-plan", str(PLAN), "--stems", str(HERE.parent / "demo_stems"),
                     "--out", str(out), "--quiet"])
    assert code in (0, None)
    with (out / "mix.wav").open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == WAV_SHA256
