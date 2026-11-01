"""CLI-level tests.

The library tests cover the engine; these cover the thing a user actually
types.  They run the module in a subprocess so argument parsing, exit codes and
the files it writes are all exercised for real.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(HERE, "demo_stems")


def run_cli(*args, cwd=HERE):
    return subprocess.run(
        [sys.executable, "-m", "automix.cli", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_plan_only_writes_a_plan_and_no_audio(tmp_path):
    out = tmp_path / "out"
    proc = run_cli("--stems", DEMO, "--style", "pop_2020s", "--plan-only", "--out", str(out))
    assert proc.returncode == 0, proc.stderr[-800:]
    assert (out / "mixplan.json").exists()
    assert not (out / "mix.wav").exists()
    assert "--plan-only" in proc.stdout
    # the summary is the point: it must show the decisions, not just "ok"
    assert "track" in proc.stdout and "decisions" in proc.stdout
    assert "target" in proc.stdout


def test_plan_only_plan_is_the_same_plan_a_render_would_use(tmp_path):
    """Plan-only is for inspecting decisions; it must not be a different engine."""
    plan_out = tmp_path / "a"
    run_cli("--stems", DEMO, "--style", "pop_2020s", "--plan-only", "--out", str(plan_out))
    render_out = tmp_path / "b"
    proc = run_cli("--stems", DEMO, "--style", "pop_2020s", "--out", str(render_out))
    assert proc.returncode == 0, proc.stderr[-800:]

    a = json.load(open(plan_out / "mixplan.json"))
    b = json.load(open(render_out / "mixplan.json"))
    ga = {t["id"]: t["gain_db"] for t in a["tracks"]}
    gb = {t["id"]: t["gain_db"] for t in b["tracks"]}
    assert ga == gb
    assert a["buses"]["master"]["target_lufs"] == b["buses"]["master"]["target_lufs"]


def test_an_unknown_style_fails_cleanly(tmp_path):
    proc = run_cli("--stems", DEMO, "--style", "nope", "--out", str(tmp_path / "x"))
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "nope" in combined


def test_list_styles_needs_no_stems():
    proc = run_cli("--list-styles")
    assert proc.returncode == 0
    for style in ("pop_2020s", "rock", "lofi", "trap", "acoustic", "cinematic"):
        assert style in proc.stdout


def test_missing_stems_directory_is_a_clear_error(tmp_path):
    proc = run_cli("--stems", str(tmp_path / "does-not-exist"), "--out", str(tmp_path / "y"))
    assert proc.returncode != 0
    assert "not a directory" in (proc.stdout + proc.stderr)

def test_export_stems_writes_tracks_reverb_and_a_note(tmp_path):
    out = tmp_path / "mix"
    stems = tmp_path / "stems"
    proc = run_cli(
        "--stems", DEMO, "--style", "pop_2020s",
        "--export-stems", str(stems), "--out", str(out),
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    wavs = sorted(p.name for p in stems.glob("*.wav"))
    assert "vocal_lead.wav" in wavs
    assert "_reverb.wav" in wavs
    assert len(wavs) >= 7
    assert (stems / "README.txt").exists()

    note = (stems / "README.txt").read_text(encoding="utf-8")
    # the note has to state what is NOT in the files, or the export is a trap
    assert "NOT included" in note
    assert "master" in note
    assert "does NOT reproduce mix.wav" in note

    for name in wavs:
        assert (stems / name).stat().st_size > 1000

    # the master chain, so the mix can be rebuilt instead of guessed at
    chain = json.loads((stems / "_master_chain.json").read_text(encoding="utf-8"))
    master = chain["master"]
    assert master["target_lufs"] == -10.5
    assert master["limiter_ceiling_db"] == -1.0
    assert master["eq"], "the master EQ belongs in the chain file"
    # the note states how far the stems are from the finished mix, measured
    assert "Measured" in note
    assert "lands" in note
    assert "loudness normalisation alone was" in note


def test_the_exported_stems_are_the_balance_the_mix_has(tmp_path):
    """Sum the exported tracks: the shape must match the mix, the level must
    differ by exactly the loudness gain the note claims."""
    import numpy as np

    from automix import analyze, dsp

    out = tmp_path / "mix"
    stems = tmp_path / "stems"
    proc = run_cli(
        "--stems", DEMO, "--style", "pop_2020s",
        "--export-stems", str(stems), "--out", str(out),
    )
    assert proc.returncode == 0, proc.stderr[-800:]

    acc = None
    for p in sorted(stems.glob("*.wav")):
        audio, _ = analyze.load_stem_file(str(p))
        acc = audio if acc is None else acc + audio
    mix, _ = analyze.load_stem_file(str(out / "mix.wav"))
    n = min(acc.shape[0], mix.shape[0])
    a = acc[:n].mean(axis=1)
    b = mix[:n].mean(axis=1)
    assert float(np.corrcoef(a, b)[0, 1]) > 0.9, "the exported balance must match"

    rms_sum = dsp.lin_to_db(float(np.sqrt(np.mean(acc.astype(np.float64) ** 2))))
    rms_mix = dsp.lin_to_db(float(np.sqrt(np.mean(mix.astype(np.float64) ** 2))))
    measured = rms_mix - rms_sum

    # the number in the note has to be the measured one, not a component of it
    note = (stems / "README.txt").read_text(encoding="utf-8")
    import re

    m = re.search(r"lands ([+-][\d.]+) dB", note)
    assert m, note
    assert abs(float(m.group(1)) - measured) < 0.3, (m.group(1), measured)

    chain = json.loads((stems / "_master_chain.json").read_text(encoding="utf-8"))
    claimed = float(chain["master"]["loudness_gain_db"])
    # The actual gap also includes the documented common export headroom gain.
    exported = json.loads((stems / "_export_report.json").read_text())
    assert measured == pytest.approx(exported["comparison"]["reference_minus_sum_rms_dbfs"], abs=.01)
    assert np.isfinite(claimed)

def test_plan_only_refuses_to_export_stems(tmp_path):
    proc = run_cli(
        "--stems", DEMO, "--plan-only", "--export-stems", str(tmp_path / "s"),
        "--out", str(tmp_path / "o"),
    )
    assert proc.returncode != 0
    assert "cannot --export-stems" in (proc.stdout + proc.stderr)
