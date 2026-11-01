"""Preflight: a saved plan is checked before anything is loaded or written.

The CLI used to render a saved plan first and validate it afterwards, so an
invalid plan still wrote a mix.wav and then died on a raw jsonschema
traceback.  These tests are self-contained: they build their own stems, their
own plan and their own output folder.  No demo_stems, no App, no server.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import wave

import numpy as np
import pytest

from automix import analyze, plan as planner

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 48000


def run_cli(*args, cwd=HERE):
    return subprocess.run(
        [sys.executable, "-m", "automix.cli", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
    )


def _write_wav(path, audio, sr=SR):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _synthetic_stems(tmp_path, sr=SR):
    """Two short stems of our own -- nothing borrowed from demo_stems."""
    stems_dir = tmp_path / "stems"
    stems_dir.mkdir()
    n = int(1.5 * sr)
    t = np.arange(n) / sr
    voice = 0.2 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.sin(2 * np.pi * 660 * t)
    guitar = 0.3 * np.sin(2 * np.pi * 110 * t) + 0.1 * np.sin(2 * np.pi * 330 * t)
    _write_wav(stems_dir / "vocal_lead.wav", np.stack([voice, voice], axis=1), sr)
    _write_wav(stems_dir / "guitar_elec.wav", np.stack([guitar, guitar], axis=1), sr)
    return stems_dir


def _plan_for(stems_dir):
    """A plan the engine accepts, pointing at our own stems."""
    stems = analyze.load_stems(str(stems_dir))
    analyze.detect_roles(stems)
    analyze.align(stems)
    for s in stems:
        analyze.measure(s)
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    p["source"]["stems_dir"] = os.path.abspath(str(stems_dir))
    planner.validate(p)
    return p


@pytest.fixture(scope="module")
def plan_file(tmp_path_factory):
    """A saved plan on disk, plus the stems it points at."""
    root = tmp_path_factory.mktemp("preflight")
    stems_dir = _synthetic_stems(root)
    path = root / "mixplan.json"
    planner.write_plan(_plan_for(stems_dir), str(path))
    return path


def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _assert_no_traceback(proc):
    assert "Traceback" not in proc.stderr, proc.stderr[-2000:]
    assert "Traceback" not in proc.stdout, proc.stdout[-2000:]


def test_a_valid_plan_renders_and_replays_byte_exact(plan_file, tmp_path):
    """The saved plan still renders, and rendering it again gives the same audio."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    one = run_cli("--render-plan", str(plan_file), "--out", str(first))
    two = run_cli("--render-plan", str(plan_file), "--out", str(second))
    assert one.returncode == 0, one.stderr[-800:]
    assert two.returncode == 0, two.stderr[-800:]
    _assert_no_traceback(one)
    assert (first / "mix.wav").exists()
    assert (first / "report.md").exists()
    assert (first / "mixplan.json").exists()
    assert _sha256(first / "mix.wav") == _sha256(second / "mix.wav")


def test_relocated_sources_replay_byte_exact_and_record_the_selected_folder(tmp_path):
    stems_dir = _synthetic_stems(tmp_path)
    path = tmp_path / "mixplan.json"
    plan = _plan_for(stems_dir)
    planner.write_plan(plan, str(path))
    saved_sha = _sha256(path)

    first = tmp_path / "first"
    one = run_cli("--render-plan", str(path), "--out", str(first))
    assert one.returncode == 0, one.stderr[-800:]

    moved = tmp_path / "moved_stems"
    stems_dir.rename(moved)
    second = tmp_path / "second"
    two = run_cli("--render-plan", str(path), "--stems", str(moved),
                  "--out", str(second))
    assert two.returncode == 0, two.stderr[-800:]
    _assert_no_traceback(two)
    assert _sha256(first / "mix.wav") == _sha256(second / "mix.wav")
    assert _sha256(path) == saved_sha
    written = json.loads((second / "mixplan.json").read_text(encoding="utf-8"))
    plan["source"]["stems_dir"] = os.path.abspath(str(moved))
    assert written == plan


@pytest.mark.parametrize("empty_override", [False, True])
def test_an_unusable_explicit_source_folder_never_falls_back_to_the_saved_one(
    plan_file, tmp_path, empty_override
):
    override = "" if empty_override else str(tmp_path / "not-here")
    out = tmp_path / "out"
    saved_sha = _sha256(plan_file)
    proc = run_cli("--render-plan", str(plan_file), "--stems", override,
                   "--out", str(out))
    assert proc.returncode != 0
    _assert_no_traceback(proc)
    assert "stems_dir" in proc.stderr
    assert repr(override) in proc.stderr
    assert _sha256(plan_file) == saved_sha
    assert not (out / "mix.wav").exists()
    assert not (out / "report.md").exists()
    assert not (out / "mixplan.json").exists()


@pytest.mark.parametrize("source_rate", [44100, 96000])
def test_saved_plan_replay_keeps_its_sample_clock(plan_file, tmp_path, source_rate):
    source_root = tmp_path / "different_rate"
    source_root.mkdir()
    stems_dir = _synthetic_stems(source_root, sr=source_rate)
    saved_sha = _sha256(plan_file)
    out = tmp_path / "out"
    proc = run_cli("--render-plan", str(plan_file), "--stems", str(stems_dir),
                   "--out", str(out))
    assert proc.returncode == 0, proc.stderr[-800:]
    _assert_no_traceback(proc)
    with wave.open(str(out / "mix.wav"), "rb") as audio:
        assert audio.getframerate() == SR
        assert audio.getnframes() == int(1.5 * SR)
    assert _sha256(plan_file) == saved_sha
    written = json.loads((out / "mixplan.json").read_text(encoding="utf-8"))
    conversions = [w for w in written["warnings"] if "for saved plan replay" in w]
    assert len(conversions) == 2
    assert all(f"{source_rate} Hz" in w and f"{SR} Hz" in w for w in conversions)

    again = tmp_path / "again"
    proc = run_cli("--render-plan", str(out / "mixplan.json"), "--out", str(again))
    assert proc.returncode == 0, proc.stderr[-800:]
    repeated = json.loads((again / "mixplan.json").read_text(encoding="utf-8"))
    assert repeated["warnings"] == written["warnings"]
    assert _sha256(again / "mix.wav") == _sha256(out / "mix.wav")


def test_saved_plan_refuses_a_missing_required_track(tmp_path):
    stems_dir = _synthetic_stems(tmp_path)
    path = tmp_path / "mixplan.json"
    planner.write_plan(_plan_for(stems_dir), str(path))
    saved_sha = _sha256(path)
    (stems_dir / "vocal_lead.wav").unlink()
    out = tmp_path / "out"
    proc = run_cli("--render-plan", str(path), "--out", str(out))
    assert proc.returncode == 2
    _assert_no_traceback(proc)
    assert "missing" in proc.stderr
    assert "vocal_lead" in proc.stderr
    assert _sha256(path) == saved_sha
    assert not (out / "mix.wav").exists()
    assert not (out / "report.md").exists()
    assert not (out / "mixplan.json").exists()


def test_saved_plan_allows_an_extra_unplanned_source(tmp_path):
    stems_dir = _synthetic_stems(tmp_path)
    path = tmp_path / "mixplan.json"
    plan = _plan_for(stems_dir)
    planner.write_plan(plan, str(path))
    n = int(0.5 * SR)
    extra = 0.1 * np.sin(2 * np.pi * 90 * np.arange(n) / SR)
    _write_wav(stems_dir / "unused_loop.wav", np.stack([extra, extra], axis=1))
    out = tmp_path / "out"
    proc = run_cli("--render-plan", str(path), "--out", str(out))
    assert proc.returncode == 0, proc.stderr[-800:]
    _assert_no_traceback(proc)
    with wave.open(str(out / "mix.wav"), "rb") as audio:
        assert audio.getframerate() == SR
        assert audio.getnframes() == int(1.5 * SR)
    written = json.loads((out / "mixplan.json").read_text(encoding="utf-8"))
    assert written["tracks"] == plan["tracks"]


def test_a_plan_that_breaks_the_schema_is_refused_before_anything_is_written(plan_file, tmp_path):
    """The R29 case: a bus threshold outside the schema range."""
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    plan["buses"]["music"]["comp"]["threshold_db"] = 3.5
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(plan), encoding="utf-8")

    out = tmp_path / "out"
    proc = run_cli("--render-plan", str(broken), "--out", str(out))
    assert proc.returncode != 0
    _assert_no_traceback(proc)
    # the actual reason is reported, not a generic refusal
    assert "threshold_db" in proc.stderr
    assert "3.5" in proc.stderr
    # nothing was rendered, and the output folder was not even created
    assert not out.exists()


def test_plan_files_that_are_not_a_plan_are_refused(tmp_path):
    """Empty, malformed and non-object files all fail cleanly."""
    cases = {
        "empty.json": "",
        "malformed.json": '{"schema_version": ',
        "array.json": "[1, 2, 3]",
        "number.json": "42",
    }
    for name, body in cases.items():
        path = tmp_path / name
        path.write_text(body, encoding="utf-8")
        out = tmp_path / ("out-" + name)
        proc = run_cli("--render-plan", str(path), "--out", str(out))
        assert proc.returncode != 0, name
        _assert_no_traceback(proc)
        assert not out.exists(), name


def test_a_plan_missing_a_required_key_is_refused(plan_file, tmp_path):
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    del plan["buses"]
    broken = tmp_path / "no_buses.json"
    broken.write_text(json.dumps(plan), encoding="utf-8")

    out = tmp_path / "out"
    proc = run_cli("--render-plan", str(broken), "--out", str(out))
    assert proc.returncode != 0
    _assert_no_traceback(proc)
    assert "buses" in proc.stderr
    assert not out.exists()


def test_a_plan_whose_source_is_gone_is_refused(plan_file, tmp_path):
    """A plan that no longer points at real stems must not render a mix."""
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    plan["source"]["stems_dir"] = str(tmp_path / "not-here")
    broken = tmp_path / "no_source.json"
    broken.write_text(json.dumps(plan), encoding="utf-8")

    out = tmp_path / "out"
    proc = run_cli("--render-plan", str(broken), "--out", str(out))
    assert proc.returncode != 0
    _assert_no_traceback(proc)
    assert "stems_dir" in proc.stderr
    assert not (out / "mix.wav").exists()
    assert not (out / "report.md").exists()
    assert not (out / "mixplan.json").exists()


def test_a_refused_plan_leaves_existing_results_untouched(plan_file, tmp_path):
    """A failure must not touch what a previous run already wrote."""
    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "mix.wav"
    sentinel.write_bytes(b"an earlier mix, byte for byte")
    before = _sha256(sentinel)

    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    plan["buses"]["music"]["comp"]["threshold_db"] = 3.5
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(plan), encoding="utf-8")

    proc = run_cli("--render-plan", str(broken), "--out", str(out))
    assert proc.returncode != 0
    _assert_no_traceback(proc)
    assert _sha256(sentinel) == before
    assert not (out / "report.md").exists()
    assert not (out / "mixplan.json").exists()


def test_an_invalid_plan_never_reaches_the_renderer(plan_file, tmp_path, monkeypatch):
    """The loader and the renderer must not be called at all."""
    from automix import cli, render

    called = []

    def spy(name):
        def inner(*args, **kwargs):
            called.append(name)
            raise AssertionError(f"{name} must not be reached")
        return inner

    monkeypatch.setattr(analyze, "load_stems", spy("load_stems"))
    monkeypatch.setattr(render, "render", spy("render"))

    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    plan["buses"]["music"]["comp"]["threshold_db"] = 3.5
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(plan), encoding="utf-8")

    code = cli.main(["--render-plan", str(broken), "--out", str(tmp_path / "out")])
    assert code == 2
    assert called == []


def test_the_reverb_override_still_applies(plan_file, tmp_path):
    """--reverb-amount keeps working, and lands on the same audio as baking it in."""
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    baked = tmp_path / "baked.json"
    plan["reverb"] = dict(plan.get("reverb") or {})
    plan["reverb"]["manual_amount"] = 0.0
    baked.write_text(json.dumps(plan), encoding="utf-8")

    overridden = tmp_path / "overridden"
    proc = run_cli("--render-plan", str(baked), "--out", str(overridden),
                   "--reverb-amount", "0.0")
    assert proc.returncode == 0, proc.stderr[-800:]
    _assert_no_traceback(proc)

    via_flag = tmp_path / "via_flag"
    proc = run_cli("--render-plan", str(plan_file), "--out", str(via_flag),
                   "--reverb-amount", "0.0")
    assert proc.returncode == 0, proc.stderr[-800:]

    assert _sha256(overridden / "mix.wav") == _sha256(via_flag / "mix.wav")
    written = json.loads((via_flag / "mixplan.json").read_text(encoding="utf-8"))
    assert written["reverb"]["manual_amount"] == 0.0


@pytest.mark.parametrize("bad_path", [["stems", "elsewhere"], {"dir": "stems"}])
def test_a_source_path_that_is_not_a_string_is_refused(plan_file, tmp_path, bad_path):
    """The CLI walks source.stems_dir as a path: it has to be a string."""
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    plan["source"]["stems_dir"] = bad_path
    broken = tmp_path / "bad_path.json"
    broken.write_text(json.dumps(plan), encoding="utf-8")

    out = tmp_path / "out"
    proc = run_cli("--render-plan", str(broken), "--out", str(out))
    assert proc.returncode != 0
    _assert_no_traceback(proc)
    assert "stems_dir" in proc.stderr
    assert not out.exists()

    # an earlier run's results are left exactly as they were
    kept = tmp_path / "kept"
    kept.mkdir()
    sentinel = kept / "mix.wav"
    sentinel.write_bytes(b"an earlier mix, byte for byte")
    before = _sha256(sentinel)
    proc = run_cli("--render-plan", str(broken), "--out", str(kept))
    assert proc.returncode != 0
    _assert_no_traceback(proc)
    assert _sha256(sentinel) == before
    assert not (kept / "report.md").exists()
    assert not (kept / "mixplan.json").exists()


def test_a_number_python_will_not_convert_is_a_parse_error_not_a_traceback(tmp_path):
    """A 5000-digit integer is well-formed JSON that Python refuses to parse."""
    path = tmp_path / "large_integer.json"
    path.write_text('{"source": ' + "9" * 5000 + "}", encoding="utf-8")

    out = tmp_path / "out"
    proc = run_cli("--render-plan", str(path), "--out", str(out))
    assert proc.returncode != 0
    _assert_no_traceback(proc)
    # the limit is stated, not guessed, and not worked around
    assert "could not be parsed" in proc.stderr
    assert "digits" in proc.stderr
    assert not out.exists()
