"""What the loader decodes, and what a plan remembers about it.

The defect these tests pin down: the loader forced every non-WAV file through
ffmpeg's 16-bit PCM output, so a 24-bit FLAC was quantised onto the PCM16 grid
before anything else ever saw it -- a precision reduction, not a rate
conversion.  The loader now decodes to float32 instead, and the old decoder
stays reachable so that a plan written before the change still replays
exactly the way it was built.

Nothing here reads a real recording: every fixture is synthesised in
``tmp_path``.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess

import numpy as np
import pytest

from automix import analyze, cli, plan as planner

SR = 44100

needs_ffmpeg = pytest.mark.skipif(
    not analyze._ffmpeg_available(), reason="ffmpeg is not installed"
)


# --------------------------------------------------------------- fixtures


def _write_wav(path, samples, sr, channels, bits, tag):
    """Write a RIFF/WAVE file by hand (PCM 16/24 or IEEE float 32/64)."""
    if tag == 1:
        if bits == 16:
            payload = np.asarray(samples, dtype="<i2").tobytes()
        elif bits == 24:
            v = np.asarray(samples, dtype=np.int64) & 0xFFFFFF
            b = np.stack(
                [v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF], axis=-1
            ).astype(np.uint8)
            payload = b.tobytes()
        else:
            raise ValueError(f"unsupported PCM width {bits}")
    else:
        payload = np.asarray(
            samples, dtype=("<f4" if bits == 32 else "<f8")
        ).tobytes()
    align = channels * bits // 8
    header = b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVE"
    header += b"fmt " + struct.pack(
        "<IHHIIHH", 16, tag, channels, sr, sr * align, align, bits
    )
    header += b"data" + struct.pack("<I", len(payload))
    with open(path, "wb") as fh:
        fh.write(header + payload)


def _run_ffmpeg(args):
    cmd = [analyze._ffmpeg_path(), "-v", "error", "-y"] + args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"ffmpeg failed: {proc.stderr[:400]}"


def _stereo24(frames=16):
    """Stereo 24-bit integers: full range, quiet values, low bytes set.

    Every value fits a signed 24-bit word, because that is what the format
    holds: a value outside the range would be read back as its two's
    complement and the comparison below would be measuring the wrong thing.
    """
    left = [
        0, 1, -1, 255, -255, 32767, -32768, 8388607,
        -8388608, 6000001, -6000001, 7, -7, 8388600, -8388600, 100,
    ]
    right = [
        1, 0, 2, -2, 256, -256, 4096, -4096,
        65536, -65536, 32768, -32768, 99, -99, 8388606, -8388606,
    ]
    pattern = []
    for l, r in zip(left, right):
        pattern += [l, r]
    out = []
    for i in range(frames):
        out += pattern
    return out


def _expected_float32(values):
    """What a 24-bit source looks like when it is decoded to float32."""
    return (np.asarray(values, dtype=np.float64) / 8388608.0).astype(np.float32)


def _expected_legacy(values):
    """What the old decoder produced: the value truncated to the PCM16 grid."""
    v = np.asarray(values, dtype=np.float64)
    return (np.floor(v / 256.0) / 32768.0).astype(np.float32)


def _noise(seconds=2.0, seed=7):
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    return (rng.standard_normal((n, 2)) * 0.15).astype(np.float32)


def _write_pcm16_stem_dir(path, seconds=2.0, seed=7):
    """A directory with one ordinary 16-bit stereo stem."""
    path.mkdir(parents=True, exist_ok=True)
    audio = _noise(seconds, seed)
    ints = np.clip(np.round(audio * 32767.0), -32768, 32767).astype("<i2")
    _write_wav(path / "kick.wav", ints.reshape(-1), SR, 2, 16, 1)
    return path


def _plan_stems(policy, n=3, silent=False, suffix=".wav"):
    """Hand-built stems, with or without decoder provenance."""
    rng = np.random.default_rng(11)
    stems = []
    for i in range(n):
        audio = (rng.standard_normal((int(SR * 1.0), 2)) * 0.2).astype(np.float32)
        stems.append(
            analyze.Stem(
                name=f"s{i}{suffix}",
                path=f"s{i}{suffix}",
                audio=audio,
                sr=SR,
                decode_policy=policy,
            )
        )
    if silent:
        stems.append(
            analyze.Stem(
                name=f"silent{suffix}",
                path=f"silent{suffix}",
                audio=np.zeros((int(SR * 1.0), 2), dtype=np.float32),
                sr=SR,
                decode_policy=policy,
            )
        )
    return stems


def _build_plan(stems):
    return planner.build_plan(stems, planner.load_preset("pop_2020s"))


def _valid_plan(tmp_path, name="plan"):
    """A plan that validates, with a usable source.stems_dir."""
    stems_dir = _write_pcm16_stem_dir(tmp_path / "stems")
    stems = analyze.load_stems(str(stems_dir))
    for s in stems:
        analyze.measure(s)
    plan = _build_plan(stems)
    plan["source"]["stems_dir"] = str(stems_dir)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return plan, path


# ------------------------------------------------------- the decoder itself


@needs_ffmpeg
def test_float32_policy_keeps_every_bit_of_a_24_bit_flac(tmp_path):
    values = _stereo24()
    src = tmp_path / "src24.wav"
    _write_wav(src, values, SR, 2, 24, 1)
    flac = tmp_path / "src24.flac"
    _run_ffmpeg(["-i", str(src), "-c:a", "flac", str(flac)])

    audio, sr = analyze.load_stem_file(str(flac), "float32_v1")
    expected = _expected_float32(values).reshape(-1, 2)

    assert sr == SR
    assert audio.dtype == np.float32
    assert audio.shape == expected.shape
    # not "close": the same samples, bit for bit
    assert np.array_equal(audio, expected)


@needs_ffmpeg
def test_float32_policy_keeps_every_bit_of_a_24_bit_aiff(tmp_path):
    values = _stereo24()
    src = tmp_path / "src24.wav"
    _write_wav(src, values, SR, 2, 24, 1)
    aiff = tmp_path / "src24.aiff"
    _run_ffmpeg(["-i", str(src), "-c:a", "pcm_s24be", str(aiff)])

    audio, sr = analyze.load_stem_file(str(aiff), "float32_v1")
    expected = _expected_float32(values).reshape(-1, 2)

    assert sr == SR
    assert np.array_equal(audio, expected)


@needs_ffmpeg
def test_legacy_policy_reproduces_the_old_pcm16_readback(tmp_path):
    values = _stereo24()
    src = tmp_path / "src24.wav"
    _write_wav(src, values, SR, 2, 24, 1)
    flac = tmp_path / "src24.flac"
    _run_ffmpeg(["-i", str(src), "-c:a", "flac", str(flac)])

    legacy, _ = analyze.load_stem_file(str(flac), "legacy_pcm16_v1")

    # exactly the grid the old decoder used: the value truncated to PCM16
    assert np.array_equal(legacy, _expected_legacy(values).reshape(-1, 2))

    # and exactly what ffmpeg's own 16-bit conversion gives
    s16 = tmp_path / "src16.wav"
    _run_ffmpeg(["-i", str(src), "-c:a", "pcm_s16le", str(s16)])
    independent, _ = analyze._read_wav(str(s16))
    assert np.array_equal(legacy, independent)

    # the low bits really are gone, and the below-16-bit-LSB content with them
    assert not np.array_equal(legacy, _expected_float32(values).reshape(-1, 2))
    quiet = [i for i, v in enumerate(values) if 0 < abs(v) < 128]
    assert quiet
    true_quiet = _expected_float32(values).reshape(-1)[quiet]
    got_quiet = legacy.reshape(-1)[quiet]
    # every legacy sample sits on the 16-bit grid
    grid = got_quiet * 32768.0
    assert np.allclose(grid, np.round(grid), atol=1e-3)
    # and the quiet content is not the content any more
    assert not np.array_equal(got_quiet, true_quiet)
    # the positive quiet values are simply gone
    assert np.any(got_quiet == 0.0)


def test_wav_is_read_the_same_way_under_both_policies(tmp_path):
    pcm16 = tmp_path / "a16.wav"
    v16 = [0, 32767, -32768, 1234, -1234, 100, -100, 9999]
    _write_wav(pcm16, v16, SR, 2, 16, 1)
    a = analyze.load_stem_file(str(pcm16), "float32_v1")[0]
    b = analyze.load_stem_file(str(pcm16), "legacy_pcm16_v1")[0]
    assert np.array_equal(a, b)
    assert np.array_equal(
        a, (np.asarray(v16, dtype=np.float64) / 32768.0).astype(np.float32).reshape(-1, 2)
    )

    f32 = tmp_path / "f32.wav"
    v32 = [0.0, 0.5, -0.5, 0.25, -0.25, 1e-8, -1e-8, 0.75]
    _write_wav(f32, v32, SR, 2, 32, 3)
    a = analyze.load_stem_file(str(f32), "float32_v1")[0]
    b = analyze.load_stem_file(str(f32), "legacy_pcm16_v1")[0]
    assert np.array_equal(a, b)
    assert np.array_equal(a, np.asarray(v32, dtype=np.float32).reshape(-1, 2))

    # float64 WAV is explicitly brought down to float32
    f64 = tmp_path / "f64.wav"
    v64 = [0.0, 0.5, -0.5, 0.25, -0.25, 1e-8, -1e-8, 0.75]
    _write_wav(f64, v64, SR, 2, 64, 3)
    a = analyze.load_stem_file(str(f64), "float32_v1")[0]
    assert a.dtype == np.float32
    assert np.array_equal(a, np.asarray(v64, dtype=np.float32).reshape(-1, 2))


def test_mono_and_multichannel_keep_the_existing_rules(tmp_path):
    mono = tmp_path / "m.wav"
    _write_wav(mono, [0.0, 0.5, -0.5, 0.25], SR, 1, 32, 3)
    out = analyze.load_stem_file(str(mono), "float32_v1")[0]
    assert out.shape == (4, 2)
    assert np.array_equal(out[:, 0], out[:, 1])

    quad = tmp_path / "q.wav"
    values = [0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4]
    _write_wav(quad, values, SR, 4, 32, 3)
    out = analyze.load_stem_file(str(quad), "float32_v1")[0]
    assert out.shape == (2, 2)
    assert np.array_equal(out[:, 0], np.asarray([0.1, -0.1], dtype=np.float32))
    assert np.array_equal(out[:, 1], np.asarray([0.2, -0.2], dtype=np.float32))


@needs_ffmpeg
def test_mp3_decodes_to_the_float_ffmpeg_itself_produces(tmp_path):
    src = tmp_path / "tone.wav"
    tone = (np.sin(np.arange(int(SR * 1.0)) * 2 * np.pi * 220.0) * 0.4)
    stereo = np.stack([tone, tone * 0.5], axis=1).reshape(-1)
    ints = np.clip(np.round(stereo * 32767.0), -32768, 32767).astype("<i2")
    _write_wav(src, ints, SR, 2, 16, 1)
    mp3 = tmp_path / "tone.mp3"
    _run_ffmpeg(["-i", str(src), "-c:a", "libmp3lame", "-b:a", "192k", str(mp3)])

    loaded, _ = analyze.load_stem_file(str(mp3), "float32_v1")

    # an independent float32 conversion of the same file -- this says the
    # loader hands back what ffmpeg decodes to, not that an MP3 was undone
    ref = tmp_path / "ref.wav"
    _run_ffmpeg(["-i", str(mp3), "-ac", "2", "-c:a", "pcm_f32le", str(ref)])
    independent, _ = analyze._read_wav(str(ref))
    assert np.array_equal(loaded, independent)

    # The old decoder still works and still lands on the 16-bit grid -- but
    # an MP3 decoder may hand over 16-bit samples to begin with, in which
    # case the two policies agree here.  That is a property of the codec,
    # not a claim about the loader.
    legacy, _ = analyze.load_stem_file(str(mp3), "legacy_pcm16_v1")
    assert legacy.shape == loaded.shape
    assert np.allclose(legacy * 32768.0, np.round(legacy * 32768.0), atol=1e-3)


def test_an_invalid_policy_is_refused_before_anything_is_read(tmp_path):
    missing = tmp_path / "does-not-exist.wav"
    for bad in ("bogus", "", None, True, 1, ["float32_v1"], "FLOAT32_V1"):
        with pytest.raises(analyze.StemError) as exc:
            analyze.load_stem_file(str(missing), bad)
        assert "decode policy" in str(exc.value)

    # the directory loader refuses before it even looks at the directory
    with pytest.raises(analyze.StemError) as exc:
        analyze.load_stems(str(tmp_path / "no-such-dir"), "bogus")
    assert "decode policy" in str(exc.value)


def test_the_loader_stamps_the_policy_it_used(tmp_path):
    stems_dir = _write_pcm16_stem_dir(tmp_path / "stems")

    # the default changed to float32_v2 in R47 (mono channel rule); an explicit
    # float32_v1 load still stamps float32_v1
    stems = analyze.load_stems(str(stems_dir))
    assert [s.decode_policy for s in stems] == ["float32_v2"]

    stems = analyze.load_stems(str(stems_dir), "float32_v1")
    assert [s.decode_policy for s in stems] == ["float32_v1"]

    stems = analyze.load_stems(str(stems_dir), "legacy_pcm16_v1")
    assert [s.decode_policy for s in stems] == ["legacy_pcm16_v1"]


# ------------------------------------------------------------- the plan


def test_a_plan_records_the_policy_the_stems_agree_on():
    plan = _build_plan(_plan_stems("float32_v1"))
    assert plan["source"]["decode_policy"] == "float32_v1"

    plan = _build_plan(_plan_stems("legacy_pcm16_v1"))
    assert plan["source"]["decode_policy"] == "legacy_pcm16_v1"


def test_a_plan_from_hand_built_stems_records_no_policy():
    plan = _build_plan(_plan_stems(None))
    assert "decode_policy" not in plan["source"]


def test_a_plan_refuses_stems_that_disagree_about_the_decoder():
    stems = _plan_stems("float32_v1", n=2)
    stems.append(_plan_stems("legacy_pcm16_v1", n=1)[0])
    with pytest.raises(planner.PlanError) as exc:
        _build_plan(stems)
    assert "decoder policy" in str(exc.value).lower()


def test_a_plan_refuses_stems_where_only_some_know_their_decoder():
    # a source ffmpeg has to decode: which decoder ran over the untagged
    # stem is not knowable, so the plan refuses instead of guessing
    stems = _plan_stems("float32_v1", n=2, suffix=".flac")
    stems.append(_plan_stems(None, n=1, suffix=".flac")[0])
    with pytest.raises(planner.PlanError) as exc:
        _build_plan(stems)
    assert "only some of the stems" in str(exc.value)


def test_a_plan_accepts_partial_provenance_when_every_source_is_a_wav():
    # both policies read a WAV through the same parser, so the decoder
    # choice cannot change what these files decode to: the plan is built and
    # claims nothing about precision
    stems = _plan_stems("float32_v1", n=2, suffix=".wav")
    stems.append(_plan_stems(None, n=1, suffix=".WAV")[0])
    plan = _build_plan(stems)
    assert "decode_policy" not in plan["source"]


def test_a_plan_refuses_partial_provenance_for_a_non_wav_source():
    # the narrow exception is about the file format only
    stems = _plan_stems("float32_v1", n=2, suffix=".aiff")
    stems.append(_plan_stems(None, n=1, suffix=".aiff")[0])
    with pytest.raises(planner.PlanError) as exc:
        _build_plan(stems)
    assert "only some of the stems" in str(exc.value)


def test_a_non_wav_partial_refusal_comes_before_a_silent_stem_is_dropped():
    # the silent stem is dropped by the planner; the refusal must not wait
    # for that, or dropping it would make the rest look unanimous
    stems = _plan_stems("float32_v1", n=2, suffix=".flac")
    stems.append(_plan_stems(None, n=1, suffix=".flac", silent=True)[-1])
    assert not np.any(stems[-1].audio)  # the appended stem really is silent
    with pytest.raises(planner.PlanError) as exc:
        _build_plan(stems)
    assert "only some of the stems" in str(exc.value)


def test_a_plan_refuses_an_invalid_provenance_value():
    for bad in ([], {}, True, 1, 3.5, "bogus", "FLOAT32_V1"):
        stems = _plan_stems(None, n=1)
        stems[0].decode_policy = bad
        with pytest.raises(planner.PlanError) as exc:
            _build_plan(stems)
        assert "unknown decode policy" in str(exc.value)


def test_an_invalid_provenance_value_is_refused_even_for_wav_sources():
    # the WAV exception must not bless provenance nobody can vouch for
    stems = _plan_stems(None, n=1, suffix=".wav")
    stems[0].decode_policy = ["float32_v1"]
    with pytest.raises(planner.PlanError) as exc:
        _build_plan(stems)
    assert "unknown decode policy" in str(exc.value)


def test_a_dropped_silent_stem_cannot_hide_a_mixed_policy():
    stems = _plan_stems("float32_v1", n=2)
    stems.append(_plan_stems("legacy_pcm16_v1", n=1, silent=True)[-1])
    assert not np.any(stems[-1].audio)  # the appended stem really is silent
    with pytest.raises(planner.PlanError):
        _build_plan(stems)


# ------------------------------------------------------- the schema


def test_the_schema_accepts_the_policy_names_and_nothing_else(tmp_path):
    plan, _ = _valid_plan(tmp_path)

    plan["source"].pop("decode_policy", None)
    planner.validate(plan)  # a plan from before the field still validates

    for good in ("float32_v1", "legacy_pcm16_v1"):
        plan["source"]["decode_policy"] = good
        planner.validate(plan)

    for bad in ("bogus", None, True, 1, "", "Float32_V1"):
        plan["source"]["decode_policy"] = bad
        with pytest.raises(Exception):
            planner.validate(plan)


def test_the_engine_refuses_an_unknown_policy_even_without_the_schema(tmp_path):
    plan, _ = _valid_plan(tmp_path)
    plan["source"]["decode_policy"] = "bogus"
    # the schema check is what usually catches this; the engine's own
    # invariant catches it when a plan is checked against some other schema
    with pytest.raises(planner.PlanError) as exc:
        planner._engine_invariants(plan)
    assert "decode_policy" in str(exc.value)


def test_a_plan_without_a_policy_is_replayed_the_legacy_way():
    plan = {"source": {}}
    assert planner.plan_decode_policy(plan) == "legacy_pcm16_v1"
    plan["source"]["decode_policy"] = "float32_v1"
    assert planner.plan_decode_policy(plan) == "float32_v1"
    plan["source"]["decode_policy"] = "bogus"
    with pytest.raises(planner.PlanError):
        planner.plan_decode_policy(plan)


# ------------------------------------------------------------- the CLI


@needs_ffmpeg
def test_a_saved_plan_without_a_policy_replays_with_the_legacy_decoder(
    tmp_path, monkeypatch
):
    values = _stereo24(SR)  # one second, long enough to be measured
    src = tmp_path / "src24.wav"
    _write_wav(src, values, SR, 2, 24, 1)
    stems_dir = tmp_path / "stems"
    stems_dir.mkdir()
    _run_ffmpeg(["-i", str(src), "-c:a", "flac", str(stems_dir / "a.flac")])

    stems = analyze.load_stems(str(stems_dir))
    for s in stems:
        analyze.measure(s)
    plan = _build_plan(stems)
    plan["source"]["stems_dir"] = str(stems_dir)

    seen = []
    original = analyze.load_stems

    def spy(stem_dir, decode_policy=None):
        if decode_policy is None:
            decode_policy = analyze.DEFAULT_DECODE_POLICY
        seen.append(decode_policy)
        return original(stem_dir, decode_policy)

    monkeypatch.setattr(analyze, "load_stems", spy)

    # a plan written before the field existed
    old = dict(plan)
    old["source"] = {k: v for k, v in plan["source"].items() if k != "decode_policy"}
    old_path = tmp_path / "old.json"
    old_path.write_text(json.dumps(old), encoding="utf-8")
    out_old = tmp_path / "out_old"
    assert cli.main(
        ["--render-plan", str(old_path), "--out", str(out_old), "--quiet"]
    ) == 0
    assert seen == ["legacy_pcm16_v1"]

    # the same plan with the legacy policy recorded replays the same way
    seen.clear()
    legacy = dict(plan)
    legacy["source"] = dict(plan["source"], decode_policy="legacy_pcm16_v1")
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    out_legacy = tmp_path / "out_legacy"
    assert cli.main(
        ["--render-plan", str(legacy_path), "--out", str(out_legacy), "--quiet"]
    ) == 0
    assert seen == ["legacy_pcm16_v1"]
    assert (out_old / "mix.wav").read_bytes() == (out_legacy / "mix.wav").read_bytes()

    # a plan that records the new policy is replayed with it, whatever the
    # current default is
    seen.clear()
    monkeypatch.setattr(analyze, "DEFAULT_DECODE_POLICY", "legacy_pcm16_v1")
    new_path = tmp_path / "new.json"
    new_path.write_text(json.dumps(plan), encoding="utf-8")
    out_new = tmp_path / "out_new"
    assert cli.main(
        ["--render-plan", str(new_path), "--out", str(out_new), "--quiet"]
    ) == 0
    assert seen == ["float32_v2"]  # the default a new plan records since R47

    # and a plan recorded with float32_v1 still replays with float32_v1
    seen.clear()
    v1 = dict(plan)
    v1["source"] = dict(plan["source"], decode_policy="float32_v1")
    v1_path = tmp_path / "v1.json"
    v1_path.write_text(json.dumps(v1), encoding="utf-8")
    assert cli.main(
        ["--render-plan", str(v1_path), "--out", str(tmp_path / "out_v1"), "--quiet"]
    ) == 0
    assert seen == ["float32_v1"]


def test_a_saved_plan_with_a_bad_policy_is_refused_before_the_output_directory(
    tmp_path
):
    plan, _ = _valid_plan(tmp_path)
    for i, bad in enumerate(["bogus", None, True, 1]):
        plan["source"]["decode_policy"] = bad
        path = tmp_path / f"bad{i}.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        out = tmp_path / f"out{i}"
        assert cli.main(["--render-plan", str(path), "--out", str(out), "--quiet"]) == 2
        assert not out.exists()


def test_a_new_plan_records_the_new_policy(tmp_path):
    stems_dir = _write_pcm16_stem_dir(tmp_path / "stems")
    out = tmp_path / "out"
    assert cli.main(
        ["--stems", str(stems_dir), "--out", str(out), "--plan-only", "--quiet"]
    ) == 0
    saved = json.loads((out / "mixplan.json").read_text(encoding="utf-8"))
    assert saved["source"]["decode_policy"] == "float32_v2"


def test_the_policy_survives_a_round_trip_through_the_schema(tmp_path):
    plan, path = _valid_plan(tmp_path)
    plan["source"]["decode_policy"] = "float32_v1"
    path.write_text(json.dumps(plan), encoding="utf-8")
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    planner.validate(reloaded)
    assert planner.plan_decode_policy(reloaded) == "float32_v1"
    assert os.path.exists(path)
