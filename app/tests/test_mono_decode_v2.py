"""A mono source loads at the same level whatever container it came in.

``float32_v1`` let ffmpeg turn a mono FLAC into stereo, and ffmpeg's matrix
puts 1/sqrt(2) on each side, while the WAV reader duplicates mono at unity.
The same quantised samples therefore loaded 3.01 dB quieter from a FLAC than
from a WAV.  ``float32_v2`` decodes at the source's own channel count and uses
the WAV channel rule.  The older policies must keep decoding exactly as they
did, because saved plans replay with them.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from automix import analyze, plan as planner

SR = 48000
needs_ffmpeg = pytest.mark.skipif(not analyze._ffmpeg_available(), reason="ffmpeg not installed")


def _int16(channels: int, n: int = SR) -> np.ndarray:
    """One already-quantised array, so every container holds the same samples."""
    t = np.arange(n) / SR
    cols = [0.4 * np.sin(2 * np.pi * (220.0 + 110.0 * c) * t) for c in range(channels)]
    return np.clip(np.round(np.stack(cols, axis=1) * 32767.0), -32768, 32767).astype(np.int16)


def _write(tmp_path, name: str, pcm: np.ndarray, fmt: str) -> str:
    path = str(tmp_path / name)
    sf.write(path, pcm, SR, subtype="PCM_16", format=fmt)
    back, _ = sf.read(path, dtype="int16", always_2d=True)
    assert np.array_equal(back, pcm)  # the container holds exactly these samples
    return path


def _db(a: np.ndarray, b: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(a.astype(np.float64) ** 2)) /
                         np.sqrt(np.mean(b.astype(np.float64) ** 2)))


@needs_ffmpeg
def test_a_mono_flac_loads_exactly_like_the_same_mono_wav(tmp_path):
    pcm = _int16(1)
    wav = _write(tmp_path, "m.wav", pcm, "WAV")
    flac = _write(tmp_path, "m.flac", pcm, "FLAC")
    from_wav, _ = analyze.load_stem_file(wav, "float32_v2")
    from_flac, sr = analyze.load_stem_file(flac, "float32_v2")
    assert sr == SR and from_flac.dtype == np.float32
    assert np.array_equal(from_flac, from_wav)
    # and that is the native sample at unity on both sides, not a rescale
    native = (pcm[:, 0].astype(np.float64) / 32768.0).astype(np.float32)
    assert np.array_equal(from_flac[:, 0], native) and np.array_equal(from_flac[:, 1], native)


@needs_ffmpeg
def test_float32_v1_keeps_its_old_meaning_for_replay(tmp_path):
    """The difference this ticket removes must still be there under v1."""
    pcm = _int16(1)
    wav = _write(tmp_path, "m.wav", pcm, "WAV")
    flac = _write(tmp_path, "m.flac", pcm, "FLAC")
    v1, _ = analyze.load_stem_file(flac, "float32_v1")
    v2, _ = analyze.load_stem_file(flac, "float32_v2")
    assert abs(_db(v1, v2) - (-3.0103)) < 1e-3
    assert abs(_db(v1, analyze.load_stem_file(wav, "float32_v1")[0]) - (-3.0103)) < 1e-3
    # v1 is still exactly what ffmpeg's own -ac 2 float decode gives
    ref = str(tmp_path / "ref.wav")
    analyze.subprocess.run([analyze._ffmpeg_path(), "-v", "error", "-y", "-i", flac,
                            "-ac", "2", "-c:a", "pcm_f32le", ref], check=True)
    assert np.array_equal(v1, analyze._read_wav(ref)[0])
    legacy, _ = analyze.load_stem_file(flac, "legacy_pcm16_v1")
    assert abs(_db(legacy, v2) - (-3.0103)) < 1e-2


@needs_ffmpeg
@pytest.mark.parametrize("policy", ["float32_v1", "float32_v2"])
def test_stereo_is_the_same_in_every_container_and_policy(tmp_path, policy):
    pcm = _int16(2)
    wav = _write(tmp_path, "s.wav", pcm, "WAV")
    flac = _write(tmp_path, "s.flac", pcm, "FLAC")
    assert np.array_equal(analyze.load_stem_file(flac, policy)[0],
                          analyze.load_stem_file(wav, policy)[0])


@needs_ffmpeg
def test_more_than_two_channels_are_still_downmixed_by_ffmpeg(tmp_path):
    pcm = _int16(4)
    flac = _write(tmp_path, "q.flac", pcm, "FLAC")
    v1, _ = analyze.load_stem_file(flac, "float32_v1")
    v2, _ = analyze.load_stem_file(flac, "float32_v2")
    assert v2.shape == v1.shape and np.array_equal(v1, v2)


@needs_ffmpeg
@pytest.mark.parametrize("channels, full_args", [(1, []), (2, []), (4, ["-ac", "2"])])
def test_v2_decodes_the_whole_file_once_at_the_right_width(tmp_path, monkeypatch, channels, full_args):
    """Only a 0.1 s probe runs at the native width; a surround file is never decoded whole at it."""
    flac = _write(tmp_path, "x.flac", _int16(channels), "FLAC")
    calls = []
    real = analyze.subprocess.run

    def spy(cmd, *args, **kwargs):
        calls.append(cmd[cmd.index("-i") + 2 : cmd.index("-c:a")])
        return real(cmd, *args, **kwargs)

    monkeypatch.setattr(analyze.subprocess, "run", spy)
    analyze.load_stem_file(flac, "float32_v2")
    assert calls == [["-t", "0.1"], full_args]


@needs_ffmpeg
def test_mono_and_stereo_sources_keep_their_relative_balance(tmp_path):
    """The bug's real cost: mixed formats shifted one stem against another."""
    mono, stereo = _int16(1), _int16(2)
    loaded = {}
    for fmt, ext in (("WAV", "wav"), ("FLAC", "flac")):
        m, _ = analyze.load_stem_file(_write(tmp_path, f"m.{ext}", mono, fmt), "float32_v2")
        s, _ = analyze.load_stem_file(_write(tmp_path, f"s.{ext}", stereo, fmt), "float32_v2")
        loaded[ext] = _db(m, s)
    assert loaded["wav"] == loaded["flac"]


@needs_ffmpeg
def test_new_imports_record_v2_and_replay_with_what_was_recorded(tmp_path):
    stems_dir = tmp_path / "stems"
    stems_dir.mkdir()
    _write(stems_dir, "voice.flac", _int16(1), "FLAC")
    _write(stems_dir, "pad.wav", _int16(2), "WAV")
    stems = analyze.load_stems(str(stems_dir))
    assert [s.decode_policy for s in stems] == ["float32_v2", "float32_v2"]
    analyze.detect_roles(stems)
    analyze.align(stems)
    for st in stems:
        analyze.measure(st)
    plan = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    planner.validate(plan)
    assert plan["source"]["decode_policy"] == "float32_v2"
    assert planner.plan_decode_policy(plan) == "float32_v2"
    for old in ("float32_v1", "legacy_pcm16_v1"):
        plan["source"]["decode_policy"] = old
        planner.validate(plan)
        assert planner.plan_decode_policy(plan) == old
    del plan["source"]["decode_policy"]
    planner.validate(plan)
    assert planner.plan_decode_policy(plan) == "legacy_pcm16_v1"
    plan["source"]["decode_policy"] = "float32_v3"
    with pytest.raises(Exception):
        planner.validate(plan)


def test_the_wav_path_is_the_same_under_every_policy(tmp_path):
    pcm = _int16(1)
    wav = _write(tmp_path, "m.wav", pcm, "WAV")
    outs = [analyze.load_stem_file(wav, p)[0] for p in analyze.DECODE_POLICIES]
    assert all(np.array_equal(outs[0], o) for o in outs[1:])


@needs_ffmpeg
@pytest.mark.parametrize("policy", ["float32_v1", "float32_v2"])
def test_the_balance_proof_decodes_the_reference_like_the_sources(tmp_path, monkeypatch, policy):
    """A same-content proof is sample by sample: both sides need one decoder."""
    from automix import source_balance

    stems_dir = tmp_path / "stems"
    stems_dir.mkdir()
    pcm = _int16(1, n=SR * 12)
    _write(stems_dir, "voice.flac", pcm, "FLAC")
    ref = _write(tmp_path, "ref.flac", pcm, "FLAC")
    stems = analyze.load_stems(str(stems_dir), policy)
    seen = []
    real = analyze.load_stem_file

    def spy(path, decode_policy=analyze.DEFAULT_DECODE_POLICY):
        seen.append(decode_policy)
        return real(path, decode_policy)

    monkeypatch.setattr(analyze, "load_stem_file", spy)
    for profile in ("auto", "reference_balance"):
        seen.clear()
        source_balance.resolve(stems, {"path": ref}, profile)
        assert seen == [policy], profile
    # and with one decoder on both sides the same content proves itself at unity
    evidence = source_balance.proof(stems, {"path": ref})
    assert evidence["passed"] and abs(evidence["gain_db"]) < 1e-6
