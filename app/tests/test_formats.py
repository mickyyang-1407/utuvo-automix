"""Non-WAV input: FLAC, MP3, AIFF.

The WAV path is tested in detail elsewhere; these formats go through ffmpeg,
which is a completely separate code path that is easy to break and impossible
to notice until a user drops an MP3.  The tests shell out to ffmpeg to build the
fixtures, so they are skipped when it is not installed.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from automix import analyze

SR = 48000
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") and not __import__("os").path.exists(FFMPEG),
    reason="ffmpeg is required for non-WAV decoding",
)


def _tone_wav(path, dur=2.0, freq=440.0, amp=0.3):
    n = int(dur * SR)
    t = np.arange(n) / SR
    mono = amp * np.sin(2 * np.pi * freq * t)
    pcm = (np.stack([mono, mono], 1) * 32767).astype("<i2")
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return str(path)


def _convert(src, dst, codec):
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", src] + codec + [dst],
        check=True,
    )
    return str(dst)


@pytest.mark.parametrize(
    "suffix,codec",
    [
        (".flac", ["-c:a", "flac"]),
        (".aiff", []),
        (".mp3", ["-c:a", "libmp3lame", "-b:a", "192k"]),
    ],
)
def test_compressed_and_container_formats_decode(tmp_path, suffix, codec):
    src = _tone_wav(tmp_path / "tone.wav")
    path = _convert(src, tmp_path / f"tone{suffix}", codec)

    audio, sr = analyze.load_stem_file(path)
    assert sr == SR
    assert audio.ndim == 2 and audio.shape[1] == 2
    assert audio.shape[0] == SR * 2

    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    # lossless must match the WAV closely; MP3 is allowed more slack
    tolerance = 0.02 if suffix == ".mp3" else 0.002
    assert abs(rms - 0.3 / np.sqrt(2)) < tolerance, (suffix, rms)


def test_a_corrupt_compressed_file_is_a_clear_error(tmp_path):
    path = tmp_path / "broken.flac"
    path.write_bytes(b"not really a flac file")
    with pytest.raises(analyze.StemError) as exc:
        analyze.load_stem_file(str(path))
    assert "ffmpeg failed" in str(exc.value) or "not a WAV" in str(exc.value)


def test_mixed_container_types_mix_together(tmp_path):
    """A session can legitimately hold a WAV, a FLAC and an MP3."""
    from automix import pipeline
    from automix import plan as planner

    d = tmp_path / "session"
    d.mkdir()
    _tone_wav(d / "bass.wav", freq=55.0)
    _convert(str(d / "bass.wav"), d / "vocal_lead.flac", ["-c:a", "flac"])
    _convert(str(d / "bass.wav"), d / "guitar_elec.mp3", ["-c:a", "libmp3lame", "-b:a", "192k"])
    (d / "bass.wav").unlink()

    stems = analyze.load_stems(str(d))
    assert len(stems) == 2
    analyze.detect_roles(stems)
    analyze.align(stems)
    for st in stems:
        analyze.measure(st)
    plan, audio, report = pipeline.mix(stems, planner.load_preset("pop_2020s"))
    assert len(plan["tracks"]) == 2
    assert report["lufs"] < 0.0


def test_ffmpeg_is_found_outside_path_where_package_managers_put_it(monkeypatch, tmp_path):
    """Opened from Finder the App has a bare PATH; Intel Homebrew must still work."""
    fake = tmp_path / "ffmpeg"
    fake.write_text("")
    monkeypatch.setattr(analyze.shutil, "which", lambda name: None)
    monkeypatch.setattr(analyze, "FFMPEG_FALLBACKS", ("/nonexistent/ffmpeg", str(fake)))
    assert analyze._ffmpeg_path() == str(fake) and analyze._ffmpeg_available()
    monkeypatch.setattr(analyze, "FFMPEG_FALLBACKS", ("/nonexistent/ffmpeg",))
    assert analyze._ffmpeg_path() is None and not analyze._ffmpeg_available()
