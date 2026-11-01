"""WAV decoding tests.

The stdlib's ``wave`` module refuses IEEE-float WAVs outright ("unknown format:
3"), and 32-bit float is a standard DAW export.  So the reader is hand-written
and these tests cover every container shape it claims to support, plus the
garbage it must reject with a clear message instead of a traceback.
"""

from __future__ import annotations

import os
import struct

import numpy as np
import pytest

from automix import analyze

SR = 48000


def riff(fmt_tag: int, channels: int, sr: int, bits: int, raw: bytes) -> bytes:
    block = channels * (bits // 8)
    fmt = struct.pack("<HHIIHH", fmt_tag, channels, sr, sr * block, block, bits)
    chunks = (
        b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"data" + struct.pack("<I", len(raw)) + raw
    )
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def write_wav(path, data: np.ndarray, fmt_tag: int, bits: int, channels: int = 2) -> str:
    mono = np.asarray(data, dtype=np.float64)
    sig = np.stack([mono] * channels, axis=1)
    if fmt_tag == 3:
        raw = (sig.astype("<f4") if bits == 32 else sig.astype("<f8")).tobytes()
    elif bits == 8:
        raw = ((sig * 127.0) + 128.0).astype(np.uint8).tobytes()
    elif bits == 16:
        raw = (sig * 32767.0).astype("<i2").tobytes()
    elif bits == 24:
        v = (sig * 8388607.0).astype(np.int32).reshape(-1)
        b = np.empty((v.size, 3), dtype=np.uint8)
        u = v & 0xFFFFFF
        b[:, 0] = u & 0xFF
        b[:, 1] = (u >> 8) & 0xFF
        b[:, 2] = (u >> 16) & 0xFF
        raw = b.tobytes()
    elif bits == 32:
        raw = (sig * 2147483647.0).astype("<i4").tobytes()
    else:
        raise ValueError(bits)
    with open(path, "wb") as fh:
        fh.write(riff(fmt_tag, channels, SR, bits, raw))
    return str(path)


def tone(dur=1.0, freq=440.0, amp=0.25):
    n = int(dur * SR)
    return amp * np.sin(2 * np.pi * freq * np.arange(n) / SR)


@pytest.mark.parametrize("bits", [8, 16, 24, 32])
def test_pcm_widths_decode(tmp_path, bits):
    path = write_wav(tmp_path / f"pcm{bits}.wav", tone(), 1, bits)
    audio, sr = analyze.load_stem_file(path)
    assert sr == SR
    assert audio.shape == (SR, 2)
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    assert abs(rms - 0.25 / np.sqrt(2)) < 0.01, (bits, rms)


@pytest.mark.parametrize("bits", [32, 64])
def test_ieee_float_decodes(tmp_path, bits):
    """The case the stdlib cannot read at all."""
    path = write_wav(tmp_path / f"f{bits}.wav", tone(), 3, bits)
    audio, sr = analyze.load_stem_file(path)
    assert audio.shape == (SR, 2)
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    assert abs(rms - 0.25 / np.sqrt(2)) < 1e-4


def test_wave_format_extensible_is_unwrapped(tmp_path):
    """DAWs write 0xFFFE with the real tag inside a GUID."""
    path = write_wav(tmp_path / "x24.wav", tone(), 1, 24)
    data = bytearray(open(path, "rb").read())
    i = data.find(b"fmt ")
    size = struct.unpack_from("<I", data, i + 4)[0]
    fmt = bytearray(data[i + 8 : i + 8 + size])
    fmt[0:2] = struct.pack("<H", 0xFFFE)
    # WAVEFORMATEXTENSIBLE: base 16 bytes + cbSize(2) + validBits(2) +
    # channelMask(4) + subformat GUID(16) == 40 bytes, subformat at offset 24
    fmt += struct.pack("<H", 22)          # cbSize
    fmt += struct.pack("<H", 24)          # wValidBitsPerSample
    fmt += struct.pack("<I", 3)           # dwChannelMask (L + R)
    fmt += b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
    data[i + 4 : i + 8] = struct.pack("<I", len(fmt))
    open(path, "wb").write(bytes(data[: i + 8]) + bytes(fmt) + bytes(data[i + 8 + size :]))

    audio, sr = analyze.load_stem_file(path)
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    assert abs(rms - 0.25 / np.sqrt(2)) < 0.01


def test_mono_is_duplicated_to_stereo(tmp_path):
    path = write_wav(tmp_path / "mono.wav", tone(), 1, 16, channels=1)
    audio, _ = analyze.load_stem_file(path)
    assert audio.shape == (SR, 2)
    np.testing.assert_allclose(audio[:, 0], audio[:, 1], atol=1e-6)


def test_garbage_is_rejected_with_a_clear_message(tmp_path):
    path = tmp_path / "bad.wav"
    path.write_bytes(b"this is not a wav file")
    with pytest.raises(analyze.StemError) as exc:
        analyze.load_stem_file(str(path))
    assert "not a RIFF/WAVE" in str(exc.value)


def test_truncated_data_chunk_still_loads_what_is_there(tmp_path):
    path = write_wav(tmp_path / "t.wav", tone(), 1, 16)
    raw = open(path, "rb").read()
    open(path, "wb").write(raw[: len(raw) // 2])
    audio, sr = analyze.load_stem_file(path)
    assert sr == SR
    assert 0 < audio.shape[0] < SR


def test_missing_data_chunk_is_an_error(tmp_path):
    path = tmp_path / "nodata.wav"
    fmt = struct.pack("<HHIIHH", 1, 2, SR, SR * 4, 4, 16)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    path.write_bytes(b"RIFF" + struct.pack("<I", 4 + len(body)) + body)
    with pytest.raises(analyze.StemError) as exc:
        analyze.load_stem_file(str(path))
    assert "data chunk" in str(exc.value)
