"""R128: the drum bus's loudest peaks clipped before the master limiter.

"Clip, then limit": a limiter that catches a snare's first millisecond turns
the whole mix down for its release; a clipper takes the same peak off in the
waveform, for the length of the peak only.  The ceiling comes from the bus's
own loud hits, so a quiet bus and a loud one are clipped by the same dB; it
is 8x oversampled; and it is a plan field (buses.drum.clip): plans without it
are untouched (test_replay_r11*.py).
"""
from __future__ import annotations

import copy
import glob
import os

import jsonschema
import numpy as np
import pytest

from automix import analyze, dsp, plan as planner, render

SR = 48000
SPEC = {"law": "drum_clip_os8_v1", "depth_db": 2.0, "knee_db": 1.0}
HIT = SR // 10


def _kick(n=SR // 4):
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * 55 * t) * np.exp(-t / 0.08)


def _snare(n=SR // 4, seed=0):
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    return 0.6 * rng.standard_normal(n) * np.exp(-t / 0.04) + 0.5 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.06)


def _groove(seconds=4, loud_every=4):
    """kick and snare alternating every 0.25 s; every loud_every-th hit 6 dB up, a quiet hat bed throughout"""
    n = SR * seconds
    x = np.zeros(n)
    rng = np.random.default_rng(1)
    for k, start in enumerate(range(HIT, n - SR // 4, SR // 4)):
        one = _kick() if k % 2 == 0 else _snare(seed=k)
        x[start:start + one.size] += (0.5 if k % loud_every == 0 else 0.25) * one
    x += 0.005 * rng.standard_normal(n)
    return np.stack([x, 0.9 * x], 1)


def test_a_quiet_bus_and_a_loud_bus_are_clipped_alike():
    x = _groove()
    y, rec = dsp.drum_clip(x, SR, SPEC)
    k = 10 ** (-30 / 20)
    y2, rec2 = dsp.drum_clip(x * k, SR, SPEC)
    assert np.max(np.abs(y2 - y * k)) <= 1e-9 * np.max(np.abs(y * k))
    assert rec == rec2
    assert rec["peak_cut_db"] < -1.0


def test_the_ceiling_sits_depth_db_under_the_loud_hits():
    x = _groove()
    mag = np.max(np.abs(x), axis=1)
    cell = SR // 100
    peaks = mag[:mag.size // cell * cell].reshape(-1, cell).max(axis=1)
    active = peaks[peaks > peaks.max() * 10 ** (-40 / 20)]
    want = np.percentile(active, 99.0) * 10 ** (-2.0 / 20)
    assert dsp.drum_clip_ceiling(x, SR, 2.0) == pytest.approx(want, rel=1e-12)
    assert dsp.drum_clip_ceiling(x, SR, 3.0) == pytest.approx(want * 10 ** (-1.0 / 20), rel=1e-12)
    # the channels are linked: the louder side sets it, whichever side that is
    assert dsp.drum_clip_ceiling(x[:, ::-1].copy(), SR, 2.0) == pytest.approx(want, rel=1e-12)


def test_the_loud_hits_come_down_and_the_quiet_ones_do_not():
    # a low kick (little above the original Nyquist to lose on the way back down)
    n = SR * 4
    x = np.zeros(n)
    loud, quiet = [], []
    for k, start in enumerate(range(HIT, n - SR // 4, SR // 4)):
        level = 0.8 if k % 4 == 0 else 0.2
        x[start:start + SR // 4] += level * _kick()
        (loud if k % 4 == 0 else quiet).append(start)
    x = np.stack([x, x], 1)
    y, rec = dsp.drum_clip(x, SR, SPEC)
    ceiling = dsp.drum_clip_ceiling(x, SR, 2.0)
    for s in loud:
        seg = slice(s, s + SR // 20)
        assert np.max(np.abs(y[seg])) <= ceiling * 10 ** (0.3 / 20)
        assert np.max(np.abs(y[seg])) < np.max(np.abs(x[seg])) * 10 ** (-1.5 / 20)
    for s in quiet:
        seg = slice(s + SR // 100, s + SR // 5)
        assert np.max(np.abs(y[seg] - x[seg])) < 1e-3 * np.max(np.abs(x[seg]))
    assert 0.0 < rec["time_clipped"] < 0.05
    assert rec["distortion_db"] < -20
    # the receipt's loud-hit number: the median of how far each 10 ms cell at or above the
    # reference came down (the reference is the 99th percentile cell, so the hits at the very
    # top come down a little more than depth_db)
    cell = SR // 100
    c0 = np.max(np.abs(x), axis=1)[:x.shape[0] // cell * cell].reshape(-1, cell).max(axis=1)
    c1 = np.max(np.abs(y), axis=1)[:y.shape[0] // cell * cell].reshape(-1, cell).max(axis=1)
    loud = c0 >= ceiling * 10 ** (2.0 / 20) * (1 - 1e-12)
    assert rec["loud_hit_cut_db"] == pytest.approx(round(float(np.median(20 * np.log10(c1[loud] / c0[loud]))), 2), abs=0.011)
    assert -3.5 < rec["loud_hit_cut_db"] < -1.6
    assert rec["peak_cut_db"] <= rec["loud_hit_cut_db"] + 0.05


def test_the_curve_is_the_identity_up_to_the_knee_and_never_passes_the_ceiling():
    u = np.linspace(-3, 3, 200001)
    y = dsp._soft_clip_curve(u, 1.0, 10 ** (-1 / 20))
    knee = 10 ** (-1 / 20)
    inside = np.abs(u) <= knee
    assert np.array_equal(y[inside], u[inside])
    assert np.max(np.abs(y)) <= 1.0                         # tanh reaches it only in float rounding
    assert np.all(np.diff(y) >= 0)                          # monotonic
    step = u[1] - u[0]
    slope = np.diff(y) / step
    assert np.max(np.abs(np.diff(slope))) < 1e-3            # no corner at the knee


def test_untouched_blocks_are_bit_identical_and_far_samples_barely_move():
    x = _groove(seconds=20)
    x[:SR * 12] *= 0.05                                      # the first 12 s never reach the knee
    y, _ = dsp.drum_clip(x, SR, SPEC)
    assert np.array_equal(y[:SR * 7], x[:SR * 7])            # a whole 8 s block with nothing hot
    ceiling = dsp.drum_clip_ceiling(x, SR, 2.0)
    hot = np.max(np.abs(x), axis=1) > ceiling * 10 ** (-1 / 20)
    near = np.convolve(hot.astype(float), np.ones(int(0.05 * SR)), mode="same") > 0
    far = ~near
    # measured 8.7e-5; resampling the whole block there and back (instead of only what was cut) gave 4e-4
    assert np.max(np.abs(y[far] - x[far])) < 2e-4 * np.max(np.abs(x))


def test_a_hit_on_a_block_edge_is_clipped_as_if_there_were_no_blocks():
    n = SR * 12
    x = np.zeros(n)
    for k, start in enumerate(range(HIT, n - SR // 4, SR // 4)):
        x[start:start + SR // 4] += (0.5 if k % 4 == 0 else 0.25) * _snare(seed=k)
    edge = 8 * SR - int(0.002 * SR)                          # the 8 s blocks meet 2 ms into this hit
    x[edge:edge + SR // 4] += 0.6 * _snare(seed=99)
    x = np.stack([x, x], 1)
    blocked, _ = dsp.drum_clip(x, SR, SPEC)
    whole, _ = dsp.drum_clip(x, SR, SPEC, block_s=1000.0)
    # measured 1.5e-5; with no real neighbours either side of a block it was 9.5e-3
    assert np.max(np.abs(blocked - whole)) < 1e-4 * np.max(np.abs(x))


def test_the_channels_are_linked_whichever_side_is_louder():
    x = _groove()                                            # the right side is 0.9 x the left
    y, rec = dsp.drum_clip(x, SR, SPEC)
    flipped, rec2 = dsp.drum_clip(x[:, ::-1].copy(), SR, SPEC)
    assert np.allclose(flipped, y[:, ::-1], atol=1e-12) and rec == rec2


def test_mono_and_float32_keep_their_shape_and_type():
    x = _groove()[:, 0].astype(np.float32)
    y, _ = dsp.drum_clip(x, SR, SPEC)
    assert y.shape == x.shape and y.dtype == np.float32


def test_eight_times_oversampling_keeps_the_harmonics_from_folding_back():
    t = np.arange(SR * 2) / SR
    x = 0.9 * np.sin(2 * np.pi * 9000.0 * t)
    x = np.stack([x, x], 1)
    y, _ = dsp.drum_clip(x, SR, {**SPEC, "depth_db": 3.0})
    seg = y[SR // 2:SR // 2 + SR, 0] * np.hanning(SR)
    spec = np.abs(np.fft.rfft(seg))
    f = np.fft.rfftfreq(SR, 1 / SR)
    fund = spec[np.argmin(np.abs(f - 9000))]
    # 3rd harmonic 27 kHz would fold to 21 kHz, the 5th (45 kHz) to 3 kHz at 1x; 4x left -56 dB at 21 kHz
    for fold in (21000.0, 15000.0, 3000.0):
        assert 20 * np.log10(spec[np.argmin(np.abs(f - fold))] / fund) < -75
    naive = np.clip(x[:, 0], -0.9 * 10 ** (-3 / 20), 0.9 * 10 ** (-3 / 20))
    ns = np.abs(np.fft.rfft(naive[SR // 2:SR // 2 + SR] * np.hanning(SR)))
    assert 20 * np.log10(ns[np.argmin(np.abs(f - 21000.0))] / fund) > -40   # what 1x clipping would do


@pytest.mark.parametrize("n", [1000, 1001, 12345])
@pytest.mark.parametrize("stereo", [False, True])
def test_the_oversampling_comes_back_exactly(n, stereo):
    # only what the curve cut off goes back down, so the way up and the way down must be
    # exact inverses -- also when the block's length is padded to a faster FFT size
    rng = np.random.default_rng(n)
    x = rng.standard_normal((n, 2) if stereo else n)
    X = np.fft.rfft(x, axis=0)
    X[int(0.45 * X.shape[0]):] = 0                          # band-limited, as audio under Nyquist is
    x = np.fft.irfft(X, n, axis=0)
    up, nf = dsp._os_up(x, 8)
    assert up.shape[0] == 8 * nf
    assert np.allclose(up[:8 * n:8], x, atol=1e-9)           # it passes through the original samples
    assert np.max(np.abs(dsp._os_down(up, 8, nf, n) - x)) < 1e-12


def test_silence_is_returned_as_it_came():
    x = np.zeros((SR, 2))
    y, rec = dsp.drum_clip(x, SR, SPEC)
    assert y is x and rec["peak_cut_db"] == 0.0 and rec["loud_hit_cut_db"] == 0.0


def test_a_wrong_law_is_refused():
    with pytest.raises(ValueError):
        dsp.drum_clip(_groove(), SR, {**SPEC, "law": "nope"})


# ------------------------------------------------------------- plan + render

def _session(seconds=6):
    n = SR * seconds
    t = np.arange(n) / SR
    rng = np.random.default_rng(3)

    def loop(one, every, offset=0):
        out = np.zeros(n)
        for start in range(HIT + offset, n - one.size, int(every * SR)):
            out[start:start + one.size] += one
        return out

    parts = {"Kick.wav": 0.6 * loop(_kick(), 0.5),
             "Snare.wav": 0.6 * loop(_snare(), 1.0, SR // 4),
             "Hi Hat.wav": 0.1 * loop(rng.standard_normal(SR // 20) * np.exp(-np.arange(SR // 20) / (0.01 * SR)), 0.25),
             "Bass.wav": 0.3 * np.sin(2 * np.pi * 82 * t),
             "Lead Vocal.wav": 0.2 * np.sin(2 * np.pi * 330 * t) * ((t % 2.0) < 1.5)}
    stems = [analyze.Stem(name=name, path=name, sr=SR, audio=np.stack([a, a], 1).astype(np.float32))
             for name, a in parts.items()]
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    return stems


@pytest.fixture(scope="module")
def stems():
    return _session()


def _clip(p):
    return (p["buses"].get("drum") or {}).get("clip")


def test_new_plans_clip_the_drum_bus_by_the_style(stems):
    rock = planner.load_preset("rock")
    p = planner.build_plan(stems, rock)
    planner.validate(p)
    amount = rock["techniques"]["drum_clip_db"]
    assert amount > 0
    assert _clip(p) == {"law": "drum_clip_os8_v1", "depth_db": round(amount, 2), "knee_db": 1.0}
    half = planner.build_plan(stems, rock, intensity=0.5)
    assert _clip(half)["depth_db"] == round(amount * 0.5, 2)
    for name, bus in p["buses"].items():
        assert name == "drum" or "clip" not in bus


@pytest.mark.parametrize("style", ["jazz_natural", "ambient_space", "acoustic", "lofi"])
def test_styles_that_ask_for_none_get_none(stems, style):
    assert _clip(planner.build_plan(stems, planner.load_preset(style))) is None


def test_under_a_quarter_db_nothing_is_written(stems):
    preset = copy.deepcopy(planner.load_preset("rock"))
    preset["techniques"]["drum_clip_db"] = 0.4
    assert _clip(planner.build_plan(stems, preset, intensity=0.5)) is None     # 0.2 dB
    assert _clip(planner.build_plan(stems, preset, intensity=0.75))["depth_db"] == 0.3


def test_every_shipped_style_states_its_amount():
    names = [os.path.basename(f)[:-5] for f in glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(planner.__file__))), "presets", "*.json"))]
    assert len(names) >= 14
    for name in names:
        amount = planner.load_preset(name)["techniques"]["drum_clip_db"]
        assert 0.0 <= amount <= 6.0, name


def test_the_amount_is_capped(stems):
    preset = copy.deepcopy(planner.load_preset("rock"))
    preset["techniques"]["drum_clip_db"] = 20.0
    p = planner.build_plan(stems, preset)
    assert _clip(p)["depth_db"] == 6.0
    planner.validate(p)


def test_a_plan_without_drums_gets_no_clip():
    rng = np.random.default_rng(5)
    n = SR * 3
    t = np.arange(n) / SR
    parts = {"Bass.wav": 0.3 * np.sin(2 * np.pi * 82 * t), "Lead Vocal.wav": 0.2 * np.sin(2 * np.pi * 330 * t),
             "Keys.wav": 0.1 * rng.standard_normal(n)}
    st = [analyze.Stem(name=k, path=k, sr=SR, audio=np.stack([v, v], 1).astype(np.float32)) for k, v in parts.items()]
    analyze.detect_roles(st)
    for s in st:
        analyze.measure(s)
    p = planner.build_plan(st, planner.load_preset("rock"))
    assert "clip" not in (p["buses"].get("drum") or {})


def test_the_render_clips_last_on_the_drum_bus_and_says_so(stems, tmp_path):
    p = planner.build_plan(stems, planner.load_preset("rock"))
    mix, rep = render.render(stems, p)
    got = rep["mix_moves"]["drum_clip"]
    assert got["law"] == "drum_clip_os8_v1" and got["peak_cut_db"] < -0.5 and got["distortion_db"] < -15
    path = tmp_path / "report.md"
    render.write_report(str(path), p, rep, stems)
    assert "drum peaks clipped before the limiter" in path.read_text(encoding="utf-8")

    off = copy.deepcopy(p)
    off["buses"]["drum"].pop("clip")
    mix0, rep0 = render.render(stems, off)
    assert "drum_clip" not in (rep0.get("mix_moves") or {})
    assert not np.array_equal(np.asarray(mix), np.asarray(mix0))


def test_the_clip_is_the_last_step_on_the_drum_bus(stems, monkeypatch):
    p = planner.build_plan(stems, planner.load_preset("rock"))
    p["buses"]["drum"]["width"] = 1.4
    widened, clipped = [], []
    real_width, real_clip = render._set_width, render._drum_clip

    def width(sig, w):
        out = real_width(sig, w)
        widened.append(out)
        return out

    def clip(sig, sr, spec):
        clipped.append(sig)
        return real_clip(sig, sr, spec)

    monkeypatch.setattr(render, "_set_width", width)
    monkeypatch.setattr(render, "_drum_clip", clip)
    render.render(stems, p)
    # what the clipper got is exactly what the drum bus's width step had just handed back
    assert len(clipped) == 1 and any(clipped[0] is w for w in widened)


def test_the_master_limiter_has_less_to_do(stems):
    p = planner.build_plan(stems, planner.load_preset("rock"))
    p["buses"]["master"]["limiter"] = {"ceiling_db": -1.0, "law": "tp_release_v1"}
    off = copy.deepcopy(p)
    off["buses"]["drum"].pop("clip")
    _, rep = render.render(stems, p)
    _, rep0 = render.render(stems, off)
    assert rep["lufs"] == pytest.approx(rep0["lufs"], abs=0.2)
    assert rep["limiter"]["gain_reduction_p95_db"] > rep0["limiter"]["gain_reduction_p95_db"]


@pytest.mark.parametrize("bad", [{"law": "nope"}, {"depth_db": 0.1}, {"depth_db": 7.0}, {"knee_db": 0.0},
                                 {"knee_db": 7.0}, {"depth_db": float("nan")}, {"depth_db": True},
                                 {"extra": 1}])
def test_a_wrong_clip_block_is_refused_by_the_renderer(stems, bad):
    p = planner.build_plan(stems, planner.load_preset("rock"))
    p["buses"]["drum"]["clip"].update(bad)
    with pytest.raises(render.RenderError):
        render.render(stems, p)


def test_a_clip_block_missing_a_field_is_refused_by_the_renderer(stems):
    p = planner.build_plan(stems, planner.load_preset("rock"))
    del p["buses"]["drum"]["clip"]["knee_db"]
    with pytest.raises(render.RenderError):
        render.render(stems, p)


@pytest.mark.parametrize("bad", [{"law": "nope"}, {"depth_db": 7.0}, {"knee_db": 0.1}, {"extra": 1}])
def test_the_schema_refuses_it_too(stems, bad):
    p = planner.build_plan(stems, planner.load_preset("rock"))
    p["buses"]["drum"]["clip"].update(bad)
    with pytest.raises(jsonschema.ValidationError):
        planner.validate(p)


@pytest.mark.parametrize("bus", ["vocal", "bass", "music", "master"])
def test_only_the_drum_bus_has_one(stems, bus, schema_free):
    p = planner.build_plan(stems, planner.load_preset("rock"))
    p["buses"].setdefault(bus, {})["clip"] = copy.deepcopy(_clip(p))
    with pytest.raises(planner.PlanError):
        planner.validate(p, schema_free)


def test_a_clip_with_no_drum_track_is_refused(stems, schema_free):
    p = planner.build_plan(stems, planner.load_preset("rock"))
    p["tracks"] = [t for t in p["tracks"] if planner.ROLE_GROUP.get(t["role"], "music") != "drum"]
    p.pop("drum_automation", None)
    for t in p["tracks"]:
        t.pop("transient", None)
    p["buses"]["drum"].pop("parallel", None)
    p["buses"]["drum"].pop("balance", None)
    p["buses"]["bass"].pop("duck", None)
    with pytest.raises(planner.PlanError, match="buses.drum.clip"):
        planner.validate(p, schema_free)
