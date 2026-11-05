"""R120: the kit's microphones lined up in time and polarity (align.py).

Several mics on one kit hear each hit a few ms apart; summed, they comb and
a reversed mic cancels the low end.  The plan moves a mic only when the
evidence is clear (GCC-PHAT on the instrument's own hits, a peak that stands
out, four groups of hits that agree), and it leaves alone what it cannot be
sure of: uncorrelated mics, a pure-ambience mic, an already aligned pair, room
and tom mics, and anything the user declared as a capture group.  A plan
without the field renders as before (test_replay_r11*.py).
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from automix import align, analyze, dsp, plan as planner, render

SR = 48000
RNG = np.random.default_rng(7)


def _hit_train(make, times, n):
    x = np.zeros(n)
    for i, tm in enumerate(times):
        h = make(i)
        s = int(tm * SR)
        e = min(n, s + h.size)
        x[s:e] += h[: e - s] * (0.75 + 0.25 * ((i * 7) % 5) / 4)
    return x


def _kick(i):
    t = np.arange(int(0.4 * SR)) / SR
    f = 50 + 120 * np.exp(-t * 40)
    r = np.random.default_rng(100 + i)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9) + 0.3 * r.standard_normal(t.size) * np.exp(-t * 200)


def _snare(i):
    t = np.arange(int(0.3 * SR)) / SR
    r = np.random.default_rng(200 + i)
    return 0.6 * np.sin(2 * np.pi * 190 * t) * np.exp(-t * 25) + r.standard_normal(t.size) * np.exp(-t * 30)


def _later(x, ms):
    return align.advance(x, SR, -ms)


def _lowpass(x, fc):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, 1 / SR)
    return np.fft.irfft(X / np.sqrt(1 + (f / fc) ** 4), x.size)


def _room_ir():
    ir = np.zeros(int(0.08 * SR))
    ir[0] = 1.0
    r = np.random.default_rng(3)
    ir[r.integers(200, ir.size, 40)] = r.standard_normal(40) * 0.25
    return ir


DUR = 24.0
N = int(DUR * SR)
KICK = _hit_train(_kick, np.arange(0.3, DUR - 1, 1.0), N)
SNARE = _hit_train(_snare, np.arange(0.8, DUR - 1, 1.0), N)


@pytest.mark.parametrize("ms, polarity", [(0.37, 1), (1.9, -1), (3.33, 1), (5.21, -1), (7.8, 1)])
def test_the_overheads_lag_and_polarity_are_found_to_a_sample(ms, polarity):
    oh = polarity * dsp.fft_convolve(_lowpass(_later(0.5 * SNARE + 0.35 * KICK, ms), 9000), _room_ir())
    oh += 0.02 * RNG.standard_normal(N)
    est = align.estimate(SNARE, oh, SR, "snare")
    assert est["ok"], est
    assert est["lag_ms"] == pytest.approx(ms, abs=0.02)      # 1 sample at 48 kHz = 0.0208 ms
    assert est["polarity"] == polarity


@pytest.mark.parametrize("ms, polarity", [(0.9, 1), (1.4, -1)])
def test_kick_in_and_out_are_found_through_the_snare_bleed(ms, polarity):
    kout = polarity * 0.8 * _lowpass(_later(KICK, ms), 4000) + 0.05 * SNARE + 0.01 * RNG.standard_normal(N)
    est = align.estimate(KICK + 0.1 * SNARE, kout, SR, "kick")
    assert est["ok"], est
    assert est["lag_ms"] == pytest.approx(ms, abs=0.02)
    assert est["polarity"] == polarity


def test_what_is_not_the_same_hit_is_refused():
    """The no-op list: nothing that is not clearly the same source moves."""
    noise = 0.3 * RNG.standard_normal(N)
    assert not align.estimate(SNARE, noise, SR, "snare")["ok"]
    tail = np.exp(-np.arange(int(1.2 * SR)) / SR * 5) * RNG.standard_normal(int(1.2 * SR))
    amb = dsp.fft_convolve(RNG.standard_normal(N), tail)
    assert not align.estimate(SNARE, amb / np.max(np.abs(amb)), SR, "snare")["ok"]
    same = align.estimate(SNARE, 0.7 * SNARE, SR, "snare")
    assert same["ok"] and abs(same["lag_ms"]) < align.ALREADY_ALIGNED_MS and same["polarity"] == 1
    # too few hits to judge: refused, with the reason
    sparse = np.zeros(N)
    sparse[: SR] = SNARE[int(0.5 * SR): int(1.5 * SR)]
    est = align.estimate(sparse, sparse, SR, "snare")
    assert not est["ok"] and "hits" in est["reason"]


def test_the_advance_is_fractional_exact_and_never_wraps():
    x = np.zeros((SR, 2))
    x[1000, :] = 1.0
    x[-1, :] = 0.5          # the last sample must not come round to the start
    y = align.advance(x, SR, 10 * 1e3 / SR)   # exactly 10 samples
    assert y.shape == x.shape
    assert np.argmax(y[:, 0]) == 990 and y[990, 0] == pytest.approx(1.0, abs=1e-9)
    assert abs(y[-11, 0] - 0.5) < 1e-9 and np.max(np.abs(y[-5:, 0])) < 1e-9
    # a fractional move keeps a band-limited signal's shape: there and back again
    # (faded in and out: a hard edge is not band-limited, its sinc tail would
    # leak; real tracks start and end in near silence)
    s = _lowpass(RNG.standard_normal(SR), 8000) * np.hanning(SR)
    back = align.advance(align.advance(s, SR, -0.3712), SR, 0.3712)
    err = back - s       # only the Nyquist bin cannot turn its phase: far below hearing
    assert 10 * np.log10(np.mean(err ** 2) / np.mean(s ** 2)) < -70.0
    assert align.advance(x, SR, 0.0) is x


@pytest.mark.parametrize("bad", [{"law": "gccphat_v9", "advance_ms": 1.0, "polarity": 1},
                                 {"law": "gccphat_v1", "advance_ms": 11.0, "polarity": 1},
                                 {"law": "gccphat_v1", "advance_ms": -1.0, "polarity": 1},
                                 {"law": "gccphat_v1", "advance_ms": float("nan"), "polarity": 1},
                                 {"law": "gccphat_v1", "advance_ms": 1.0, "polarity": 2},
                                 {"law": "gccphat_v1", "advance_ms": True, "polarity": 1}])
def test_a_wrong_align_block_is_refused(bad):
    with pytest.raises(ValueError):
        align.apply(np.zeros((100, 2)), SR, bad)


# --------------------------------------------------------------- a whole kit

KICK_OUT_MS, SNARE_BOTTOM_MS, OH_MS = 1.2, 0.4, 3.0


def _kit():
    """Kick in/out (out late, reversed), snare top/bottom (bottom reversed),
    overheads 3 ms late, a room mic, bass and a vocal."""
    t = np.arange(N) / SR
    oh = dsp.fft_convolve(_lowpass(_later(0.5 * SNARE, OH_MS) + _later(0.35 * KICK, OH_MS + 1.0), 12000),
                          _room_ir()) + 0.003 * RNG.standard_normal(N)
    room = dsp.fft_convolve(_later(0.3 * SNARE + 0.3 * KICK, 14.0),
                            np.exp(-np.arange(int(0.6 * SR)) / SR * 8) * np.random.default_rng(5).standard_normal(int(0.6 * SR)) * 0.05)
    tracks = {
        "Kick In.wav": 0.6 * KICK + 0.04 * SNARE,
        "Kick Out.wav": -0.45 * _lowpass(_later(KICK, KICK_OUT_MS), 5000) + 0.03 * SNARE,
        "Snare Top.wav": 0.5 * SNARE + 0.05 * KICK,
        "Snare Bottom.wav": -0.3 * _lowpass(_later(SNARE, SNARE_BOTTOM_MS), 10000) + 0.02 * KICK,
        "Overheads.wav": 0.6 * oh,
        "Room.wav": room,
        "Bass.wav": 0.1 * np.sin(2 * np.pi * 55 * t),
        "Lead Vocal.wav": 0.12 * ((t % 2.0) < 1.2) * np.sin(2 * np.pi * 440 * t),
    }
    stems = []
    for name, a in tracks.items():
        a = a + 1e-4 * np.random.default_rng(len(name)).standard_normal(N)
        stems.append(analyze.Stem(name=name, path=name, sr=SR, audio=np.stack([a, a], 1).astype(np.float32)))
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    return stems


@pytest.fixture(scope="module")
def kit():
    return _kit()


def _id(name):
    return analyze.track_id(name)


def test_the_kit_comes_out_lined_up(kit):
    roles = {s.name: s.role for s in kit}
    assert roles["Kick Out.wav"] == "kick" and roles["Snare Bottom.wav"] == "snare"
    assert roles["Overheads.wav"] == "overheads" and roles["Room.wav"] == "room"
    p = planner.build_plan(kit, planner.load_preset("rock"))
    planner.validate(p)
    al = {t["id"]: t.get("align") for t in p["tracks"]}
    assert al[_id("Kick In.wav")] is None and al[_id("Snare Top.wav")] is None
    ko = al[_id("Kick Out.wav")]
    assert ko["advance_ms"] == pytest.approx(KICK_OUT_MS, abs=0.03) and ko["polarity"] == -1
    sb = al[_id("Snare Bottom.wav")]
    assert sb["advance_ms"] == pytest.approx(SNARE_BOTTOM_MS, abs=0.03) and sb["polarity"] == -1
    oh = al[_id("Overheads.wav")]
    assert oh["advance_ms"] == pytest.approx(OH_MS - align.CLOSE_LEAD_MS, abs=0.03) and oh["polarity"] == 1
    assert al[_id("Room.wav")] is None and al[_id("Bass.wav")] is None
    rec = p["drum_align"]
    assert rec["law"] == "gccphat_v1" and rec["mode"] == "full"
    room = [d for d in rec["decisions"] if d["track"] == _id("Room.wav")]
    assert room and room[0]["decision"] == "unchanged"


def test_lining_up_brings_the_low_end_back(kit):
    """The reason for all of it: the kick pair's low band sums up instead of cancelling."""
    p = planner.build_plan(kit, planner.load_preset("rock"))
    by = {s.name: s.audio.astype(np.float64) for s in kit}
    ko = next(t["align"] for t in p["tracks"] if t["id"] == _id("Kick Out.wav"))
    lo = lambda x: float(np.mean(align.bandpass(x[:, 0], SR, 40, 250) ** 2))
    raw = lo(by["Kick In.wav"] + by["Kick Out.wav"])
    fixed = lo(by["Kick In.wav"] + align.apply(by["Kick Out.wav"], SR, ko))
    assert 10 * np.log10(fixed / raw) > 3.0


def test_the_render_applies_it_first_and_says_so(kit, tmp_path):
    p = planner.build_plan(kit, planner.load_preset("rock"))
    out, rep = render.render(kit, p)
    ko = next(t for t in rep["tracks"] if t["id"] == _id("Kick Out.wav"))
    assert any("aligned" in c["step"] and "polarity flipped" in c["step"] for c in ko["chain"])
    assert ko["chain"][0]["step"].startswith("aligned")
    assert {m["track"] for m in rep["drum_align"]["moved"]} == {
        _id("Kick Out.wav"), _id("Snare Bottom.wav"), _id("Overheads.wav")}
    path = tmp_path / "report.md"
    render.write_report(str(path), p, rep, kit)
    text = path.read_text()
    assert "| drum mics lined up (time, polarity) | " in text and "polarity flipped" in text
    # the same plan without the moves is a different mix
    off = copy.deepcopy(p)
    for t in off["tracks"]:
        t.pop("align", None)
    off.pop("drum_align")
    out0, rep0 = render.render(kit, off)
    assert "drum_align" not in rep0
    assert not np.allclose(out, out0)


def test_a_wrong_align_in_a_plan_is_refused_by_the_render(kit):
    p = planner.build_plan(kit, planner.load_preset("rock"))
    t = next(t for t in p["tracks"] if t.get("align"))
    t["align"] = {**t["align"], "law": "gccphat_v9"}
    with pytest.raises(render.RenderError):
        render.render(kit, p)


def test_jazz_and_acoustic_keep_the_time_and_fix_only_polarity(kit):
    for style in ("jazz_natural", "acoustic"):
        p = planner.build_plan(kit, planner.load_preset(style))
        planner.validate(p)
        moved = {t["id"]: t["align"] for t in p["tracks"] if t.get("align")}
        assert p["drum_align"]["mode"] == "polarity"
        assert all(m["advance_ms"] == 0.0 for m in moved.values())
        assert moved[_id("Kick Out.wav")]["polarity"] == -1
        assert moved[_id("Snare Bottom.wav")]["polarity"] == -1


def test_a_declared_capture_group_is_never_touched(kit):
    group = [{"id": "kick pair", "members": [_id("Kick In.wav"), _id("Kick Out.wav")]}]
    p = planner.build_plan(kit, planner.load_preset("rock"), capture_groups=group)
    planner.validate(p)
    al = {t["id"]: t.get("align") for t in p["tracks"]}
    assert al[_id("Kick In.wav")] is None and al[_id("Kick Out.wav")] is None
    assert al[_id("Snare Bottom.wav")] is not None      # the rest of the kit still lines up


def test_a_style_without_it_and_a_session_without_a_kit_write_nothing(kit):
    preset = planner.load_preset("rock")
    preset["techniques"] = {k: v for k, v in preset["techniques"].items() if k != "drum_align"}
    p = planner.build_plan(kit, preset)
    assert "drum_align" not in p and not any(t.get("align") for t in p["tracks"])
    one = [s for s in kit if s.name in ("Kick In.wav", "Bass.wav", "Lead Vocal.wav")]
    p1 = planner.build_plan(one, planner.load_preset("rock"))
    assert "drum_align" not in p1 and not any(t.get("align") for t in p1["tracks"])


def test_every_style_names_its_mode():
    import glob, json, os
    root = os.path.join(os.path.dirname(planner.default_schema_path()), "..", "presets")
    seen = {}
    for path in sorted(glob.glob(os.path.join(root, "*.json"))):
        seen[os.path.basename(path)] = json.load(open(path))["techniques"]["drum_align"]
    assert len(seen) == 14
    assert {k for k, v in seen.items() if v == "polarity"} == {"jazz_natural.json", "acoustic.json"}
    assert set(seen.values()) == {"full", "polarity"}


def _tom(i):
    t = np.arange(int(0.5 * SR)) / SR
    return np.sin(2 * np.pi * (95 + 40 * np.exp(-t * 30)) * t) * np.exp(-t * 6)


@pytest.fixture(scope="module")
def kit_with_tom():
    """The kit plus a tom mic that is reversed against the overheads."""
    stems = _kit()
    tom = _hit_train(_tom, np.arange(0.55, DUR - 1, 2.0), N)
    oh = next(s for s in stems if s.name == "Overheads.wav")
    oh_new = oh.audio.astype(np.float64) + 0.3 * np.stack([_later(tom, OH_MS + 0.5)] * 2, 1)
    oh.audio = oh_new.astype(np.float32)
    a = -0.5 * tom + 0.03 * SNARE + 1e-4 * np.random.default_rng(9).standard_normal(N)
    stems.append(analyze.Stem(name="Tom 1.wav", path="Tom 1.wav", sr=SR, audio=np.stack([a, a], 1).astype(np.float32)))
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    return stems


def test_a_reversed_tom_is_flipped_but_never_moved(kit_with_tom):
    assert {s.name: s.role for s in kit_with_tom}["Tom 1.wav"] == "toms"
    p = planner.build_plan(kit_with_tom, planner.load_preset("rock"))
    planner.validate(p)
    tom = next(t for t in p["tracks"] if t["id"] == _id("Tom 1.wav"))
    assert tom["align"] == {"law": "gccphat_v1", "advance_ms": 0.0, "polarity": -1}
    row = next(d for d in p["drum_align"]["decisions"] if d["track"] == _id("Tom 1.wav"))
    assert row["check"] == "polarity only" and row["decision"].startswith("flipped")
    oh = next(t for t in p["tracks"] if t["id"] == _id("Overheads.wav"))
    assert oh["align"]["advance_ms"] == pytest.approx(OH_MS - align.CLOSE_LEAD_MS, abs=0.03)


def test_a_tom_that_agrees_with_the_overheads_is_left_alone(kit):
    """No tom in the overheads at all (pure bleed-free tom): not clear, not flipped."""
    tom = _hit_train(_tom, np.arange(0.55, DUR - 1, 2.0), N)
    stems = list(kit) + [analyze.Stem(name="Tom 1.wav", path="Tom 1.wav", sr=SR,
                                      audio=np.stack([tom, tom], 1).astype(np.float32))]
    analyze.detect_roles(stems)
    analyze.measure(stems[-1])
    p = planner.build_plan(stems, planner.load_preset("rock"))
    tom_t = next(t for t in p["tracks"] if t["id"] == _id("Tom 1.wav"))
    assert "align" not in tom_t


# ------------------------------------------------- each gate, on its own
# (each case below gets through every other gate, so it is this one that stops it)


def test_polarity_needs_both_the_sign_and_the_low_band():
    """Low band in phase, everything above 1 kHz reversed: the correlation's
    sign says "reversed" (most of the band is above 1 kHz), the low band says
    "in phase" -- the two disagree, so nothing is flipped."""
    lo = align.bandpass(SNARE, SR, 20, 1000)
    mem = _later(lo - (SNARE - lo), 0.8)
    est = align.estimate(SNARE, mem, SR, "snare")
    assert est["ok"] and est["sign"] == -1 and est["flip_gain_db"] < 0
    assert est["polarity"] == 0


def test_two_equal_peaks_are_not_a_lag():
    """The same hit arriving twice (1 ms and 6 ms): every group of hits agrees,
    but no single lag stands out -- the peak gate refuses it."""
    mem = 0.5 * _later(SNARE, 1.0) + 0.5 * _later(SNARE, 6.0)
    est = align.estimate(SNARE, mem, SR, "snare")
    assert not est["ok"] and "does not stand out" in est["reason"]
    assert est["spread_ms"] <= align.AGREE_MS


def test_hits_that_disagree_about_the_lag_are_refused():
    """Every other pair of hits 0.3 ms later (an edited or moving mic): one
    clear peak overall, but the groups of hits disagree -- refused."""
    times = np.arange(0.8, DUR - 1, 1.0)
    mem = np.zeros(N)
    for i, tm in enumerate(times):
        h = _snare(i) * (0.75 + 0.25 * ((i * 7) % 5) / 4)
        d = int(round((1.0 if i % 4 in (0, 1) else 1.3) * 1e-3 * SR))
        s = int(tm * SR) + d
        e = min(N, s + h.size)
        mem[s:e] += h[: e - s]
    est = align.estimate(SNARE, mem, SR, "snare")
    assert not est["ok"] and "disagree" in est["reason"]
    assert est["psr"] >= align.PSR_MIN


def test_a_member_silent_where_the_reference_hits_is_refused():
    """Nothing to correlate (0/0 must not read as an infinitely clear peak)."""
    est = align.estimate(SNARE, np.zeros(N), SR, "snare")
    assert not est["ok"]
    quiet = np.zeros(N)
    quiet[: int(0.5 * SR)] = SNARE[int(0.8 * SR): int(1.3 * SR)]   # one hit, then silence
    assert not align.estimate(SNARE, quiet, SR, "snare")["ok"]
    # a faint exact copy is still a clear, agreeing peak to GCC-PHAT (it
    # whitens the level away), so only the level gate can refuse it: 70 dB
    # down is refused as silent, 50 dB down is still measured
    faint = align.estimate(SNARE, 10 ** (-70 / 20) * _later(SNARE, 1.0), SR, "snare")
    assert not faint["ok"] and "silent" in faint["reason"]
    low = align.estimate(SNARE, 10 ** (-50 / 20) * _later(SNARE, 1.0), SR, "snare")
    assert low["ok"] and abs(low["lag_ms"] - 1.0) < 0.03



# ------------------------------------------------ review round 1 (R120)


def _stems_of(tracks):
    stems = []
    for name, a in tracks.items():
        a = a + 1e-4 * np.random.default_rng(len(name)).standard_normal(N)
        stems.append(analyze.Stem(name=name, path=name, sr=SR, audio=np.stack([a, a], 1).astype(np.float32)))
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    return stems


def test_three_mics_on_one_drum_never_need_more_than_the_window():
    """Kick in (loudest), a sub 2 ms earlier, kick out 9 ms later: 11 ms from
    the earliest to the latest.  The plan stays valid (every move <= 10 ms);
    the sub, which would need a delay, keeps its time."""
    stems = _stems_of({
        "Kick In.wav": 0.6 * KICK + 0.04 * SNARE,
        "Kick Sub.wav": 0.35 * _lowpass(_later(KICK, -2.0), 300) + 0.02 * SNARE,
        "Kick Out.wav": 0.4 * _lowpass(_later(KICK, 9.0), 5000) + 0.03 * SNARE,
        "Snare Top.wav": 0.5 * SNARE + 0.05 * KICK,
    })
    assert [s.role for s in stems].count("kick") == 3
    p = planner.build_plan(stems, planner.load_preset("rock"))
    planner.validate(p)
    al = {t["id"]: t.get("align") for t in p["tracks"]}
    assert all(a["advance_ms"] <= 10.0 for a in al.values() if a)
    assert al[_id("Kick Out.wav")]["advance_ms"] == pytest.approx(align.MAX_LAG_MS - 0.05, abs=0.05)
    assert al[_id("Kick In.wav")]["advance_ms"] == pytest.approx(align.MAX_LAG_MS - 0.05 - 9.0, abs=0.05)
    assert al[_id("Kick Sub.wav")] is None
    row = next(d for d in p["drum_align"]["decisions"] if d["track"] == _id("Kick Sub.wav"))
    assert row["decision"] == "time kept" and "would need a delay" in row["reason"]
    render.render(stems, p)          # and it renders


def _reversed_oh_kit(snare_bottom_clear=True, oh_ms=OH_MS):
    oh = -dsp.fft_convolve(_lowpass(_later(0.5 * SNARE, oh_ms) + _later(0.35 * KICK, oh_ms + 1.0), 12000), _room_ir())
    bottom = (-0.3 * _lowpass(_later(SNARE, SNARE_BOTTOM_MS), 10000) + 0.02 * KICK if snare_bottom_clear
              else 0.3 * np.random.default_rng(11).standard_normal(N) * np.abs(SNARE) ** 0.0 * 0.02)
    return _stems_of({
        "Kick In.wav": 0.6 * KICK + 0.04 * SNARE,
        "Snare Top.wav": 0.5 * SNARE + 0.05 * KICK,
        "Snare Bottom.wav": bottom,
        "Overheads.wav": 0.6 * oh,
    })


def test_a_reversed_drum_flips_every_mic_on_it():
    """The overheads say the snare is reversed; the snare bottom could not be
    lined up (nothing clear to correlate) -- it is flipped with the top all
    the same, or the two snare mics would end up opposite."""
    stems = _reversed_oh_kit(snare_bottom_clear=False)
    p = planner.build_plan(stems, planner.load_preset("rock"))
    planner.validate(p)
    al = {t["id"]: t.get("align") for t in p["tracks"]}
    bottom_row = next(d for d in p["drum_align"]["decisions"] if d["track"] == _id("Snare Bottom.wav"))
    assert al[_id("Snare Top.wav")]["polarity"] == -1
    assert al[_id("Snare Bottom.wav")] == {"law": "gccphat_v1", "advance_ms": 0.0, "polarity": -1}
    # ... and its receipt says so (it was not lined up, but it was changed)
    assert bottom_row["decision"] == "flipped with its drum (time as recorded)" and bottom_row["note"]
    _, rep = render.render(stems, p)
    assert _id("Snare Bottom.wav") in {m["track"] for m in rep["drum_align"]["moved"]}
    assert _id("Snare Bottom.wav") not in {m["track"] for m in rep["drum_align"]["left_alone"]}


def test_polarity_mode_fixes_polarity_whatever_the_time_says():
    """Jazz: overheads that arrive before the close mic (a shifted session)
    and are reversed -- the time is kept, the snare still flips."""
    stems = _reversed_oh_kit(oh_ms=-1.5)
    p = planner.build_plan(stems, planner.load_preset("jazz_natural"))
    oh_row = next(d for d in p["drum_align"]["decisions"] if d["role"] == "overheads" and d.get("against") == _id("Snare Top.wav"))
    assert oh_row["decision"].startswith("time kept") and oh_row.get("flipped") == "snare"
    al = {t["id"]: t.get("align") for t in p["tracks"]}
    assert al[_id("Snare Top.wav")]["polarity"] == -1 and al[_id("Snare Top.wav")]["advance_ms"] == 0.0
    full = planner.build_plan(stems, planner.load_preset("rock"))    # the same in full mode
    assert {t["id"]: t.get("align") for t in full["tracks"]}[_id("Snare Top.wav")]["polarity"] == -1


def test_an_unknown_mode_is_a_plan_error(kit):
    preset = planner.load_preset("rock")
    preset["techniques"]["drum_align"] = "yes"
    with pytest.raises(planner.PlanError):
        planner.build_plan(kit, preset)


def test_an_align_block_without_its_numbers_is_refused():
    for bad in ({"law": "gccphat_v1"}, {"law": "gccphat_v1", "advance_ms": 1.0},
                {"law": "gccphat_v1", "polarity": -1}):
        with pytest.raises(ValueError):
            align.apply(np.zeros((100, 2)), SR, bad)
