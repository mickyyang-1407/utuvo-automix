"""R114: a bigger, less squashed mix with a wetter vocal.

Reported 2026-11-04: "整個 mix 都會有一點小夾子器或是 over compress" (the fast
limiter flutters at audio rate), "歌的畫面要很大" (a wider top end), "vocal
都會有一點偏幹，你的 reverb 要用好一點" (a reverb that sounds like a space).
"""
import numpy as np
import pytest

from automix import dsp

SR = 48000


def _hf_share_db(y):
    f = np.fft.rfftfreq(len(y), 1 / SR)
    s = np.abs(np.fft.rfft(y[:, 0])) ** 2
    return 10 * np.log10(s[f > 1000].sum() / s.sum())


def _loud_bass_with_kicks():
    n = SR * 6
    t = np.arange(n) / SR
    x = .6 * np.sin(2 * np.pi * 80 * t)
    k = np.arange(int(.12 * SR)) / SR
    hit = np.sin(2 * np.pi * 55 * k) * np.exp(-k / .05)
    for s in range(0, n - hit.size, SR // 2):
        x[s:s + hit.size] += .9 * hit
    return np.stack([x, x], 1) * 2.5


def test_the_release_limiter_adds_far_less_distortion():
    x = _loud_bass_with_kicks()
    fast = dsp.true_peak_limit(x, SR, -1.0, law="tp_fast_v1")
    slow = dsp.true_peak_limit(x, SR, -1.0, law="tp_release_v1")
    assert _hf_share_db(slow) < _hf_share_db(fast) - 6.0
    assert dsp.true_peak_db(slow, SR) <= -1.0


def test_the_release_limiter_holds_the_ceiling_on_isolated_peaks():
    x = np.zeros((SR * 2, 2))
    x[SR:SR + 50] = 3.0
    y = dsp.true_peak_limit(x, SR, -1.0, law="tp_release_v1")
    assert dsp.true_peak_db(y, SR) <= -1.0


def test_plate_v2_sounds_like_a_space():
    ir = dsp.make_ir_plate_v2(SR, decay_s=2.0, predelay_ms=0.0, high_cut_hz=12000.0)
    assert np.sum(ir ** 2) == pytest.approx(1.0, rel=1e-6)
    f = np.fft.rfftfreq(ir.shape[0], 1 / SR)
    s = np.abs(np.fft.rfft(ir[:, 0])) ** 2
    assert s[f < 120].sum() / s.sum() < .01                       # no low end in the return
    late = ir[int(1.0 * SR):]
    early = ir[: int(.3 * SR)]
    def band(x, lo, hi):
        F = np.fft.rfftfreq(x.shape[0], 1 / SR); S = np.abs(np.fft.rfft(x[:, 0])) ** 2
        return S[(F >= lo) & (F < hi)].sum()
    # highs die faster than the mids: their share falls from early to late
    assert band(late, 5000, 10000) / band(late, 600, 3000) < .5 * band(early, 5000, 10000) / band(early, 600, 3000)
    # early reflections: distinct peaks before 50 ms stand out of the build-up
    first = np.abs(ir[: int(.05 * SR), :]).max()
    assert first > 4 * np.abs(ir[int(.002 * SR): int(.006 * SR)]).max()
    # the two sides are not the same reverb
    assert abs(np.corrcoef(ir[:, 0], ir[:, 1])[0, 1]) < .2


def test_side_air_widens_the_top_and_leaves_mono_alone():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((SR * 2, 2)) * .1
    x[:, 1] = .7 * x[:, 0] + .3 * x[:, 1]
    y = dsp.side_shelf(x, SR, 2.0)
    assert np.allclose(y[:, 0] + y[:, 1], x[:, 0] + x[:, 1], atol=1e-9)   # mono fold-down unchanged
    def side_hf(z):
        s = .5 * (z[:, 0] - z[:, 1]); F = np.fft.rfftfreq(len(s), 1 / SR); S = np.abs(np.fft.rfft(s)) ** 2
        return 10 * np.log10(S[F > 9000].sum())
    assert side_hf(y) - side_hf(x) == pytest.approx(2.0, abs=.4)


def test_the_release_limiter_holds_the_ceiling_across_its_block_edges():
    # blocks are 8 s: put peaks right at the edges and in between
    x = .05 * np.random.default_rng(3).standard_normal((SR * 20, 2))
    for at in (8 * SR - 30, 8 * SR, 16 * SR + 7, 12 * SR):
        x[at:at + 60] = 2.5
    y = dsp.true_peak_limit(x, SR, -1.0, law="tp_release_v1")
    assert dsp.true_peak_db(y, SR) <= -1.0
    # the release carries on across the edge: a peak that ends 3 ms before
    # 8 s is not seen by the next block's own samples, yet 10 ms after the
    # edge the gain is still well down (30 dB/s from ~9 dB)
    x2 = .05 * np.random.default_rng(4).standard_normal((SR * 12, 2))
    x2[8 * SR - 200: 8 * SR - 140] = 2.5
    y2 = dsp.true_peak_limit(x2, SR, -1.0, law="tp_release_v1")
    at = 8 * SR + int(.01 * SR)
    seg = slice(at, at + 48)
    gain = np.sqrt(np.mean(y2[seg] ** 2) / np.mean(x2[seg] ** 2))
    assert gain < .5, gain


def test_drums_and_bass_both_land_on_their_targets():
    from automix import analyze, plan as planner, render
    t = np.arange(SR * 12) / SR
    rng = np.random.default_rng(9)
    kick = np.zeros_like(t)
    for s in range(0, t.size - 4800, SR // 2):
        kick[s:s + 4800] += np.sin(2 * np.pi * 60 * t[:4800]) * np.exp(-t[:4800] / .06)
    stems = []
    def add(name, a):
        stems.append(analyze.Stem(name=name, path=name, sr=SR, audio=np.stack([a, a], 1).astype(np.float32)))
    add("Kick.wav", .6 * kick)
    add("Bass.wav", .12 * np.sin(2 * np.pi * 55 * t))
    add("Keys.wav", .08 * np.sin(2 * np.pi * 330 * t) + .02 * rng.standard_normal(t.size))
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    _, rep = render.render(stems, p)
    for name, pb in rep["part_balance"].items():
        assert pb["result_db"] == pytest.approx(pb["target_db"], abs=.5), (name, pb)


def test_the_squash_guard_gives_back_loudness_instead_of_limiting_hard():
    """R115: a dense song limited hard on its loudest moments comes out a
    little quieter instead; plans without the guard render as before."""
    import copy
    from automix import analyze, plan as planner, render
    t = np.arange(SR * 10) / SR
    rng = np.random.default_rng(11)
    hits = np.zeros_like(t)
    for s in range(0, t.size - 2400, SR // 4):
        hits[s:s + 2400] = rng.standard_normal(2400) * np.exp(-np.arange(2400) / 300)
    stems = []
    def add(name, a):
        stems.append(analyze.Stem(name=name, path=name, sr=SR, audio=np.stack([a, a], 1).astype(np.float32)))
    add("Drums.wav", .9 * hits)
    add("Bass.wav", .3 * np.sin(2 * np.pi * 55 * t))
    add("Synth.wav", .2 * np.sign(np.sin(2 * np.pi * 220 * t)))
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    p = planner.build_plan(stems, planner.load_preset("trap"))
    p["buses"]["master"]["target_lufs"] = -7.0           # push it into the limiter
    guarded_mix, guarded = render.render(stems, p)
    q = copy.deepcopy(p)
    for k in ("max_gr_p95_db", "max_backoff_db"):
        del q["buses"]["master"]["limiter"][k]
    open_mix, unguarded = render.render(stems, q)
    assert unguarded["limiter"]["gain_reduction_p95_db"] < -3.0          # the case the guard is for
    assert 0 < guarded["limiter"]["backed_off_db"] <= 1.5
    assert guarded["limiter"]["gain_reduction_p95_db"] > unguarded["limiter"]["gain_reduction_p95_db"]
    assert guarded["lufs"] < unguarded["lufs"]
    assert any("quieter than the target" in w for w in guarded["warnings"])
    assert "backed_off_db" not in unguarded["limiter"]
    assert dsp.true_peak_db(guarded_mix, SR) <= -1.0 + 1e-6 or guarded["true_peak_db"] <= p["buses"]["master"]["limiter"]["ceiling_db"]


def test_a_band_that_starts_far_too_loud_still_reaches_its_targets():
    """R116: with a quiet rest of the band the drums and bass need more than
    6 dB of cut; the plan now allows 12 (6 dB plans replay as they were)."""
    import copy
    from automix import analyze, plan as planner, render
    t = np.arange(SR * 12) / SR
    rng = np.random.default_rng(9)
    kick = np.zeros_like(t)
    for s in range(0, t.size - 4800, SR // 2):
        kick[s:s + 4800] += np.sin(2 * np.pi * 60 * t[:4800]) * np.exp(-t[:4800] / .06)
    stems = []
    def add(name, a):
        stems.append(analyze.Stem(name=name, path=name, sr=SR, audio=np.stack([a, a], 1).astype(np.float32)))
    add("Kick.wav", .6 * kick)
    add("Bass.wav", .12 * np.sin(2 * np.pi * 55 * t))
    add("Keys.wav", .02 * np.sin(2 * np.pi * 330 * t) + .005 * rng.standard_normal(t.size))
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    for name in ("drum", "bass"):
        assert p["buses"][name]["balance"]["max_cut_db"] == planner.PART_BALANCE_MAX_CUT_DB == 12.0
        assert p["buses"][name]["balance"]["max_boost_db"] == 6.0
    planner.validate(p)
    _, rep = render.render(stems, p)
    moved = {n: pb["applied_db"] for n, pb in rep["part_balance"].items()}
    assert min(moved.values()) < -6.0, moved          # the case really needs more than 6
    for name, pb in rep["part_balance"].items():
        assert pb["result_db"] == pytest.approx(pb["target_db"], abs=.5), (name, pb)
    # the same plan with the old 6 dB limit stops short (what 0.115.0 did)
    old = copy.deepcopy(p)
    for name in ("drum", "bass"):
        old["buses"][name]["balance"]["max_cut_db"] = 6.0
    _, rep_old = render.render(stems, old)
    assert any(pb["result_db"] - pb["target_db"] > .5 for pb in rep_old["part_balance"].values()), rep_old["part_balance"]
    assert all(pb["applied_db"] >= -6.0 - 1e-9 for pb in rep_old["part_balance"].values())



def test_the_balance_is_measured_through_the_master_eq():
    """R116: Cinematic's +2 dB low shelf at 120 Hz left kick and bass
    0.5-0.7 dB above target in every song when the bare buses were measured.
    New plans measure through the master EQ; plans without meter do not."""
    import copy
    from automix import analyze, dsp, plan as planner, render
    t = np.arange(SR * 12) / SR
    rng = np.random.default_rng(9)
    kick = np.zeros_like(t)
    for s in range(0, t.size - 4800, SR // 2):
        kick[s:s + 4800] += np.sin(2 * np.pi * 60 * t[:4800]) * np.exp(-t[:4800] / .06)
    env = (np.sin(2 * np.pi * .25 * t) > 0).astype(float)
    stems = []
    def add(name, a):
        stems.append(analyze.Stem(name=name, path=name, sr=SR, audio=np.stack([a, a], 1).astype(np.float32)))
    add("Kick.wav", .6 * kick)
    add("Bass.wav", .12 * np.sin(2 * np.pi * 55 * t))
    add("Keys.wav", .08 * np.sin(2 * np.pi * 330 * t) + .02 * rng.standard_normal(t.size))
    add("Lead Vocal.wav", .1 * env * np.sin(2 * np.pi * 440 * t))
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    p = planner.build_plan(stems, planner.load_preset("cinematic"))
    assert any(b.get("type") == "low_shelf" and b.get("gain_db", 0) > 1 for b in p["buses"]["master"]["eq"])
    for name in ("vocal", "drum", "bass"):
        assert p["buses"][name]["balance"]["meter"] == "master_eq_v1"
    planner.validate(p)

    seen = {}
    real = render._vocal_balance
    def spy(buses, spec, sr, warnings, meter=None):
        out = real(buses, spec, sr, warnings, meter=meter)
        bands = render._eq_bands(p["buses"]["master"]["eq"])
        eq = {k: dsp.apply_eq(v, sr, bands) for k, v in buses.items() if k != "fx"}
        for n in ("drum", "bass"):
            rest = sum(v for k, v in eq.items() if k not in (n, "vocal"))
            seen[n] = dsp.vocal_to_rest_db(eq[n], rest, sr)[0]
        seen["vocal"] = dsp.vocal_to_rest_db(eq["vocal"], sum(v for k, v in eq.items() if k != "vocal"), sr)[0]
        return out
    render._vocal_balance = spy
    try:
        _, rep = render.render(stems, p)
        old = copy.deepcopy(p)
        for name in ("vocal", "drum", "bass"):
            old["buses"][name]["balance"].pop("meter")
        _, rep_old = render.render(stems, old)
        seen_old = dict(seen)
        _, _ = render.render(stems, p)
    finally:
        render._vocal_balance = real
    # what the listener hears is on target with the meter ...
    for n in ("drum", "bass", "vocal"):
        target = p["buses"][n]["balance"]["target_db"]
        assert seen[n] == pytest.approx(target, abs=.25), (n, seen[n], target)
    assert rep["part_balance"]["drum"]["meter"] == "master_eq_v1" and rep["vocal_balance"]["meter"] == "master_eq_v1"
    # ... and was not without it (the low shelf pushed kick and bass up)
    assert seen_old["drum"] - p["buses"]["drum"]["balance"]["target_db"] > .25 or \
        seen_old["bass"] - p["buses"]["bass"]["balance"]["target_db"] > .25, seen_old
    assert "meter" not in rep_old["part_balance"]["drum"] and "meter" not in rep_old["vocal_balance"]


def test_an_unknown_balance_meter_is_refused():
    from automix import render
    buses = {"drum": np.ones((10, 2)), "bass": np.ones((10, 2))}
    with pytest.raises(render.RenderError):
        render._balance_meter(buses, {"drum": {"balance": {"law": "part_var_v1", "target_db": -5, "meter": "x"}}}, SR)


def _band(keys_level=.08, with_keys=True, seconds=12):
    from automix import analyze
    t = np.arange(SR * seconds) / SR
    rng = np.random.default_rng(9)
    kick = np.zeros_like(t)
    for s in range(0, t.size - 4800, SR // 2):
        kick[s:s + 4800] += np.sin(2 * np.pi * 60 * t[:4800]) * np.exp(-t[:4800] / .06)
    stems = []
    def add(name, a):
        stems.append(analyze.Stem(name=name, path=name, sr=SR, audio=np.stack([a, a], 1).astype(np.float32)))
    add("Kick.wav", .6 * kick)
    add("Bass.wav", .12 * np.sin(2 * np.pi * 55 * t))
    if with_keys:
        add("Keys.wav", keys_level * np.sin(2 * np.pi * 330 * t) + .25 * keys_level * rng.standard_normal(t.size))
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    return stems


def test_when_the_cut_limit_binds_the_metered_copies_follow_the_buses():
    """The cap correction scales the bus and its metered copy together: the
    reported result is what the master-EQ'd buses really measure."""
    import copy
    from automix import plan as planner, render, dsp
    stems = _band(keys_level=.02)
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    for name in ("drum", "bass"):
        p["buses"][name]["balance"]["max_cut_db"] = 2.0
    seen = {}
    real = render._vocal_balance
    def spy(buses, spec, sr, warnings, meter=None):
        bands = render._eq_bands(p["buses"]["master"]["eq"])
        eq = {k: dsp.apply_eq(v, sr, bands) for k, v in buses.items() if k != "fx"}
        for n in ("drum", "bass"):
            seen[n] = dsp.vocal_to_rest_db(eq[n], sum(v for k, v in eq.items() if k not in (n, "vocal")), sr)[0]
            seen[n + "_meter"] = dsp.vocal_to_rest_db(meter[n], sum(v for k, v in meter.items() if k not in (n, "vocal")), sr)[0]
        return real(buses, spec, sr, warnings, meter=meter)
    render._vocal_balance = spy
    try:
        _, rep = render.render(stems, p)
    finally:
        render._vocal_balance = real
    for n in ("drum", "bass"):
        pb = rep["part_balance"][n]
        assert pb["applied_db"] >= -2.0 - 1e-9
        assert seen[n] == pytest.approx(seen[n + "_meter"], abs=1e-6)
        assert pb["result_db"] == pytest.approx(seen[n], abs=.01)
    assert any(pb["applied_db"] == pytest.approx(-2.0) for pb in rep["part_balance"].values())


def test_a_band_of_only_drums_and_bass_is_left_alone_in_new_plans():
    import copy
    from automix import plan as planner, render
    stems = _band(with_keys=False)
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    _, rep = render.render(stems, p)
    assert rep.get("part_balance") in (None, {})
    assert any("each is the other's only reference" in w for w in rep["warnings"])
    # an old plan (no meter) still runs the alternation as before
    old = copy.deepcopy(p)
    for name in ("vocal", "drum", "bass"):
        (old["buses"].get(name) or {}).get("balance", {}).pop("meter", None)
    _, rep_old = render.render(stems, old)
    assert set(rep_old["part_balance"]) == {"drum", "bass"}


def test_a_flat_master_eq_measures_the_bare_buses_and_does_not_claim_a_meter():
    from automix import render
    buses = {"drum": np.ones((10, 2)), "bass": np.ones((10, 2)), "music": np.ones((10, 2))}
    specs = {"drum": {"balance": {"law": "part_var_v1", "target_db": -5, "meter": "master_eq_v1"}},
             "master": {"eq": []}}
    assert render._balance_meter(buses, specs, SR) is None


def test_the_report_says_when_the_balance_was_measured_through_the_master_eq(tmp_path):
    import copy
    from automix import analyze, plan as planner, render
    stems = _band()
    t = np.arange(SR * 12) / SR
    env = (np.sin(2 * np.pi * .25 * t) > 0).astype(float)
    v = analyze.Stem(name="Lead Vocal.wav", path="v", sr=SR,
                     audio=np.stack([.1 * env * np.sin(2 * np.pi * 440 * t)] * 2, 1).astype(np.float32))
    analyze.detect_roles([v]); analyze.measure(v)
    stems.append(v)
    analyze.detect_roles(stems)
    p = planner.build_plan(stems, planner.load_preset("pop_2020s"))
    _, rep = render.render(stems, p)
    render.write_report(str(tmp_path / "new.md"), p, rep, stems)
    new = (tmp_path / "new.md").read_text()
    assert "| vocal against the rest (where it sings, through the master EQ) |" in new
    assert "| drum against the band (where it plays, no vocal, through the master EQ) |" in new
    assert "| bass against the band (where it plays, no vocal, through the master EQ) |" in new
    old = copy.deepcopy(p)
    for name in ("vocal", "drum", "bass"):
        old["buses"][name]["balance"].pop("meter")
    _, rep_old = render.render(stems, old)
    render.write_report(str(tmp_path / "old.md"), old, rep_old, stems)
    text = (tmp_path / "old.md").read_text()
    assert "through the master EQ" not in text
    assert "| vocal against the rest (where it sings, before the master stage) |" in text


def test_a_part_far_under_the_band_does_not_count_as_a_reference():
    """The drums+bass-only guard goes by level: 60 dB under the band is no
    reference, 20 dB under is one."""
    from automix import render
    t = np.arange(SR * 12) / SR
    kick = np.zeros_like(t)
    for s in range(0, t.size - 4800, SR // 2):
        kick[s:s + 4800] += np.sin(2 * np.pi * 60 * t[:4800]) * np.exp(-t[:4800] / .06)
    st = lambda a: np.stack([a, a], 1)
    specs = {n: {"balance": {"law": "part_var_v1", "target_db": -5.0, "max_boost_db": 6.0, "max_cut_db": 12.0,
                             "meter": "master_eq_v1"}} for n in ("drum", "bass")}
    for under, expect_skip in ((60.0, True), (20.0, False)):
        keys = .1 * np.sin(2 * np.pi * 330 * t) * 10 ** (-under / 20)
        buses = {"drum": st(.6 * kick), "bass": st(.12 * np.sin(2 * np.pi * 55 * t)), "music": st(keys)}
        warnings = []
        out = render._balance_band(buses, specs, SR, warnings, meter={k: v.copy() for k, v in buses.items()})
        skipped = out == {} and any("each is the other's only reference" in w for w in warnings)
        assert skipped is expect_skip, (under, out, warnings)
