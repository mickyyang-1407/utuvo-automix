"""Three more mixing moves: kick sidechain on the bass, drum parallel
compression, and a vocal delay locked to the tempo.

Requested 2026-11-02: "據續往下走，找到更多的混音方式放進去".  Each move is a
block in the plan, so every test here asks two things: does the audio do what
the block says, and does a plan without the block stay exactly as it was?
"""
import copy

import numpy as np
import pytest

from automix import analyze, dsp, plan as planner, render

SR = 48000


def _clicks(bpm: float, seconds: float = 30.0, seed: int = 0) -> np.ndarray:
    """A kick on every beat, a snare on 2 and 4, hats on the eighths."""
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    x = np.zeros(n)
    k = np.arange(int(.15 * SR)) / SR
    kick = np.sin(2 * np.pi * 55 * k) * np.exp(-k / .04)
    h = np.arange(int(.03 * SR)) / SR
    hat = .3 * rng.standard_normal(h.size) * np.exp(-h / .008)
    s = np.arange(int(.12 * SR)) / SR
    snare = .6 * rng.standard_normal(s.size) * np.exp(-s / .03)
    beat = 60.0 / bpm
    i = 0
    while (i + 1) * beat < seconds - .3:
        for sub in (0, 1):
            at = int(((i + sub / 2) * beat + rng.normal(0, .004)) * SR)
            x[at:at + hat.size] += hat
            if sub == 0:
                x[at:at + (kick.size if i % 2 == 0 else snare.size)] += kick if i % 2 == 0 else snare
        i += 1
    return np.stack([x, x], 1)


# ---------------------------------------------------------------- tempo


@pytest.mark.parametrize("bpm", [85, 97, 110, 128, 140, 160])
def test_tempo_is_found_on_a_steady_beat(bpm):
    found = dsp.estimate_tempo(_clicks(bpm), SR)
    assert found["halves_agree"] == 2
    assert found["strength"] >= planner.TEMPO_LOCK_MIN_STRENGTH
    # the same pulse within 1%, or exactly twice / half of it (still on grid)
    assert min(abs(found["bpm"] / bpm - r) for r in (.5, 1, 2)) <= .01, found


def test_noise_has_no_tempo_to_lock_to():
    x = .1 * np.random.default_rng(1).standard_normal((SR * 30, 2))
    found = dsp.estimate_tempo(x, SR)
    assert found is None or found["strength"] < planner.TEMPO_LOCK_MIN_STRENGTH or found["halves_agree"] < 2


# ---------------------------------------------------------------- sidechain


def test_duck_dips_on_the_hit_and_not_before_it():
    n = SR * 4
    bass = np.ones((n, 2)) * .1
    key = np.zeros((n, 2))
    hits = [SR, 2 * SR, 3 * SR]
    for at in hits:
        key[at:at + 2400] = .8
    out, gain = dsp.duck(bass, key, SR, 4.0, attack_ms=5.0, release_ms=120.0)
    g_db = 20 * np.log10(gain)
    for at in hits:
        assert g_db[at + 1000] == pytest.approx(-4.0, abs=.05)   # on the hit: full depth
        assert g_db[at - int(.02 * SR)] > -.05                  # 20 ms before: untouched
        assert g_db[at + 2400 + int(.06 * SR)] > -4.0           # releasing afterwards
        assert g_db[at + 2400 + int(.3 * SR)] > -.05            # back by 300 ms
    assert np.allclose(out, bass * gain[:, None])


def test_duck_follows_how_hard_the_key_hits():
    n = SR * 3
    key = np.zeros((n, 2))
    key[SR:SR + 2400] = .8
    key[2 * SR:2 * SR + 2400] = .4          # 6 dB softer
    _, gain = dsp.duck(np.ones((n, 2)), key, SR, 6.0)
    loud, soft = 20 * np.log10(gain[SR + 1000]), 20 * np.log10(gain[2 * SR + 1000])
    assert loud == pytest.approx(-6.0, abs=.05)
    assert -4.0 < soft < -2.0


# ---------------------------------------------------------------- parallel


def test_parallel_keeps_the_level_and_lowers_the_peaks():
    # hits with a room behind them: what parallel compression brings up is
    # the quiet part between the hits, which pure clicks over silence lack
    room = .003 * np.random.default_rng(5).standard_normal((SR * 10, 2))
    drums = _clicks(100, 10) + room
    out = dsp.parallel_compress(drums, SR, .4)
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    assert rms(out) == pytest.approx(rms(drums), rel=1e-6)
    assert analyze._envelope_crest_db(out, SR) < analyze._envelope_crest_db(drums, SR) - 1.0
    assert dsp.parallel_compress(drums, SR, 0.0) is drums


# ---------------------------------------------------------------- delay


def test_delay_repeats_on_time_and_dies_away():
    x = np.zeros((SR * 3, 2))
    x[1000] = 1.0
    wet = dsp.delay_send(x, SR, 250.0, .5, hpf_hz=20.5, lpf_hz=20000.0)
    d = int(SR * .25)
    peaks = [float(np.max(np.abs(wet[1000 + k * d - 50:1000 + k * d + 50, 0]))) for k in (1, 2, 3)]
    assert peaks[1] / peaks[0] == pytest.approx(.5, rel=.05)
    assert peaks[2] / peaks[1] == pytest.approx(.5, rel=.05)
    assert float(np.max(np.abs(wet[:1000 + d - 200]))) < .01 * peaks[0]   # nothing before the first echo


# ---------------------------------------------------------------- the plan


def _session(bpm=110, with_kick=True, with_bass=True, with_vocal=True, seconds=24.0):
    n = int(SR * seconds)
    t = np.arange(n) / SR
    beat = _clicks(bpm, seconds)
    stems = []

    def add(name, audio):
        stems.append(analyze.Stem(name=name, path=name, sr=SR, audio=audio.astype(np.float32)))

    add("Kick.wav" if with_kick else "Drums.wav", beat)
    if with_bass:
        add("Bass.wav", .2 * np.sin(2 * np.pi * 55 * t)[:, None] * np.ones((1, 2)))
    if with_vocal:
        env = (np.sin(2 * np.pi * .25 * t) > 0).astype(float)
        add("Lead Vocal.wav", (.15 * env * np.sin(2 * np.pi * 330 * t))[:, None] * np.ones((1, 2)))
    analyze.detect_roles(stems)
    for st in stems:
        analyze.measure(st)
    return stems


def _plan(stems, style="pop_2020s", intensity=1.0):
    return planner.build_plan(stems, planner.load_preset(style), intensity=intensity)


def test_style_moves_are_written_into_the_plan():
    stems = _session()
    p = _plan(stems)
    duck = p["buses"]["bass"]["duck"]
    # R121: new plans duck only the bass's low band, split at 1.6 x the
    # kick's fundamental, held to 90-130 Hz (the synthetic kick is a 55 Hz
    # thump, found within a 2 Hz bin: 1.6 x 55 = 88 -> 90 Hz)
    assert duck["law"] == "kick_duck_band_v1" and duck["key_ids"] == ["Kick"] and duck["depth_db"] == 2.0
    assert abs(duck["kick_f0_hz"] - 55.0) <= 2.0 and duck["crossover_hz"] == 90.0
    assert duck["key_lowpass_hz"] == 120.0
    # R116: half of R111's 0.35, the published "crushed copy about 15 dB under the kit"
    assert p["buses"]["drum"]["parallel"]["blend"] == .175
    delay = p["delay"]
    assert delay["division"] == "1/4" and delay["tempo"]["bpm"] == pytest.approx(110, rel=.01)
    assert delay["time_ms"] == pytest.approx(60000 / delay["tempo"]["bpm"], abs=.01)
    vocal = next(t for t in p["tracks"] if t["role"] == "lead_vocal")
    assert vocal["sends"]["delay"] == .1
    planner.validate(p)


def test_a_style_without_them_writes_none():
    p = _plan(_session(), "jazz_natural")
    assert "duck" not in p["buses"]["bass"] and "parallel" not in p["buses"]["drum"]
    assert "delay" not in p
    assert all("delay" not in (t.get("sends") or {}) for t in p["tracks"])


def test_intensity_zero_writes_none():
    p = _plan(_session(), intensity=0.0)
    assert "duck" not in p["buses"]["bass"] and "delay" not in p


def test_a_whole_kit_keys_the_duck_on_its_low_end():
    p = _plan(_session(with_kick=False))
    duck = p["buses"]["bass"]["duck"]
    assert duck["key_ids"] == ["Drums"] and duck["key_lowpass_hz"] == 120.0     # R121 (v1: 150)
    assert 90.0 <= duck["crossover_hz"] <= 130.0


def test_no_bass_no_duck_and_no_vocal_no_delay():
    assert "duck" not in _plan(_session(with_bass=False))["buses"]["bass"]
    assert "delay" not in _plan(_session(with_vocal=False))


def test_without_a_clear_tempo_the_delay_is_a_slapback(monkeypatch):
    monkeypatch.setattr(dsp, "estimate_tempo", lambda x, sr: {"bpm": 120.0, "strength": .2, "halves_agree": 1})
    p = _plan(_session())
    assert p["delay"]["division"] == "slap" and p["delay"]["time_ms"] == planner.SLAP_MS
    assert p["delay"]["feedback"] <= planner.SLAP_MAX_FEEDBACK
    assert "tempo" not in p["delay"]
    assert any("no clear tempo" in w for w in p["warnings"])


# ---------------------------------------------------------------- rendering


def test_the_render_does_what_the_plan_says():
    stems = _session()
    p = _plan(stems)
    mix, report = render.render(stems, p)
    moves = report["mix_moves"]
    assert -2.0 <= moves["kick_duck"]["under_hits_db"] <= -1.0
    assert moves["drum_parallel"]["crest_after_db"] < moves["drum_parallel"]["crest_before_db"]
    assert moves["vocal_delay"]["time_ms"] == p["delay"]["time_ms"]
    assert moves["vocal_delay"]["ducking_db"] < -1.0     # the voice sings half the time
    assert np.all(np.isfinite(mix))


def test_each_move_changes_the_audio_and_removing_it_removes_only_it():
    stems = _session()
    p = _plan(stems)
    full, _ = render.render(stems, p)
    for strip in ("duck", "parallel", "delay"):
        q = copy.deepcopy(p)
        if strip == "delay":
            del q["delay"]
            for t in q["tracks"]:
                (t.get("sends") or {}).pop("delay", None)
        else:
            del q["buses"]["bass" if strip == "duck" else "drum"][strip]
        other, report = render.render(stems, q)
        assert not np.allclose(other, full), strip
        assert ("kick_duck" if strip == "duck" else "drum_parallel" if strip == "parallel" else "vocal_delay") \
            not in (report.get("mix_moves") or {})


def test_report_says_what_each_move_did(tmp_path):
    stems = _session()
    p = _plan(stems)
    mix, report = render.render(stems, p)
    path = tmp_path / "report.md"
    render.write_report(str(path), p, report, stems)
    text = path.read_text()
    assert "| bass under the kick (sidechain) |" in text
    assert "| drum parallel compression |" in text
    assert "| vocal delay | 1/4 at" in text


def test_the_delay_return_is_exported_for_a_daw(tmp_path):
    stems = _session()
    p = _plan(stems)
    collect = {}
    mix, report = render.render(stems, p, collect=collect)
    render.write_stem_export(str(tmp_path), p, collect, report, mix)
    assert (tmp_path / "_delay.wav").exists()


# ---------------------------------------------------------------- invariants


def test_the_engine_refuses_moves_in_the_wrong_place():
    p = _plan(_session())
    q = copy.deepcopy(p)
    q["buses"]["drum"]["duck"] = q["buses"]["bass"]["duck"]
    with pytest.raises(planner.PlanError, match="only the bass bus"):
        planner.validate(q)
    q = copy.deepcopy(p)
    q["buses"]["bass"]["duck"]["key_ids"] = ["nobody"]
    with pytest.raises(planner.PlanError, match="key tracks"):
        planner.validate(q)
    q = copy.deepcopy(p)
    del q["delay"]
    with pytest.raises(planner.PlanError, match="no delay block"):
        planner.validate(q)


# ---------------------------------------------------------------- review round 1


def _pulses(periods, amps, seconds=30.0):
    n = int(SR * seconds)
    k = np.arange(int(.1 * SR)) / SR
    hit = np.sin(2 * np.pi * 80 * k) * np.exp(-k / .03)
    x = np.zeros(n)
    for per, a in zip(periods, amps):
        t = 0.0
        while t < seconds - .5:
            at = int(t * SR)
            x[at:at + hit.size] += a * hit
            t += per
    return np.stack([x, x], 1)


def test_two_pulses_at_3_to_4_are_not_a_clear_tempo():
    # 133 against 100 BPM, equally loud: either could be the beat
    found = dsp.estimate_tempo(_pulses((.45, .6), (1, 1)), SR)
    assert found["ambiguous"] is True


def test_an_ambiguous_tempo_is_not_locked(monkeypatch):
    monkeypatch.setattr(dsp, "estimate_tempo",
                        lambda x, sr: {"bpm": 120.0, "strength": .7, "halves_agree": 2, "ambiguous": True})
    p = _plan(_session())
    assert p["delay"]["division"] == "slap" and "tempo" not in p["delay"]


def test_a_key_without_hits_does_not_hold_the_bass_down():
    n = SR * 4
    rumble = .3 * np.random.default_rng(3).standard_normal((n, 2))
    _, gain = dsp.duck(np.ones((n, 2)), rumble, SR, 6.0)
    assert np.all(gain == 1.0)


def test_rumble_under_the_kick_is_not_ducked_between_hits():
    n = SR * 4
    key = .02 * np.random.default_rng(4).standard_normal((n, 2))   # bleed / rumble
    for at in (SR, 2 * SR, 3 * SR):
        key[at:at + 2400] += .8
    _, gain = dsp.duck(np.ones((n, 2)), key, SR, 4.0)
    g_db = 20 * np.log10(gain)
    assert g_db[SR + 1000] < -3.5                       # the hit
    assert g_db[int(1.6 * SR)] > -.2                    # between hits: left alone


def test_the_delay_taps_land_on_the_exact_sample():
    x = np.zeros((SR * 2, 2))
    x[1000] = 1.0
    wet = dsp.delay_send(x, SR, 250.0, .5, hpf_hz=20.5, lpf_hz=20000.0)
    d = int(round(SR * .25))
    for k in (1, 2):
        window = np.abs(wet[1000 + k * d - 500:1000 + k * d + 500, 0])
        assert int(np.argmax(window)) == 500, k


def test_the_section_send_contour_moves_the_delay_too():
    stems = _session()
    p = _plan(stems)
    p["sections"] = []
    _, base = render.render(stems, p)
    q = copy.deepcopy(p)
    dur = stems[0].audio.shape[0] / SR
    q["sections"] = [{"label": "verse", "start_s": 0.0, "end_s": dur, "send_scale": .5}]
    _, half = render.render(stems, q)
    drop = base["mix_moves"]["vocal_delay"]["return_vs_sent_db"] - half["mix_moves"]["vocal_delay"]["return_vs_sent_db"]
    # the sent signal is measured after the contour, so the ratio holds; the
    # return itself must be about 6 dB lower
    assert abs(drop) < .5
    assert half["mix_moves"]["vocal_delay"]["sent_db"] == pytest.approx(base["mix_moves"]["vocal_delay"]["sent_db"] - 6.02, abs=.3)


def test_moves_without_anything_to_act_on_are_refused():
    p = _plan(_session())
    guitar = copy.deepcopy(p)
    guitar["tracks"][0]["role"] = "guitar_electric"          # the kick key becomes a guitar
    guitar["buses"]["drum"].pop("balance", None)              # (R114: no drum track left to balance)
    guitar["buses"]["drum"].pop("parallel", None)
    with pytest.raises(planner.PlanError, match="kick or a whole kit"):
        planner.validate(guitar)
    kit = copy.deepcopy(p)
    kit["tracks"][0]["role"] = "drums_full"
    # a kick_duck_v1 plan (R111-R120) keyed on a whole kit must say where its kick is
    kit["buses"]["bass"]["duck"] = {"law": "kick_duck_v1", "key_ids": ["Kick"], "depth_db": 2.0}
    with pytest.raises(planner.PlanError, match="key_lowpass_hz"):
        planner.validate(kit)
    no_bass = _plan(_session())
    no_bass["tracks"] = [t for t in no_bass["tracks"] if t["role"] != "bass"]
    with pytest.raises(planner.PlanError, match="no bass track"):
        planner.validate(no_bass)
    no_drums = _plan(_session())
    del no_drums["buses"]["bass"]["duck"]
    no_drums["tracks"] = [t for t in no_drums["tracks"] if t["role"] != "kick"]
    with pytest.raises(planner.PlanError, match="no drum track"):
        planner.validate(no_drums)
    silent = copy.deepcopy(p)
    for t in silent["tracks"]:
        (t.get("sends") or {}).pop("delay", None)
    with pytest.raises(planner.PlanError, match="no track sends to it"):
        planner.validate(silent)


def test_a_return_name_is_only_taken_when_the_return_is_written(tmp_path):
    # Track ids never start with "_" (analyze.track_id strips it), so a return
    # file cannot collide with a track; and a plan without a delay writes no
    # _delay.wav at all.
    assert analyze.track_id("_delay.wav") == "delay"
    stems = _session(with_vocal=False)
    p = _plan(stems)
    assert "delay" not in p
    collect = {}
    mix, report = render.render(stems, p, collect=collect)
    render.write_stem_export(str(tmp_path), p, collect, report, mix)
    assert not (tmp_path / "_delay.wav").exists()
    assert all(f["kind"] != "return" or f["filename"] == "_reverb.wav"
               for f in report["export_summary"]["files"])


# ---------------------------------------------------------------- review round 2


def test_a_dense_kick_roll_is_still_followed():
    # 32nd notes at 120 BPM: 62.5 ms apart, faster than the 120 ms release
    n = SR * 4
    k = np.arange(int(.04 * SR)) / SR
    hit = np.sin(2 * np.pi * 60 * k) * np.exp(-k / .012)
    key = np.zeros(n)
    at = SR
    while at < 3 * SR:
        key[at:at + hit.size] += hit
        at += int(.0625 * SR)
    key = np.stack([key, key], 1)
    _, gain = dsp.duck(np.ones((n, 2)), key, SR, 4.0)
    g_db = 20 * np.log10(gain)
    assert float(np.min(g_db[SR:3 * SR])) < -3.5           # the roll is ducked
    assert float(np.max(g_db[:int(.9 * SR)])) > -.05        # before it: untouched


def test_a_send_too_small_to_render_counts_as_no_send():
    p = _plan(_session())
    for t in p["tracks"]:
        if "delay" in (t.get("sends") or {}):
            t["sends"]["delay"] = planner.SEND_MIN / 2
    with pytest.raises(planner.PlanError, match="no track sends to it"):
        planner.validate(p)
