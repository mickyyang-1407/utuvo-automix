"""R125: a side letter glued to the word ("OHL" / "OHR") and doubled takes ("BV 1-1" / "BV 1-2").

Reported 2026-11-10 on a real 49-track session: the overheads and rooms sat at
+-0.10, both unison vocals ended on the right, and six backing vocals were laid
out in name order, so both takes of part 1 and part 2 were on the left.
"""
from automix import plan as planner


def _pans(ids_roles):
    tracks = [{"id": i, "role": r} for i, r in ids_roles]
    planner._spread_duplicate_pans(tracks)
    return {t["id"]: t.get("pan") for t in tracks}, {t["id"]: t.get("evidence", []) for t in tracks}


def test_glued_letters_go_hard_left_and_right_when_the_partner_exists():
    pans, evidence = _pans([("OHL.13_02", "overheads"), ("OHR.13_02", "overheads"),
                            ("RoomL.13_02", "room"), ("RoomR.13_02", "room"),
                            ("unisL.04_02", "backing_vocal"), ("unisR.04_02", "backing_vocal"),
                            ("GtrL2", "guitar_electric"), ("GtrR2", "guitar_electric"),
                            ("PadLeft", "synth"), ("PadRight", "synth")])
    for left, right in (("OHL.13_02", "OHR.13_02"), ("RoomL.13_02", "RoomR.13_02"),
                        ("unisL.04_02", "unisR.04_02"), ("GtrL2", "GtrR2"), ("PadLeft", "PadRight")):
        assert pans[left] == -1.0 and pans[right] == 1.0, (left, right, pans)
    assert "panned hard left from the file name" in evidence["OHL.13_02"]


def test_a_trailing_letter_without_a_partner_is_not_a_side():
    # "vocal", "guitar", "choir", "bell": the last letter is just the word
    pans, _ = _pans([("Lead Vocal", "lead_vocal"), ("Guitar", "guitar_electric"), ("Choir", "backing_vocal"),
                     ("Bell", "percussion"), ("Snare", "snare")])
    assert all(p in (None, 0.0) for p in pans.values()), pans
    assert planner._glued_side_hints(["Guitar", "Choir", "Bell", "Vocal", "Guitar 2"]) == {}


def test_a_free_standing_token_still_wins_and_is_not_overridden():
    # "Gtr_L" answers by itself; "KickL_R" names both sides and stays automatic
    assert planner._glued_side_hints(["Gtr_L", "Gtr_R"]) == {}
    pans, _ = _pans([("Gtr_L", "guitar_electric"), ("Gtr_R", "guitar_electric")])
    assert pans == {"Gtr_L": -1.0, "Gtr_R": 1.0}
    # the partner must be a plain glued name too: "OHR_L" commits to the left by its own token
    assert planner._glued_side_hints(["OHL_L", "OHR_L"]) == {}


def test_the_partner_has_to_differ_in_that_one_letter_only():
    assert planner._glued_side_hints(["OHL.13_02", "OHR.14_02"]) == {}
    assert planner._glued_side_hints(["ohl", "OHR"]) == {"ohl": -1, "OHR": 1}


def test_doubled_takes_are_mirrored_and_the_stack_is_balanced():
    ids = ["BV_1-1_01", "BV_1-2_01", "BV_2-1_01", "BV_2-2_01", "BV_3-1_01", "BV_3-2_01", "unisL.04_02", "unisR.04_02"]
    pans, evidence = _pans([(i, "backing_vocal") for i in ids])
    assert pans["unisL.04_02"] == -1.0 and pans["unisR.04_02"] == 1.0
    for part in "123":
        a, b = pans[f"BV_{part}-1_01"], pans[f"BV_{part}-2_01"]
        assert a < 0 < b and a == -b, (part, a, b)
    assert abs(pans["BV_1-1_01"]) > abs(pans["BV_2-1_01"]) > abs(pans["BV_3-1_01"])
    assert [pans[f"BV_{part}-2_01"] for part in "123"] == [0.7, 0.5, 0.3]
    assert abs(sum(pans.values())) < 1e-9
    assert any("mirrored against its double BV_1-2_01" in e for e in evidence["BV_1-1_01"])


def test_takes_without_a_double_keep_the_name_order_spread():
    # three singles: nothing to mirror, the earlier layout stands
    pans, _ = _pans([("BV_1-1", "backing_vocal"), ("BV_2-1", "backing_vocal"), ("BV_3-1", "backing_vocal")])
    assert pans == {"BV_1-1": -0.2, "BV_2-1": 0.0, "BV_3-1": 0.2}
    # a pair plus a single: the pair takes the outer slots, the single the centre
    pans, _ = _pans([("BV_1-1", "backing_vocal"), ("BV_1-2", "backing_vocal"), ("BV_solo", "backing_vocal")])
    assert pans == {"BV_1-1": -0.7, "BV_1-2": 0.7, "BV_solo": 0.0}


def test_plain_names_are_spread_exactly_as_before():
    ids = [f"Gtr {i}" for i in range(1, 5)]
    pans, _ = _pans([(i, "guitar_electric") for i in ids])
    assert pans == {"Gtr 1": -0.3, "Gtr 2": -0.1, "Gtr 3": 0.1, "Gtr 4": 0.3}


# ----------------------------------------------------------- the backing stack

import numpy as np

from automix import analyze, dsp, pipeline


def _vocal_session(n_backing, seconds=12, sr=44100):
    t = np.arange(sr * seconds) / sr
    rng = np.random.default_rng(5)

    def stem(name, audio):
        return analyze.Stem(name=name, path=name, sr=sr, audio=np.stack([audio, audio], 1).astype(np.float32))
    stems = [stem("Lead Vocal.wav", 0.2 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 2 * t))),
             stem("Kick.wav", 0.3 * np.sin(2 * np.pi * 55 * t) * (np.mod(t, 0.5) < 0.1)),
             stem("Bass.wav", 0.2 * np.sin(2 * np.pi * 82 * t)),
             stem("Guitar.wav", 0.1 * np.sign(np.sin(2 * np.pi * 196 * t)))]
    for i in range(n_backing):
        # different pitches and a little noise: uncorrelated voices, as a real stack is
        stems.append(stem(f"BV {i + 1}.wav", 0.2 * np.sin(2 * np.pi * (247 + 37 * i) * t) + 0.01 * rng.standard_normal(t.size)))
    analyze.detect_roles(stems)
    for s in stems:
        analyze.measure(s)
    return stems


def _stack_minus_lead(stems, p):
    tracks = {t["id"]: t for t in p["tracks"]}

    def total(role):
        acc = 0.0
        for s in stems:
            t = tracks[analyze.track_id(s.name)]
            if t["role"] == role:
                acc = acc + s.audio.mean(axis=1) * dsp.db_to_lin(t["gain_db"])
        return 10 * np.log10(np.mean(np.asarray(acc) ** 2))
    return total("backing_vocal") - total("lead_vocal")


def test_a_stack_of_backing_vocals_is_held_under_the_lead():
    stems = _vocal_session(8)
    assert sum(s.role == "backing_vocal" for s in stems) == 8
    p, _, _ = pipeline.mix(stems, planner.load_preset("pop_2020s"))
    assert _stack_minus_lead(stems, p) <= -planner.BACKING_STACK_GAP_DB + 0.2
    notes = [e for t in p["tracks"] if t["role"] == "backing_vocal" for e in t["evidence"] if "backing stack" in e]
    assert len(notes) == 8
    # trimmed together: the stack keeps its own internal balance
    trims = {float(e.split("the stack by ")[1].split(" dB")[0]) for e in notes}
    assert len(trims) == 1 and trims.pop() < -1.0


def test_without_the_rule_that_stack_is_over_the_lead(monkeypatch):
    # the same session with the rule switched off: this is what was reported
    monkeypatch.setattr(planner, "_backing_stack_ceiling", lambda *a, **k: None)
    stems = _vocal_session(8)
    p, _, _ = pipeline.mix(stems, planner.load_preset("pop_2020s"))
    assert _stack_minus_lead(stems, p) > 0.0


def test_one_backing_vocal_and_a_hand_trim_are_left_alone():
    stems = _vocal_session(1)
    p, _, _ = pipeline.mix(stems, planner.load_preset("pop_2020s"))
    assert not [e for t in p["tracks"] for e in t["evidence"] if "backing stack" in e]
    stems = _vocal_session(8)
    p, _, _ = pipeline.mix(stems, planner.load_preset("pop_2020s"), trims={"BV_1": 1.0})
    assert not [e for t in p["tracks"] for e in t["evidence"] if "backing stack" in e]


# ------------------------------------------------- stereo files in a spread group

def test_a_stereo_file_keeps_its_own_image_and_the_mono_ones_are_spread():
    ids = ["01 EG Mono", "01 EG Stereo", "02 EG Mono", "02 EG Stereo", "03 EG Mono"]
    tracks = [{"id": i, "role": "guitar_electric"} for i in ids]
    planner._spread_duplicate_pans(tracks, frozenset({"01 EG Stereo", "02 EG Stereo"}))
    pans = {t["id"]: t["pan"] for t in tracks}
    assert pans["01 EG Stereo"] == 0.0 and pans["02 EG Stereo"] == 0.0
    assert [pans[i] for i in ("01 EG Mono", "02 EG Mono", "03 EG Mono")] == [-0.2, 0.0, 0.2]
    assert any("stereo file" in e for e in tracks[1]["evidence"])
    # without the fact, the earlier layout: this is what switched a channel off
    tracks = [{"id": i, "role": "guitar_electric"} for i in ids]
    planner._spread_duplicate_pans(tracks)
    assert {t["id"]: t["pan"] for t in tracks}["01 EG Stereo"] != 0.0


def test_stereo_guitars_in_a_real_plan_are_not_spread():
    sr, t = 44100, np.arange(44100 * 6) / 44100
    rng = np.random.default_rng(9)

    def stem(name, left, right):
        return analyze.Stem(name=name, path=name, sr=sr, audio=np.stack([left, right], 1).astype(np.float32))
    tone = lambda f: 0.2 * np.sign(np.sin(2 * np.pi * f * t))
    stems = [stem("Lead Vocal.wav", 0.2 * np.sin(2 * np.pi * 220 * t), 0.2 * np.sin(2 * np.pi * 220 * t)),
             stem("Kick.wav", *(2 * [0.3 * np.sin(2 * np.pi * 55 * t) * (np.mod(t, 0.5) < 0.1)])),
             stem("01 EG Mono.wav", tone(196), tone(196)),
             stem("01 EG Stereo.wav", tone(196) + 0.05 * rng.standard_normal(t.size), tone(196) + 0.05 * rng.standard_normal(t.size)),
             stem("02 EG Mono.wav", tone(247), tone(247)),
             stem("02 EG Stereo.wav", tone(247) + 0.05 * rng.standard_normal(t.size), tone(247) + 0.05 * rng.standard_normal(t.size))]
    analyze.detect_roles(stems)
    for s in stems:
        analyze.measure(s)
    p, _, _ = pipeline.mix(stems, planner.load_preset("pop_2020s"))
    pans = {t["id"]: t["pan"] for t in p["tracks"] if t["role"] == "guitar_electric"}
    assert len(pans) == 4, pans
    assert pans["01_EG_Stereo"] == 0.0 and pans["02_EG_Stereo"] == 0.0
    assert pans["01_EG_Mono"] < 0 < pans["02_EG_Mono"]


# ------------------------------------------------------------ review round (agyr)

def test_a_delimited_letter_with_a_number_needs_its_partner_too():
    assert planner._glued_side_hints(["OH_L1", "OH_R1"]) == {"OH_L1": -1, "OH_R1": 1}
    assert planner._glued_side_hints(["Take_R1", "Take_R2"]) == {}


def test_ids_that_differ_only_in_case_both_get_their_side():
    assert planner._glued_side_hints(["ohl", "OHL", "OHR"]) == {"ohl": -1, "OHL": -1, "OHR": 1}


def test_a_glued_letter_does_not_hard_pan_a_stereo_file():
    tracks = [{"id": "PadL", "role": "synth"}, {"id": "PadR", "role": "synth"}]
    planner._spread_duplicate_pans(tracks, frozenset({"PadL", "PadR"}))
    assert [t["pan"] for t in tracks] == [0.0, 0.0]


def test_a_silent_lead_or_a_silent_stack_changes_nothing():
    for quiet in ("Lead Vocal.wav", "BV"):
        stems = _vocal_session(4)
        for s in stems:
            if s.name.startswith(quiet):
                s.audio[:] = 0.0
        tracks = [{"id": analyze.track_id(s.name), "role": s.role, "gain_db": -3.0, "evidence": []} for s in stems]
        planner._backing_stack_ceiling(stems, tracks, set(), {})
        assert all(t["gain_db"] == -3.0 for t in tracks), quiet


def test_a_zero_trim_is_not_a_hand_set_trim():
    stems = _vocal_session(8)
    tracks = [{"id": analyze.track_id(s.name), "role": s.role, "gain_db": 0.0, "evidence": []} for s in stems]
    planner._backing_stack_ceiling(stems, tracks, set(), {"BV_1": 0.0})
    assert any("backing stack" in e for t in tracks for e in t["evidence"])
    tracks = [{"id": analyze.track_id(s.name), "role": s.role, "gain_db": 0.0, "evidence": []} for s in stems]
    planner._backing_stack_ceiling(stems, tracks, set(), {"BV_1": 1.0})
    assert not any("backing stack" in e for t in tracks for e in t["evidence"])
