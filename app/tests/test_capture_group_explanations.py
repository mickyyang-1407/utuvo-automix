"""R27 -- a protected member's evidence must describe the mix that comes out of it.

The R25 plan bypassed the per-member processing of a declared capture group
but still wrote the automatic proposals into the plan as if they had been
applied: "ride applied", "vocal masking", "post-track loudness target", the
automatic placement, the style gain, the rough-mix fit.  It also listed unmask
targets that the protection had just removed, and it told two members of the
same declared group to mute one of them.

These tests pin the final state down:

* a protected member's evidence describes its role, its source, the protection
  and the numbers the renderer will really use -- and nothing else;
* the neutral record is written even when the automatic gain was already 0 dB;
* the unmask summary lists exactly the targets that are still in the plan, and
  the vacated slot is not handed to another track;
* the shared-performance advice is silenced only between two members of the
  same declared group;
* the executable numbers and the rendered audio are the R25 ones.

Everything here is synthetic: the stems are generated in memory, no audio is
read from disk, and nothing is written outside tmp.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from automix import analyze, dsp, pipeline, plan as planner, render

SR = 24000
DUR = 8.0

#: the group the R25 counterexample was recorded on
GROUPS = [{"id": "guitar-capture", "members": ["music", "mic_b"], "pan": 0.375}]
#: the same group with a manual pan and a manual trim on one member
MANUAL = {"pans": {"music": 0.0}, "trims": {"music": 2.25}}

#: the R25 reference plans and the SHA-256 of the audio they rendered
REFERENCE_DIR = Path(__file__).resolve().parent / "r25_reference"


# ------------------------------------------------------------------ fixtures

def _t() -> np.ndarray:
    return np.arange(int(DUR * SR)) / SR


def _guitar() -> np.ndarray:
    t = _t()
    return 0.2 * np.sin(2 * np.pi * 110 * t) + 0.08 * np.sin(2 * np.pi * 3200 * t)


def _lead() -> np.ndarray:
    t = _t()
    return (0.15 * np.sin(2 * np.pi * 660 * t) + 0.06 * np.sin(2 * np.pi * 2640 * t)) * (
        (t >= 2) & (t < 4)
    )


def _stem(name: str, audio: np.ndarray, role: str) -> analyze.Stem:
    stem = analyze.Stem(
        name, name, np.column_stack([audio, audio]), SR, role=role, role_source="user"
    )
    analyze.measure(stem)
    return stem


def _baseline_stems() -> list:
    """The three-stem session the R25 counterexample was recorded on."""
    x = _guitar()
    return [
        _stem("music.wav", x, "guitar_electric"),
        _stem("mic_b.wav", x * 0.7, "guitar_electric"),
        _stem("lead.wav", _lead(), "lead_vocal"),
    ]


def _room_stems() -> list:
    """The same session with more members: a third unmask candidate that is not
    in any group, and a second group to tell the duplicate advice apart."""
    x = _guitar()
    return [
        _stem("music.wav", x, "guitar_electric"),
        _stem("mic_b.wav", x * 0.7, "guitar_electric"),
        _stem("gtr_c.wav", x * 0.5, "guitar_electric"),
        _stem("gtr_d.wav", x * 0.45, "guitar_electric"),
        _stem("gtr_e.wav", x * 0.4, "guitar_electric"),
        _stem("lead.wav", _lead(), "lead_vocal"),
    ]


def _bass_session() -> list:
    """A kick and two basses on one fundamental: the low-end carve has to
    decide whether to make room, and a protected bass cannot take the dip."""
    t = _t()
    bass = 0.25 * np.sin(2 * np.pi * 60 * t)
    kick = 0.3 * np.sin(2 * np.pi * 60 * t) * ((t % 0.5) < 0.2)
    return [
        _stem("kick.wav", kick, "kick"),
        _stem("bass_a.wav", bass, "bass"),
        _stem("bass_b.wav", bass * 0.8, "bass"),
    ]


def _preset() -> dict:
    return planner.load_preset("pop_2020s")


def _options() -> dict:
    """The plan options the R25 counterexample was recorded with."""
    return {
        "balance_preserve": 1.0,
        "source_balance_profile": "raw_tracks",
        "reverb_amount": 0.0,
        "target_lufs": -20.0,
    }


def _plan(stems, **kw):
    options = _options()
    options.update(kw)
    return planner.build_plan(stems, _preset(), **options)


def _mix(stems, **kw):
    options = _options()
    options.update(kw)
    return pipeline.mix(stems, _preset(), **options)


def _track(plan: dict, tid: str) -> dict:
    return next(t for t in plan["tracks"] if t["id"] == tid)


def _story(plan: dict, tid: str) -> str:
    return "\n".join(_track(plan, tid)["evidence"])


def _executable(plan: dict) -> dict:
    """Everything the renderer reads: the numbers, not the story."""
    return {
        "source": plan["source"],
        "tracks": [
            {k: v for k, v in track.items() if k != "evidence"} for track in plan["tracks"]
        ],
        "buses": plan["buses"],
        "sections": plan["sections"],
        "reverb": plan["reverb"],
    }


def _r25(case: str) -> dict:
    return json.loads((REFERENCE_DIR / f"OLD-{case}-plan.json").read_text(encoding="utf-8"))


def _r25_wav_sha(case: str) -> str:
    record = json.loads((REFERENCE_DIR / "r25-audio-sha256.json").read_text(encoding="utf-8"))
    return record["wav_sha256"][case]


def _without_ratio_law(plan: dict) -> dict:
    """R104 added ``ratio_law`` to every compressor; R25 plans predate it.  A
    plan without the field still renders the R25 way, so it is set aside here."""
    plan = json.loads(json.dumps(plan))
    for block in plan.get("tracks", []) + list(plan.get("buses", {}).values()):
        if isinstance(block.get("comp"), dict):
            block["comp"].pop("ratio_law", None)
    return plan


def _sha(path) -> str:
    with Path(path).open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def _duplicate_pairs(plan: dict) -> set:
    pairs = set()
    for warning in plan["warnings"]:
        match = re.match(r"^(\S+) and (\S+) carry the same performance", warning)
        if match:
            pairs.add(frozenset((match.group(1), match.group(2))))
    return pairs


# -------------------------------------------------- the numbers are R25's

def test_the_plan_without_a_group_is_the_r25_plan():
    plan, _, _ = _mix(_baseline_stems())
    plan = _without_ratio_law(plan)
    r25 = _r25("none")
    assert _executable(plan) == _executable(r25)
    assert plan["unmask"]["targets"] == r25["unmask"]["targets"]
    assert "capture_groups" not in plan
    # the whole plan, not only its numbers: without a declaration the plan is
    # the one R25 produced, down to the last evidence line
    mine = {k: v for k, v in plan.items() if k != "created_at"}
    theirs = {k: v for k, v in r25.items() if k != "created_at"}
    assert mine == theirs


def test_the_grouped_plan_keeps_the_r25_numbers():
    plan, _, _ = _mix(_baseline_stems(), capture_groups=GROUPS)
    plan = _without_ratio_law(plan)
    r25 = _r25("group")
    assert _executable(plan) == _executable(r25)


def test_the_manual_plan_keeps_the_r25_numbers():
    plan, _, _ = _mix(_baseline_stems(), capture_groups=GROUPS, **MANUAL)
    plan = _without_ratio_law(plan)
    r25 = _r25("manual")
    assert _executable(plan) == _executable(r25)


@pytest.mark.parametrize(
    "case,options",
    [
        ("none", {}),
        ("group", {"capture_groups": GROUPS}),
        ("manual", {"capture_groups": GROUPS, **MANUAL}),
    ],
)
def test_the_rendered_audio_is_the_r25_audio(tmp_path, case, options):
    # today's plan, compressed the R25 way, renders R25's audio to the bit
    plan, _, _ = _mix(_baseline_stems(), **options)
    rev = _preset().get("reverb") or {}
    reverb = {"decay_s": float(rev.get("decay_s", 1.6)), "predelay_ms": float(rev.get("pre_delay_ms", 20.0))}
    audio, _ = render.render(_baseline_stems(), _without_ratio_law(plan), reverb)
    out = tmp_path / f"{case}.wav"
    render.write_wav24(str(out), audio, SR)
    assert _sha(out) == _r25_wav_sha(case)


# ------------------------------------------- what a protected member claims

def test_a_protected_member_does_not_claim_the_processing_it_bypassed():
    plan = _plan(_room_stems(), capture_groups=GROUPS)
    for tid in ("music", "mic_b"):
        story = _story(plan, tid)
        for claim in (
            "style gain for",              # the automatic gain
            "post-track loudness target",  # the loudness matching
            "vocal masking",               # the dynamic unmask
            "ride ±",                      # the ride
            "de-esser ",                   # the de-esser
            "hum notch",                   # the mains notch
            "spread to",                   # the automatic placement
            "included once in gain_db",    # the rough-mix fit
        ):
            assert claim not in story, (tid, claim)
        track = _track(plan, tid)
        assert track.get("level_match") is None
        assert track.get("dynamic_eq") is None
        assert track.get("ride") is None
        assert track.get("deess") is None
        assert track.get("comp") is None
        assert track["eq"] == []
        # and the final state is described
        assert "per-member processing bypassed" in story
        assert "final gain +0.00 dB" in story
        assert "common pan +0.38" in story


def test_the_neutral_record_is_written_even_when_the_gain_was_already_zero():
    """The protection is recorded even when it changes no number at all."""
    stems = _baseline_stems()
    plain = _plan(stems)
    assert _track(plain, "music")["gain_db"] == 0.0
    assert _track(plain, "mic_b")["gain_db"] == 0.0
    plan = _plan(stems, capture_groups=GROUPS)
    for tid in ("music", "mic_b"):
        story = _story(plan, tid)
        assert "per-member processing bypassed" in story
        assert "final gain +0.00 dB" in story
        assert "0 dB base, no manual trim" in story


def test_the_abandoned_automatic_gain_is_not_claimed():
    """With the style's own gain in play, the member's story must not keep it."""
    stems = _baseline_stems()
    plain = _plan(stems, balance_preserve=0.0)
    assert _track(plain, "music")["gain_db"] != 0.0
    plan = _plan(stems, capture_groups=GROUPS, balance_preserve=0.0)
    assert _track(plan, "music")["gain_db"] == 0.0
    story = _story(plan, "music")
    assert "style gain for" not in story
    assert "final gain +0.00 dB" in story


def test_the_final_manual_pan_and_trim_are_described_from_the_final_numbers():
    plan = _plan(_room_stems(), capture_groups=GROUPS, **MANUAL)
    track = _track(plan, "music")
    assert track["gain_db"] == 2.25
    assert track["pan"] == 0.0
    story = _story(plan, "music")
    assert "final gain +2.25 dB" in story
    assert "manual trim +2.25 dB" in story
    assert "explicit pan +0.00 overrides the group's common pan +0.38" in story
    # the member without a manual pan keeps the group's one pan
    assert "common pan +0.38" in _story(plan, "mic_b")


def test_a_manual_trim_beyond_the_stage_limit_says_it_was_clamped():
    plan = _plan(_room_stems(), capture_groups=GROUPS, trims={"music": 40.0})
    track = _track(plan, "music")
    assert track["gain_db"] == planner.MAX_STAGE_DB
    story = _story(plan, "music")
    assert "clamped to +18.00 dB by the stage limit" in story


def test_a_hum_notch_is_not_claimed_for_a_protected_member():
    stems = _room_stems()
    for stem in stems:
        if stem.name == "music.wav":
            stem.metrics["hum_hz"] = 50.0
    plain = _plan(stems)
    assert any("hum notched" in w for w in plain["warnings"])
    plan = _plan(stems, capture_groups=GROUPS)
    assert not any("hum notched" in w for w in plan["warnings"])
    assert any("hum found, not notched" in w for w in plan["warnings"])
    assert "hum notch" not in _story(plan, "music")


def test_the_low_end_carve_is_not_claimed_for_a_protected_bass():
    stems = _bass_session()
    plain = _plan(stems)
    assert any("make room for the kick" in w for w in plain["warnings"])
    plan = _plan(
        stems,
        capture_groups=[{"id": "bass-capture", "members": ["bass_a", "bass_b"], "pan": 0.0}],
    )
    assert not any("make room for the kick" in w for w in plan["warnings"])
    assert any("dip at" in w and "not applied" in w for w in plan["warnings"])
    assert "room for the kick" not in _story(plan, "bass_a")
    # the ungrouped kick keeps its own numbers
    assert _track(plan, "kick")["eq"] == _track(plain, "kick")["eq"]


# ------------------------------------------------------- the unmask summary

def test_the_unmask_summary_matches_the_targets_that_are_really_there():
    stems = _room_stems()
    plain = _plan(stems)
    assert plain["unmask"]["targets"] == ["music", "mic_b"]
    plan = _plan(stems, capture_groups=GROUPS)
    live = [t["id"] for t in plan["tracks"] if t.get("dynamic_eq")]
    assert plan["unmask"]["targets"] == live
    assert plan["unmask"]["targets"] == []
    # the vacated slot is not handed to the next candidate
    assert all(t.get("dynamic_eq") is None for t in plan["tracks"])
    assert "gtr_c" not in plan["unmask"]["targets"]
    # and the plan says what happened to the protected candidates
    assert any("bypassed" in d for d in plan["unmask"]["diagnostics"])


def test_the_unmask_summary_keeps_its_policy_and_its_curves():
    stems = _room_stems()
    plain = _plan(stems)
    grouped = _plan(stems, capture_groups=GROUPS)
    for key in ("mode", "method", "intensity", "max_targets", "priority_source_ids", "policy"):
        assert grouped["unmask"][key] == plain["unmask"][key]


# ------------------------------------------------------ the duplicate advice

def test_the_shared_performance_advice_is_only_silenced_inside_one_group():
    groups = [
        {"id": "guitar-capture", "members": ["music", "mic_b"], "pan": 0.375},
        {"id": "second-capture", "members": ["gtr_d", "gtr_e"], "pan": -0.2},
    ]
    plan = _plan(_room_stems(), capture_groups=groups)
    pairs = _duplicate_pairs(plan)
    # the same group: the shared performance is the point of the group
    assert frozenset({"music", "mic_b"}) not in pairs
    assert frozenset({"gtr_d", "gtr_e"}) not in pairs
    # a member against a track outside the group, and across two groups
    assert frozenset({"music", "gtr_c"}) in pairs
    assert frozenset({"mic_b", "gtr_c"}) in pairs
    assert frozenset({"music", "gtr_d"}) in pairs


def test_without_a_group_every_duplicate_pair_is_still_flagged():
    plan = _plan(_room_stems())
    pairs = _duplicate_pairs(plan)
    assert frozenset({"music", "mic_b"}) in pairs
    assert frozenset({"gtr_d", "gtr_e"}) in pairs
    assert frozenset({"music", "gtr_c"}) in pairs


# ---------------------------------------------- what the group does not touch

def test_the_imported_arrays_are_not_touched():
    stems = _room_stems()
    before = [s.audio.copy() for s in stems]
    _plan(stems, capture_groups=GROUPS)
    for stem, original in zip(stems, before):
        assert np.array_equal(stem.audio, original)


def test_a_track_outside_the_group_keeps_its_numbers_and_its_story():
    stems = _room_stems()
    plain = _plan(stems)
    grouped = _plan(stems, capture_groups=GROUPS)
    for tid in ("gtr_c", "gtr_d", "gtr_e", "lead"):
        assert _track(grouped, tid) == _track(plain, tid)


def test_the_rendered_member_is_the_imported_audio_with_the_final_numbers():
    stems = _room_stems()
    collect = {}
    _mix(stems, capture_groups=GROUPS, collect=collect)
    imported = next(s.audio for s in stems if s.name == "music.wav")
    expected = render._pan(imported * dsp.db_to_lin(0.0), 0.375)
    assert float(np.max(np.abs(collect["stems"]["music"] - expected))) < 1e-12


def test_the_rendered_manual_member_is_the_imported_audio_with_the_final_numbers():
    stems = _room_stems()
    collect = {}
    _mix(stems, capture_groups=GROUPS, collect=collect, **MANUAL)
    imported = next(s.audio for s in stems if s.name == "music.wav")
    expected = render._pan(imported * dsp.db_to_lin(2.25), 0.0)
    assert float(np.max(np.abs(collect["stems"]["music"] - expected))) < 1e-12


# ------------------------------------------------------------ the receipt

def test_the_receipt_keeps_its_fields_and_the_plan_validates():
    plan = _plan(_room_stems(), capture_groups=GROUPS)
    receipt = plan["capture_groups"]
    assert receipt["mode"] == "declared_capture_group"
    assert set(receipt) == {"mode", "groups", "boundary"}
    group = receipt["groups"][0]
    assert set(group) == {"id", "members", "protected_tracks", "pan", "common_send", "notes"}
    assert group["protected_tracks"] == ["music", "mic_b"]
    planner.validate(plan)
