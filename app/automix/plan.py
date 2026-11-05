"""Turn analysed stems plus a style preset into a MixPlan.

The plan is the contract between deciding and rendering: it is plain JSON, it
conforms to ``schemas/mix_plan.schema.json``, and it records *why* each number
was chosen in ``evidence``.  Rendering reads nothing but this object, so a user
can hand-edit a plan and re-render without touching code.
"""

from __future__ import annotations

import json
import math
import numbers
import os
import re
import sys

import numpy as np
from datetime import datetime, timezone

from . import analyze, capture_groups as capmod, dsp, reference as refmod, sections, unmask, source_balance as balancemod, reference_balance as fitmod
from .analyze import DRUM_ROLES, ROLES, VOCAL_ROLES, Stem

SCHEMA_VERSION = 1
MAX_MASK_CARVES = unmask.MAX_TARGETS  # compatibility name for the one cap owner

#: how far the automatic gain staging may move a single stem.  This is not the
#: EQ guard (that is about tonal correction); a stem can legitimately need a
#: large move -- a DI bass and a miked amp are 15 dB apart -- but past ~18 dB
#: something is wrong with the source and the receipt should say so.
MAX_STAGE_DB = 18.0

#: Where the vocal sits against everything else in a finished mix, measured
#: only where it sings and with the renderer's own meter
#: (dsp.vocal_to_rest_db): the median of the 46 MedleyDB mixes of MUSDB18
#: is -1.9 dB (Pop -1.7, Rock -1.3).  R109 used 11 of them (-2.7), 0.8 dB quieter -- and
#: the owner heard the vocal as a little small.  A style's vocal_forward_db (and
#: its techniques.vocal_lift_db) is added on top.  The renderer measures the real buses
#: after compression and moves the whole vocal bus (lead and backing) to this
#: target: compressing a sustained vocal twice with half make-up cost it
#: 3.5-4.5 dB, and a raw session with many accompaniment tracks buries a
#: single vocal track however its own gain is set.
VOCAL_VAR_PRO_MEDIAN_DB = -1.9
VOCAL_BALANCE_LAW = "vocal_var_v1"
#: the correction is a balance fix, not a rescue: beyond this it is reported
VOCAL_BALANCE_MAX_BOOST_DB = 9.0
VOCAL_BALANCE_MAX_CUT_DB = 6.0
#: The drum and bass buses against the rest of the band (no vocal), where
#: they play -- the same ruler -- in the 46 MedleyDB mixes of MUSDB18.  A style's
#: drum_forward_db / bass_forward_db is added on top.
DRUM_VAR_PRO_MEDIAN_DB: float | None = -5.2   # Pop -5.2, Rock -2.5
BASS_VAR_PRO_MEDIAN_DB: float | None = -3.9   # Pop -4.6, Rock -3.7
PART_BALANCE_LAW = "part_var_v1"
PART_BALANCE_MAX_DB = 6.0
#: R116: the largest cut.  At 6 dB both buses ran out in half of the 41
#: MedleyDB songs with Pop 2020s (drums up to 9.5 dB, bass up to 5.1 dB over
#: their target, measured on the MedleyDB corpus): the planner's per-track levels start the
#: band louder than a finished mix, and when drums and bass are both cut each
#: one's reference shrinks with the other.  The limits are written into every
#: plan, so plans written with 6 dB replay as they were.
PART_BALANCE_MAX_CUT_DB = 12.0
#: R116: every balance (vocal, drums, bass) is measured through the master EQ,
#: what the listener hears and what the professional medians were measured
#: on (render.BALANCE_METER_LAW); plans without it measured the bare buses
BALANCE_METER_LAW = "master_eq_v1"
#: R117: the reverb and delay sends of a balanced bus follow its balance
#: move, as a post-fader send does.  Before, only the dry bus moved: a vocal
#: pulled down 6 dB kept its reverb where it was and came out 6 dB wetter, and drums cut
#: 12 dB kept their plate.  Plans without the field render as before.
BALANCE_SEND_LAW = "post_fader_v1"
#: R117: how tracks and buses saturate (dsp.SATURATION_LAWS).  The old law
#: normalised the tanh to each track's single largest peak at the sample
#: rate: a gain of up to k/tanh(k) below that peak (it also lifted every
#: saturated track's level: drums +2.75 dB at pop's drive) with harmonics
#: folding back.  Measured on 12 professional stems the two laws move density
#: and brightness alike; the new one
#: keeps the level contour between passages and does not fold back.
SATURATION_LAW = "env_tanh_os4_v1"
#: R119: every compressor's detector reads only the past (dsp.COMPRESSOR_LAWS);
#: plans without the field keep the centred detector they were rendered with
COMPRESSOR_LAW = "trailing_box_v1"
#: R119: the vocal reverb is turned down while the voice sings and blooms in
#: the gaps (dsp.voice_duck; the renderer reads this name from here); depth from the style's
#: techniques.vocal_reverb_duck_db, release an eighth note (clamped)
REVERB_DUCK_LAW = "voice_duck_v1"
REVERB_DUCK_MAX_DB = 8.0
REVERB_DUCK_RELEASE_MS = (120.0, 400.0)
REVERB_DUCK_DEFAULT_RELEASE_MS = 250.0
#: R117: drums that still move.  Measured on 46 MedleyDB songs the drum bus
#: came out with 7-9 dB of level spread (p90-p10 of 50 ms windows) against
#: 16.6 in the professional stems; taking out one stage at a time, the bus
#: compressor cost 3.7 dB of it and the close-mic compressors 1.9.  The bus threshold sat 3 dB under the
#: bus's average level, so every hit was pushed 6-9 dB down -- the published
#: range for drum-bus glue is 2-5 dB at 2:1-4:1 with an attack that lets the
#: hit through.  New plans:
#: the drum bus threshold sits above the average (hits get a few dB), a
#: 2:1 ceiling and a slower attack; close drum mics compress at half the
#: ratio with an attack that lets the transient through.  These are plain
#: compressor numbers in the plan, so older plans render as they were.
DRUM_BUS_THRESHOLD_OFFSET_DB = 6.0
DRUM_BUS_MAX_RATIO = 2.0
DRUM_BUS_MIN_ATTACK_MS = 30.0
DRUM_TRACK_RATIO_SCALE = 0.5
DRUM_TRACK_MIN_ATTACK_MS = 25.0
#: the receipt each changed drum compressor carries: what the style's dial
#: alone would have written (the renderer does not read it)
DRUM_OPEN_LAW = "drum_open_v1"
#: R118: the few faders a user gets after the mix ("vocal a little louder"):
#: one dB offset per group, applied by the renderer after every automatic
#: balance, the group's reverb and delay moving with it.  Absent = 0 dB, so
#: plans without them render as before.
USER_TRIM_GROUPS = ("vocal", "drum", "bass", "music", "fx")
USER_TRIM_MAX_DB = 12.0


def _user_trims(group_trims: dict | None) -> dict:
    """The user's group faders, checked: known groups, finite, within the limit."""
    out = {}
    for name, value in (group_trims or {}).items():
        if name not in USER_TRIM_GROUPS:
            raise PlanError(f"no fader for group {name!r} (groups: {', '.join(USER_TRIM_GROUPS)})")
        v = float(value)
        if not math.isfinite(v) or abs(v) > USER_TRIM_MAX_DB:
            raise PlanError(f"group fader {name}: {value!r} is not between -{USER_TRIM_MAX_DB:g} and +{USER_TRIM_MAX_DB:g} dB")
        if abs(v) >= 0.05:
            out[name] = round(v, 1)
    return out


def _open_drum_track_comp(spec: dict) -> tuple[dict, dict]:
    """R117: a close drum mic's compressor at half the ratio, slower attack.
    Returns the compressor and the receipt (``comp_style_dial``, kept next to
    the compressor: the compressor block holds only what the renderer reads)."""
    was = {"law": DRUM_OPEN_LAW, "ratio": spec["ratio"], "attack_ms": spec["attack_ms"]}
    spec["ratio"] = round(1.0 + (float(spec["ratio"]) - 1.0) * DRUM_TRACK_RATIO_SCALE, 2)
    spec["attack_ms"] = max(float(spec["attack_ms"]), DRUM_TRACK_MIN_ATTACK_MS)
    return spec, was

#: Three mixing moves a style can ask for (preset "techniques"); each is a
#: block in the plan, so a plan written before they existed has none of them
#: and renders as it did.
#:   kick_duck_v1  -- the bass bus dips under the kick (a real sidechain,
#:                    driven by the kick's own processed track)
#:   parallel_v1   -- New-York compression on the drum bus
#:   tempo_delay_v1 -- a vocal delay on the beat, ducked while the voice sings
KICK_DUCK_LAW = "kick_duck_v1"
KICK_DUCK_MAX_DB = 6.0
DRUM_PARALLEL_LAW = "parallel_v1"
VOCAL_DELAY_LAW = "tempo_delay_v1"
#:   vocal_character_v1 -- weight and edge on the vocal bus: a parallel
#:                    compressed copy, saturation, body/presence/air EQ
VOCAL_CHARACTER_LAW = "vocal_character_v1"
#:   tempo_space_v1 -- the reverb's pre-delay and decay locked to the song's
#:                    tempo (R116): each is the beat times the power of two
#:                    that comes nearest the style's own value (so never more than a
#:                    factor sqrt(2) away from it, unless a safety clamp --
#:                    0.3-8 s, 5-150 ms -- applies, which the receipt then
#:                    says: "clamped"), the published rule being a
#:                    tail that has died by the next backbeat and a pre-delay
#:                    that is a note value.  Only when the tempo is clear;
#:                    the plan's decay_s / pre_delay_ms are what the renderer
#:                    uses, so plans written before this replay as they were.
REVERB_TIMING_LAW = "tempo_space_v1"
REVERB_TIMING_PRE_MS = (5.0, 150.0)
REVERB_TIMING_DECAY_S = (0.3, 8.0)
#: beats per delay repeat, by musical division
DELAY_DIVISIONS = {"1/4": 1.0, "1/8": 0.5, "1/8d": 0.75, "1/4d": 1.5}
#: a delay is only locked to the tempo when the pulse is clear (comb score)
#: and both halves of the song find the same one, and no 2:3 / 3:4 rival
#: fits almost as well (dsp.TEMPO_RIVAL_SHARE).  Measured on 11 MUSDB18
#: songs + Electron against librosa's beat tracker: at these limits 5 songs
#: lock and all 5 are on the beat; the songs a looser limit would have added
#: include the ones that were wrong.
TEMPO_LOCK_MIN_STRENGTH = 0.35
#: when the tempo is not clear: a slapback, which needs no tempo
SLAP_MS = 110.0
#: a send at or below this is skipped by the renderer (plate and delay alike)
SEND_MIN = 1e-4
SLAP_MAX_FEEDBACK = 0.15

#: sources worth riding: sustained material whose level should stay put across
#: sections.  Drums are deliberately excluded -- their dynamics *are* the part.
RIDE_ROLES = frozenset(
    {
        "lead_vocal",
        "backing_vocal",
        "bass",
        "sub_808",
        "guitar_electric",
        "guitar_acoustic",
        "keys",
        "piano",
        "synth",
        "pad",
        "strings",
        "woodwinds",
    }
)

#: how far the ride may move a stem, and how slowly
RIDE_MAX_DB = 2.5
RIDE_SMOOTH_S = 3.0

#: sibilance index above which a vocal gets a de-esser, and how hard
DEESS_THRESHOLD = 0.03
DEESS_MAX_AMOUNT = 1.2

#: how far section character may move: chorus slightly wider and wetter, verse
#: slightly tighter and drier.  Small on purpose -- this is the difference
#: between "produced" and "the reverb is pumping".
SECTION_WIDTH = {"chorus": 1.04, "verse": 0.97, "other": 1.0}
SECTION_SEND = {"chorus": 1.12, "verse": 0.90, "other": 1.0}
#: air shelf per section, in dB.  Small on purpose: this is "the chorus opens
#: up", not "the chorus has a different EQ".
SECTION_AIR = {"chorus": 0.5, "verse": -0.4, "other": 0.0}

#: the legal compressor threshold range of the shipped plan schema
#: (``schemas/mix_plan.schema.json`` -> ``compressor.threshold_db``).  A bus's
#: estimated threshold is only a hint, but a hint outside this range makes the
#: saved plan invalid, so an estimate is bounded to it.  The renderer places
#: the threshold from ``threshold_offset_db`` whenever that exists, so
#: bounding the estimate does not move the audio.
COMP_THRESHOLD_DB_RANGE = (-60.0, 0.0)

#: how a compressor block's ``ratio`` is read.  ``legacy_amount_v1`` is what
#: every plan written before the field existed was rendered with: the ratio
#: was turned back into a 0..1 amount and then scaled a second time, so a
#: written 2:1 compressed at 1.2:1 (``1 + (r-1)**2/5`` below 6:1).  Those
#: plans keep that reading so they replay as they always did.  New plans
#: write ``literal_v1``: the written ratio is the ratio.
COMP_RATIO_LAWS = ("legacy_amount_v1", "literal_v1")
LEGACY_COMP_RATIO_LAW = "legacy_amount_v1"
COMP_RATIO_LAW = "literal_v1"



#: role -> mix group, used to look up the preset's per-group amounts
ROLE_GROUP = {
    "kick": "drum",
    "snare": "drum",
    "hats": "drum",
    "toms": "drum",
    "cymbals": "drum",
    "overheads": "drum",
    "room": "drum",
    "drums_full": "drum",
    "percussion": "drum",
    "bass": "bass",
    "sub_808": "bass",
    "lead_vocal": "vocal",
    "backing_vocal": "vocal",
    "guitar_electric": "music",
    "guitar_acoustic": "music",
    "keys": "music",
    "piano": "music",
    "synth": "music",
    "pad": "music",
    "strings": "music",
    "brass": "music",
    "woodwinds": "music",
    "fx": "fx",
    "other": "music",
}

#: high-pass corner per role: how much rumble this source has no business keeping
HPF_HZ = {
    "kick": 30.0,
    "snare": 80.0,
    "hats": 320.0,
    "toms": 60.0,
    "cymbals": 400.0,
    # the whole kit from above: keep the shells, lose the rumble
    "overheads": 140.0,
    "room": 60.0,
    "drums_full": 30.0,
    "percussion": 100.0,
    "bass": 28.0,
    "sub_808": 24.0,
    "lead_vocal": 90.0,
    "backing_vocal": 110.0,
    "guitar_electric": 95.0,
    "guitar_acoustic": 85.0,
    "keys": 60.0,
    "piano": 45.0,
    "synth": 40.0,
    "pad": 120.0,
    "strings": 70.0,
    "brass": 65.0,
    "woodwinds": 60.0,
    "fx": 40.0,
    "other": 50.0,
}

#: how much of the preset's reverb amount each role receives
REVERB_WEIGHT = {
    # R114: 0.55 -> 0.75 with the plate_v2 return (no low end in it); reported
    # 2026-11-04: "vocal 都會有一點偏幹"
    "lead_vocal": 0.75,
    "backing_vocal": 0.85,
    "snare": 0.45,
    "toms": 0.5,
    "drums_full": 0.12,
    "percussion": 0.35,
    "kick": 0.0,
    "hats": 0.10,
    "cymbals": 0.15,
    "overheads": 0.08,
    # a room mic already is the room
    "room": 0.0,
    "bass": 0.0,
    "sub_808": 0.0,
    "guitar_electric": 0.55,
    "guitar_acoustic": 0.60,
    "keys": 0.55,
    "piano": 0.60,
    "synth": 0.40,
    "pad": 0.70,
    "strings": 0.70,
    "brass": 0.45,
    "woodwinds": 0.60,
    "fx": 0.80,
    "other": 0.35,
}

#: role -> pan.  Low end stays centred; everything else gets a little space.
PAN = {
    "kick": 0.0,
    "snare": 0.0,
    "bass": 0.0,
    "sub_808": 0.0,
    "lead_vocal": 0.0,
    "hats": 0.22,
    "toms": -0.18,
    "cymbals": 0.30,
    # a stereo pair keeps its own image; a mono L/R pair is a capture group
    "overheads": 0.0,
    "room": 0.0,
    "drums_full": 0.0,
    "percussion": 0.25,
    "guitar_electric": -0.30,
    "guitar_acoustic": 0.25,
    "keys": -0.20,
    "piano": 0.18,
    "synth": -0.22,
    "pad": 0.0,
    "strings": -0.12,
    "brass": 0.16,
    "woodwinds": -0.20,
    "fx": 0.35,
    "backing_vocal": -0.15,
    "other": 0.0,
}


class PlanError(RuntimeError):
    pass


def _manual_reverb_amount(value, *, optional: bool = False):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise PlanError("reverb amount must be a finite number in 0..1")
    try:
        amount = float(value)
    except (OverflowError, ValueError):
        raise PlanError("reverb amount must be a finite number in 0..1") from None
    if not math.isfinite(amount) or not 0.0 <= amount <= 1.0:
        raise PlanError("reverb amount must be a finite number in 0..1")
    return amount


def _validate_style_intent(intent: dict) -> None:
    """Descriptive provenance is optional and never an executable processor."""
    required = {"applicability", "caveat", "numeric_basis", "source_refs"}
    if not isinstance(intent, dict) or set(intent) != required:
        raise PlanError("style intent requires applicability, caveat, numeric_basis and source_refs")
    if any(not isinstance(intent[key], str) or not intent[key].strip()
           for key in required - {"source_refs"}):
        raise PlanError("style intent descriptions must be nonempty strings")
    if not isinstance(intent["source_refs"], list):
        raise PlanError("style intent source_refs must be a list")
    for source in intent["source_refs"]:
        if (not isinstance(source, dict) or set(source) != {"title", "url", "principle", "access"}
                or any(not isinstance(value, str) or not value.strip() for value in source.values())
                or not source["url"].startswith("https://")):
            raise PlanError("style source reference requires title, https url, principle and access")


def style_text_en(preset_dir: str | None = None) -> dict:
    """English texts of the built-in styles (the presets are written in Chinese).

    Display only: the planner never reads them, so they cannot change a mix.
    """
    if preset_dir is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        preset_dir = os.path.join(os.path.dirname(here), "presets")
    try:
        with open(os.path.join(preset_dir, "i18n", "en.json"), encoding="utf-8") as fh:
            table = json.load(fh)
    except (OSError, ValueError):
        return {}
    return table if isinstance(table, dict) else {}


def load_preset(style: str, preset_dir: str | None = None) -> dict:
    """Accept either a style id (``pop_2020s``) or a path to a preset JSON."""
    if preset_dir is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        preset_dir = os.path.join(os.path.dirname(here), "presets")
    path = style if style.endswith(".json") else os.path.join(preset_dir, f"{style}.json")
    if not os.path.exists(path):
        available = []
        if os.path.isdir(preset_dir):
            available = sorted(
                os.path.splitext(n)[0] for n in os.listdir(preset_dir) if n.endswith(".json")
            )
        raise PlanError(f"unknown style {style!r}; available: {', '.join(available) or 'none'}")
    with open(path, encoding="utf-8") as fh:
        preset = json.load(fh)
    for key in ("id", "target_lufs", "true_peak_db", "role_gains_db", "compression"):
        if key not in preset:
            raise PlanError(f"{os.path.basename(path)}: missing {key!r}")
    if preset.get("intent") is not None:
        _validate_style_intent(preset["intent"])
    return preset


def role_group(role: str) -> str:
    return ROLE_GROUP.get(role, "music")


def _comp_params(group: str, amount: float) -> dict:
    """Map a 0..1 'how much compression' dial onto real compressor settings.

    More amount means a lower threshold, a higher ratio, a faster attack and a
    longer release -- the shape every engineer reaches for when asked to add
    more of the same compression.
    """
    amount = float(max(0.0, min(1.0, amount)))
    timing = {
        # attack_ms at amount=0, release_ms at amount=0
        "drum": (30.0, 3.0, 60.0, 150.0),
        "vocal": (25.0, 6.0, 80.0, 200.0),
        "bass": (28.0, 5.0, 90.0, 220.0),
        "music": (45.0, 12.0, 120.0, 300.0),
        "bus": (35.0, 15.0, 150.0, 300.0),
    }
    a_slow, a_fast, r_short, r_long = timing.get(group, timing["music"])
    attack = a_slow + (a_fast - a_slow) * amount
    release = r_short + (r_long - r_short) * amount
    return {
        "threshold_db": round(-6.0 - 34.0 * amount, 1),
        "ratio": round(1.0 + 5.0 * amount, 2),
        "ratio_law": COMP_RATIO_LAW,
        "attack_ms": round(attack, 1),
        "release_ms": round(release, 0),
        # No knee_db / makeup_db / mix here: the DSP's knee is fixed at 6 dB,
        # it is always 100% wet, and the make-up is not a fixed number -- it is
        # half the average gain reduction, which depends on the audio.  So the
        # *rule* is written instead, which is what makes the plan enough to
        # determine the audio: a renderer that reads this block and applies
        # this rule gets the same result.
        "makeup": {"mode": "half_average_reduction", "cap_db": 12.0},
    }


def _source_vocal_balance(stems: list[Stem], fitted: dict | None = None) -> float | None:
    """The vocal's level against everything else in the stems as delivered.

    ``fitted`` (track id -> dB) is an accepted same-song reference fit: that
    fit is the balance somebody vouched for, so it is applied here too --
    measuring the raw stems would undo it.
    """
    fitted = fitted or {}
    n = max((s.audio.shape[0] for s in stems), default=0)
    if n == 0:
        return None
    vocal = np.zeros((n, 2))
    rest = np.zeros((n, 2))
    for s in stems:
        a = s.audio
        if a.ndim == 1:
            a = np.stack([a, a], axis=1)
        g = dsp.db_to_lin(float(fitted.get(analyze.track_id(s.name), 0.0)))
        if s.role in VOCAL_ROLES:
            vocal[: a.shape[0]] += a * g
        elif ROLE_GROUP.get(s.role) != "fx":
            # the renderer compares dry buses and leaves the fx bus out of
            # both sides; the source is measured the same way
            rest[: a.shape[0]] += a * g
    if not vocal.any() or not rest.any():
        return None
    value, _windows = dsp.vocal_to_rest_db(vocal, rest, int(stems[0].sr))
    return value


def _gain_stage(
    stems: list[Stem],
    preset: dict,
    guards: dict,
    warnings: list,
    intensity: float = 1.0,
    balance_preserve: float = 0.5,
) -> dict:
    """Work out each track's gain so the preset's balance is imposed.

    The anchor is the median stem loudness, not the loudest one: anchoring on
    the median means about half the stems move up and half move down, which
    keeps the total gain applied small and the result closer to the source's
    own balance.  Every gain is then clamped, never silently: a clamp is
    reported as a warning.
    """
    lufs = {s.name: s.metrics.get("lufs", -70.0) for s in stems}
    live = sorted(v for v in lufs.values() if v > -60.0)
    if not live:
        anchor = -20.0
    elif len(live) % 2:
        anchor = float(live[len(live) // 2])
    else:
        # the true median: with an even number of stems the two middle values
        # are averaged.  Indexing len//2 took the upper one, which made the
        # anchor up to a few dB louder than the median -- and every quiet stem
        # then got that much extra boost.
        mid = len(live) // 2
        anchor = (live[mid - 1] + live[mid]) / 2.0

    role_gains = preset["role_gains_db"]
    max_abs = MAX_STAGE_DB
    # Every preset declares vocal_forward_db and nothing was reading it: the
    # style said "vocal 1.5 dB forward" while the planner ignored it.  Lead
    # vocals get the full amount, doubles and harmonies get half -- they should
    # sit behind the lead, not beside it.
    vocal_fwd = float(preset.get("vocal_forward_db", 0.0)) * float(intensity)
    # How much of the source's own balance to keep, 0..1.  Stems exported
    # from a session are usually already balanced by the person who made
    # them: measured on 11 real multitracks (MUSDB18, where the four stems
    # sum to the finished mix), the professional's balance put the lead
    # vocal 5.6 dB above the drums and the accompaniment within a few dB of
    # both -- and imposing the style's absolute balance on top of that moved
    # the accompaniment 14.4 dB away from what the engineer chose.  So the
    # style's opinion and the source's balance are blended instead of one
    # replacing the other.
    k = min(1.0, max(0.0, float(balance_preserve)))
    present = [
        float(role_gains.get(s.role, role_gains.get("other", -10.0))) for s in stems
    ]
    style_centre = float(sorted(present)[len(present) // 2]) if present else 0.0

    gains = {}
    for s in stems:
        base = float(role_gains.get(s.role, role_gains.get("other", -10.0)))
        if s.role == "lead_vocal":
            base += vocal_fwd
        elif s.role == "backing_vocal":
            base += vocal_fwd * 0.5
        style_offset = base - style_centre
        want = anchor + k * (lufs[s.name] - anchor) + (1.0 - k) * style_offset
        g = want - lufs[s.name]
        if abs(g) > max_abs:
            clamped = max(-max_abs, min(max_abs, g))
            warnings.append(
                f"{s.name}: gain clamped to {clamped:+.1f} dB "
                f"(wanted {g:+.1f} dB to hit {want:.1f} LUFS)"
            )
            g = clamped
        gains[s.name] = round(g, 2)
    return gains


def _tonal_eq(stem: Stem, preset: dict, guards: dict) -> list[dict]:
    """One corrective bell at most, driven by the stem's own band balance.

    The preset's tonal tilt describes the *mix*; applying it per stem as well
    would double it.  So this only fixes imbalance that is wrong for the role:
    a boomy vocal, a thin bass.  Capped by ``guards.max_eq_db``.
    """
    role = stem.role
    b = stem.metrics.get("band_energy")
    if not b:
        return []
    limit = float(guards.get("max_eq_db", 6.0))
    bands: list[dict] = []

    low = b.get("low", 0.0) + b.get("low_mid", 0.0)
    if role in VOCAL_ROLES and low > 0.40:
        cut = min(limit, (low - 0.40) * 24.0)
        if cut > 0.5:
            bands.append(
                {"type": "bell", "freq": 220.0, "gain_db": round(-cut, 1), "q": 1.0, "enabled": True}
            )
    if role == "bass" and low < 0.35:
        boost = min(limit * 0.6, (0.35 - low) * 20.0)
        if boost > 0.5:
            bands.append(
                {"type": "bell", "freq": 90.0, "gain_db": round(boost, 1), "q": 0.9, "enabled": True}
            )
    if role in DRUM_ROLES | {"guitar_electric", "synth"} and b.get("presence", 0.0) > 0.45:
        cut = min(limit * 0.7, (b["presence"] - 0.45) * 18.0)
        if cut > 0.5:
            bands.append(
                {"type": "bell", "freq": 3200.0, "gain_db": round(-cut, 1), "q": 1.2, "enabled": True}
            )
    return bands


def _baseline_sum(stems: list[Stem]) -> np.ndarray:
    """Plain sum of the stems, for measuring what the material already is."""
    n = max(s.audio.shape[0] for s in stems)
    acc = np.zeros((n, 2))
    for s in stems:
        a = s.audio
        if a.shape[0] < n:
            a = np.vstack([a, np.zeros((n - a.shape[0], 2))])
        acc += a
    return acc


def _baseline_width(stems: list[Stem]) -> float:
    return dsp.width_ratio(_baseline_sum(stems))


def _baseline_crest(stems: list[Stem]) -> float:
    return dsp.crest_db(_baseline_sum(stems))


def _simplify(times: np.ndarray, gains: np.ndarray, tol: float = 0.2) -> list[list[float]]:
    """Drop points the interpolator would reproduce anyway."""
    out: list[list[float]] = []
    last_t, last_g = None, None
    for t, g in zip(times, gains):
        if last_g is None or abs(g - last_g) >= tol:
            out.append([round(float(t), 2), round(float(g), 2)])
            last_t, last_g = float(t), float(g)
    if not out or out[-1][0] != round(float(times[-1]), 2):
        out.append([round(float(times[-1]), 2), round(float(gains[-1]), 2)])
    return out


def _ride_curve(stem, max_db: float) -> list[list[float]]:
    """Slow gain automation that keeps a sustained source at its own median.

    This is the "vocal ride" a mixer does by hand: the verse comes up, the
    chorus comes down, so the performance sits still in the mix instead of
    disappearing and then shouting.  It is bounded (±max_db), smoothed over a
    second, and stored as a handful of points so the plan stays readable and
    editable.
    """
    if max_db < 0.3:
        return []
    # a 1 s window and 3 s smoothing: rides follow sections, not vibrato or
    # tremolo.  At 0.4 s / 1 s a 7 Hz tremolo produced 681 points and the
    # "ride" was audibly pumping.
    times, db = dsp.short_term_db(stem.audio, stem.sr, 1.0, 0.25)
    if times.size < 3:
        return []

    loudest = float(np.max(db))
    active = db[db > loudest - 40.0]
    if active.size < 3:
        return []
    target = float(np.median(active))

    # deadband: a 1-2 dB swell inside a phrase is the performance, not a
    # balance problem.  Only move when the stem is clearly off its own median.
    gain = np.clip(target - db, -max_db, max_db)
    gain = np.where(np.abs(gain) < 0.75, 0.0, gain)
    # gate: do not ride up into silence, or the first note of the next section
    # arrives pre-boosted
    gain = np.where(db < loudest - 45.0, 0.0, gain)

    win = max(1, int(RIDE_SMOOTH_S / 0.25))
    if win > 1:
        kernel = np.ones(win) / win
        pad = win // 2
        padded = np.pad(gain, (pad, win - pad - 1), mode="edge")
        gain = np.convolve(padded, kernel, mode="valid")[: gain.size]

    points = _simplify(times, gain)
    # hard cap so one pathological track cannot bloat the plan
    if len(points) > 120:
        keep = np.linspace(0, len(points) - 1, 120).astype(int)
        points = [points[i] for i in keep]
    return points



#: tracks where a steady 50/60 Hz tone is *not* treated as hum, because it is
#: far more likely to be the note being played
HUM_EXEMPT = frozenset({"bass", "sub_808", "kick", "drums_full", "other_808"})
HUM_NOTCH_DB = -9.0
#: the schema caps EQ Q at 10, and a 10-Q notch at 50 Hz is ~5 Hz wide, which
#: is plenty for a 1 Hz hum line
HUM_NOTCH_Q = 10.0
#: A side hint is only read as a whole token: "guitar_01_L" asks for the left,
#: "synth_loop_a" asks for nothing (the "l" in "loop" is not a side).  The
#: token may be written in any case; the id is already normalised.
_SIDE_TOKENS = {"l": -1, "left": -1, "r": 1, "right": 1}
#: where a name that commits to a side is placed.  R114: hard left / hard
#: right (was 0.4) -- reported 2026-11-04: "只要看到檔案上面有 L 跟 R，就是要 pan
#: 到極左跟極右".  A track named for a side is half of a stereo pair recorded
#: on two tracks, so it goes all the way, whatever its role and however many
#: tracks share it.  Saved plans keep the pan they were written with.
NAMED_SIDE_PAN = 1.0
#: how far apart two automatic members of one role group are placed.
SPREAD_STEP = 0.2
#: the delimiters a file name may put between words.
_TOKEN_SPLIT = re.compile(r"[\s_.\-]+")


def _side_hint(track_id: str) -> tuple[int, bool]:
    """Which side a track's name asks for: -1 left, +1 right, 0 no answer.

    The second value says the name was ambiguous -- it carried a token for
    both sides.  Picking one of them would move music the file never
    committed to, so such a track is placed automatically instead.
    """
    tokens = {t for t in _TOKEN_SPLIT.split(str(track_id).lower()) if t}
    sides = {_SIDE_TOKENS[t] for t in tokens if t in _SIDE_TOKENS}
    if len(sides) == 1:
        return sides.pop(), False
    return 0, len(sides) > 1


def _spread_duplicate_pans(tracks: list[dict]) -> None:
    """Place same-role tracks: named sides stay put, the rest spread around centre.

    A role's pan is a single number, so two string parts end up on top of each
    other; with twenty stems that turns the middle into mush.  A file name
    that commits to a side ("guitar_01_L") is believed and held on that side;
    the rest of the group is spread symmetrically around the centre, so two
    clones of one part become a pair instead of one louder blob in the middle.
    Centre roles (bass, kick, vocal) stay centred -- spreading those would be
    wrong, not tidy.

    The automatic members of a group are ordered by track id before the slots
    are handed out, so the same tracks always get the same placement whatever
    order they arrive in.  Only that local list is ordered: the caller's list
    is left exactly as it was -- nothing here moves, replaces or reorders a
    track.
    """
    centre_roles = {"kick", "snare", "bass", "sub_808", "lead_vocal", "drums_full"}

    by_role: dict[str, list[dict]] = {}
    for t in tracks:
        by_role.setdefault(t["role"], []).append(t)

    for t in tracks:
        side, _ambiguous = _side_hint(t.get("id", ""))
        if side < 0:
            t["pan"] = -NAMED_SIDE_PAN
            t.setdefault("evidence", []).append("panned hard left from the file name")
        elif side > 0:
            t["pan"] = NAMED_SIDE_PAN
            t.setdefault("evidence", []).append("panned hard right from the file name")

    for role, group in by_role.items():
        if len(group) < 2 or role in centre_roles:
            continue
        automatic = []
        for t in group:
            side, ambiguous = _side_hint(t.get("id", ""))
            if side == 0:
                automatic.append(t)
                if ambiguous:
                    t.setdefault("evidence", []).append(
                        "file name mentions both sides; placed automatically"
                    )
        automatic.sort(key=lambda t: str(t.get("id", "")))
        count = len(automatic)
        if count == 1:
            automatic[0]["pan"] = 0.0
            automatic[0].setdefault("evidence", []).append(
                "centred: the only automatically placed track of this role"
            )
        elif count > 1:
            for i, t in enumerate(automatic):
                pan = round(
                    max(-1.0, min(1.0, (i - (count - 1) / 2.0) * SPREAD_STEP)), 2
                )
                t["pan"] = pan
                t.setdefault("evidence", []).append(
                    f"spread to {pan:+.2f} around the centre because {count} "
                    f"tracks share the {role} role"
                )


def _duplicate_warnings(tracks: list[dict], stems_by_id: dict, warnings: list,
                        group_of: dict | None = None) -> None:
    """Flag stems that are the same performance twice.

    Real sessions ship a DI and an amp, or a doubled take, or the same file
    imported under two names.  Mixing both doubles the part and usually sounds
    like a mistake -- so say so, and let the user mute one (the UI has that).

    ``group_of`` maps a track id to the declared capture group it belongs to.
    Two members of the *same* declared group are the same source by the user's
    own statement, so the shared performance is the point of the group rather
    than a duplicate to mute.  Every other pair -- no group, two groups, or a
    group member against a track outside it -- is still flagged exactly as
    before.
    """
    ids = [t["id"] for t in tracks]
    seen = set()
    for i, a in enumerate(ids):
        ea = stems_by_id.get(a, {}).get("envelope") or []
        if len(ea) < 8:
            continue
        for b in ids[i + 1 :]:
            if (a, b) in seen:
                continue
            eb = stems_by_id.get(b, {}).get("envelope") or []
            if len(eb) != len(ea):
                continue

            va, vb = np.array(ea), np.array(eb)
            # "Steady" means the envelope carries no rhythm at all, not that
            # it is mathematically constant.  Measured: a real duplicate pair
            # sits at std 0.49-0.59 (the envelope is normalised to its median),
            # while sustained tones sit at 0.00-0.06.  With the old 1e-6
            # threshold, a bass and a kick -- both steady -- came out at
            # r=1.000 and the warning called them "the same performance".
            if va.std() < 0.15 or vb.std() < 0.15:
                # steady material has no envelope to compare; fall back to a
                # spectral-shape + centroid test, which only identical parts pass
                da = stems_by_id.get(a, {})
                db = stems_by_id.get(b, {})
                ca = float(da.get("centroid_hz", 0.0))
                cb = float(db.get("centroid_hz", 0.0))
                if not (ca > 0 and cb > 0 and abs(ca - cb) / ca < 0.05):
                    continue
                ba = da.get("band_db") or {}
                bb = db.get("band_db") or {}
                bands = ("low", "low_mid", "mid", "presence", "air")
                if not ba or not bb:
                    continue
                pa = max(float(ba.get(k, -120)) for k in bands)
                pb = max(float(bb.get(k, -120)) for k in bands)
                shape_a = [float(ba.get(k, -120)) - pa for k in bands]
                shape_b = [float(bb.get(k, -120)) - pb for k in bands]
                if max(abs(x - y) for x, y in zip(shape_a, shape_b)) > 3.0:
                    continue
                r = 1.0
            else:
                r = float(np.corrcoef(va, vb)[0, 1])
            # 0.995 is deliberately high: a kick and a snare in a simple pattern
            # already reach 0.986 on a coarse envelope, while a real copy sits
            # at 0.999+.  The warning is a statement, not an instruction, so a
            # stereo pair matching is fine -- the wording says so.
            if r >= 0.995:
                seen.add((a, b))
                if group_of is not None:
                    gid = group_of.get(a)
                    if gid is not None and group_of.get(b) == gid:
                        # The user declared these two as one source recorded
                        # by two microphones.  The shared performance is what
                        # the group is for; the receipt already explains it.
                        continue
                warnings.append(
                    f"{a} and {b} carry the same performance (envelope r={r:.3f}). "
                    "If they are a stereo pair that is expected; if they are two copies "
                    "of one part, mute one."
                )


def _low_end_carve(tracks: list[dict], stems_by_id: dict, guards: dict, warnings: list,
                   grouped_ids: set | None = None) -> None:
    """Make room for the kick in the bass, statically and measurably.

    Kick and bass both live around 50-80 Hz, and when they share a fundamental
    the result is mud that no amount of master EQ fixes.  The usual fix is a
    sidechain, which we cannot do with static parameters -- so this is the
    static approximation: a narrow dip in the bass at the kick's *measured*
    fundamental.

    It is deliberately small (1.5-3.5 dB) and it is written into the plan with
    the reason attached, so it can be seen, argued with and edited.

    A protected member keeps its own EQ: the dip is planned exactly as before
    and the protection throws it away, so the warning has to say "not applied"
    rather than claim a carve nobody will hear.
    """
    by_role = {t["role"]: t for t in tracks}
    kick = by_role.get("kick")
    low = by_role.get("bass") or by_role.get("sub_808")
    if not kick or not low:
        return

    kick_hz = float(stems_by_id.get(kick["id"], {}).get("low_peak_hz", 0.0))
    bass_hz = float(stems_by_id.get(low["id"], {}).get("low_peak_hz", 0.0))
    if kick_hz <= 0.0:
        return

    # compare in semitones, not Hz: 12 Hz apart is a semitone at 700 Hz and
    # four semitones at 46 Hz, and only the musical interval matters here
    same_note = False
    if bass_hz > 0.0 and kick_hz > 0.0:
        semitones = abs(12.0 * float(np.log2(bass_hz / kick_hz)))
        same_note = semitones < 1.5
    depth = 3.5 if same_note else 1.5
    limit = float(guards.get("max_eq_db", 6.0))
    depth = min(depth, limit)

    freq = max(30.0, min(160.0, kick_hz))
    for band in low.get("eq", []):
        if abs(band["freq"] - freq) < 15.0:
            return  # already carved there

    low.setdefault("eq", []).append(
        {"type": "bell", "freq": round(freq, 1), "gain_db": -round(depth, 1), "q": 2.0, "enabled": True}
    )
    low.setdefault("evidence", []).append(
        f"room for the kick: {depth:.1f} dB dip at {freq:.0f} Hz "
        f"({'same fundamental' if same_note else 'kick fundamental'})"
    )
    if low["id"] in (grouped_ids or set()):
        warnings.append(
            f"{low['id']}: {depth:.1f} dB dip at {freq:.0f} Hz not applied "
            "(capture group protection keeps the member's own EQ)"
        )
    else:
        warnings.append(
            f"{low['id']}: {depth:.1f} dB dip at {freq:.0f} Hz to make room for the kick "
            f"({'they share a fundamental' if same_note else 'static sidechain substitute'})"
        )


def _capture_group_facts(stems: list[Stem]) -> list[dict]:
    """What the capture-group validator needs to know about each loaded track.

    The sample rate here is the one the file was *imported* at, not the rate
    the session settled on: two microphones recorded at different rates are
    not the same take, even after the session resampled them onto one rate.
    """
    facts = []
    for s in stems:
        audio = s.audio
        if audio.ndim != 2 or audio.shape[1] < 2:
            dual_mono = True
        else:
            # exactly equal, not nearly: a dual-mono file is one source, a
            # stereo file is two
            dual_mono = bool(np.array_equal(audio[:, 0], audio[:, 1]))
        facts.append(
            {
                "id": analyze.track_id(s.name),
                "role": s.role,
                "sample_rate": int(getattr(s, "resampled_from", None) or s.sr),
                "duration_s": float(s.source_duration or s.duration),
                "dual_mono": dual_mono,
                "silent": bool(s.metrics.get("silent")),
            }
        )
    return facts


def _tempo_for_delay(stems: list[Stem], tracks: list[dict]) -> dict | None:
    """The song's tempo, when it is clear enough to lock a delay to.

    The drum tracks first (the clearest pulse); if they do not give a clear
    answer, everything together.  None when neither does.
    """
    roles = {t["id"]: t["role"] for t in tracks}
    n = max(s.audio.shape[0] for s in stems)
    tried = []
    for source, keep in (("drums", lambda r: ROLE_GROUP.get(r) == "drum"), ("all tracks", lambda r: True)):
        chosen = [s for s in stems if keep(roles.get(analyze.track_id(s.name), "other"))]
        if not chosen:
            continue
        total = np.zeros((n, 2))
        for s in chosen:
            total[: s.audio.shape[0]] += s.audio
        found = dsp.estimate_tempo(total, stems[0].sr)
        del total
        if found is None:
            continue
        found["source"] = source
        tried.append(found)
        if (found["strength"] >= TEMPO_LOCK_MIN_STRENGTH and found["halves_agree"] == 2
                and not found.get("ambiguous")):
            return found
    return None


def _snap_to_beat(aim: float, beat: float) -> float:
    """The beat times the power of two nearest ``aim`` (log scale)."""
    return beat * 2.0 ** round(math.log2(aim / beat))


def _reverb_timing(reverb: dict, tempo: dict | None) -> dict | None:
    """The style's decay and pre-delay locked to the tempo, or None."""
    if tempo is None or not tempo.get("bpm"):
        return None
    beat_s = 60.0 / float(tempo["bpm"])
    aim_decay = float(reverb.get("decay_s", 1.6))
    aim_pre = float(reverb.get("pre_delay_ms", 20.0))
    if aim_decay <= 0.0 or aim_pre <= 0.0:
        return None
    snapped_decay = _snap_to_beat(aim_decay, beat_s)
    snapped_pre = _snap_to_beat(aim_pre, beat_s * 1000.0)
    decay = min(REVERB_TIMING_DECAY_S[1], max(REVERB_TIMING_DECAY_S[0], snapped_decay))
    pre = min(REVERB_TIMING_PRE_MS[1], max(REVERB_TIMING_PRE_MS[0], snapped_pre))
    out = {"law": REVERB_TIMING_LAW, "bpm": round(float(tempo["bpm"]), 2),
           "style_decay_s": round(aim_decay, 3), "style_pre_delay_ms": round(aim_pre, 2),
           "decay_s": round(decay, 3), "pre_delay_ms": round(pre, 2),
           "decay_beats": round(decay / beat_s, 4), "pre_delay_beats": round(pre / 1000.0 / beat_s, 5)}
    clamped = [name for name, a, b in (("decay_s", decay, snapped_decay), ("pre_delay_ms", pre, snapped_pre)) if a != b]
    if clamped:
        # the safety range moved it off the beat grid: say so
        out["clamped"] = clamped
    return out


def _mix_techniques(preset: dict, intensity: float, stems: list[Stem], tracks: list[dict],
                    buses: dict, grouped_ids: set, warnings: list, tempo_of=None) -> dict | None:
    """Write the style's sidechain, parallel and delay moves into the plan.

    Bus blocks are added in place; the delay block is returned (it belongs at
    the top of the plan, next to the reverb).  Every amount scales with the
    intensity, as the style's other moves do.
    """
    tech = dict(preset.get("techniques") or {})
    by_role: dict[str, list[str]] = {}
    for t in tracks:
        by_role.setdefault(t["role"], []).append(t["id"])
    has = lambda group: any(ROLE_GROUP.get(t["role"], "music") == group for t in tracks)

    depth = min(KICK_DUCK_MAX_DB, float(tech.get("kick_duck_db", 0.0)) * float(intensity))
    if depth >= 0.5 and has("bass"):
        duck = None
        if by_role.get("kick"):
            duck = {"key_ids": sorted(by_role["kick"])}
        elif by_role.get("drums_full"):
            # a whole kit: listen to its low end only, where the kick is
            duck = {"key_ids": sorted(by_role["drums_full"]), "key_lowpass_hz": 150.0}
        if duck is not None:
            buses["bass"]["duck"] = {"law": KICK_DUCK_LAW, **duck, "depth_db": round(depth, 2),
                                     "attack_ms": 5.0, "release_ms": 120.0}

    # the vocal's weight and edge (parallel density, saturation, body /
    # presence / air), on the vocal bus whenever there is a vocal
    vc = dict(tech.get("vocal_character") or {})
    k = float(intensity)
    if vc and k > 0.0 and any(t["role"] in analyze.VOCAL_ROLES for t in tracks):
        buses["vocal"]["character"] = {
            "law": VOCAL_CHARACTER_LAW,
            "parallel_blend": round(min(1.0, float(vc.get("parallel_blend", 0.0)) * k), 3),
            "saturation": round(min(0.5, float(vc.get("saturation", 0.0)) * k), 3),
            "body_db": round(float(vc.get("body_db", 0.0)) * k, 2),
            "presence_db": round(float(vc.get("presence_db", 0.0)) * k, 2),
            "air_db": round(float(vc.get("air_db", 0.0)) * k, 2),
        }

    blend = min(1.0, float(tech.get("drum_parallel", 0.0)) * float(intensity))
    if blend >= 0.05 and has("drum"):
        buses["drum"]["parallel"] = {"law": DRUM_PARALLEL_LAW, "blend": round(blend, 3), "ratio": 8.0,
                                     "threshold_offset_db": -10.0, "attack_ms": 2.0, "release_ms": 80.0}

    spec = dict(tech.get("vocal_delay") or {})
    send = float(spec.get("send", 0.0)) * float(intensity)
    singers = [t for t in tracks if t["role"] in analyze.VOCAL_ROLES and t["id"] not in grouped_ids]
    if send < 0.01 or not singers:
        return None
    division = str(spec.get("division", "slap"))
    feedback = float(spec.get("feedback", 0.0))
    tempo = None
    if division in DELAY_DIVISIONS:
        tempo = tempo_of() if tempo_of is not None else _tempo_for_delay(stems, tracks)
        if tempo is None:
            warnings.append(f"vocal delay: no clear tempo, so a {SLAP_MS:.0f} ms slapback instead of a {division} delay")
    if tempo is not None:
        time_ms = 60000.0 / float(tempo["bpm"]) * DELAY_DIVISIONS[division]
        # beyond a second and a half an echo is a separate event, not a delay
        while time_ms > 1500.0:
            time_ms /= 2.0
    else:
        division, time_ms, feedback = "slap", SLAP_MS, min(feedback, SLAP_MAX_FEEDBACK)
    for t in singers:
        amount = send if t["role"] == "lead_vocal" else send * 0.5
        t["sends"] = {**(t.get("sends") or {}), "delay": round(amount, 3)}
    block = {"law": VOCAL_DELAY_LAW, "division": division, "time_ms": round(time_ms, 2),
             "feedback": round(feedback, 3), "hpf_hz": 300.0, "lpf_hz": 5000.0,
             "duck_db": 6.0, "duck_release_ms": 250.0}
    if tempo is not None:
        block["tempo"] = tempo
    return block


def _sync_unmask_targets(summary: dict, tracks: list[dict]) -> None:
    """Make the unmask summary describe the targets that are really in the plan.

    The protection removes the dynamic EQ it finds on a protected member, so a
    summary that still lists that member claims a filter nobody will hear.
    The vacated slot stays empty on purpose: ranking the candidates again
    would hand a dynamic EQ to a track that never had one, and that is a
    change to somebody else's audio that nobody asked for.
    """
    live = {t["id"] for t in tracks if t.get("dynamic_eq") is not None}
    targets = list(summary.get("targets") or [])
    summary["targets"] = [tid for tid in targets if tid in live]
    for tid in targets:
        if tid not in live:
            summary.setdefault("diagnostics", []).append(
                f"{tid}: capture group protection bypassed the vocal unmask target; "
                "no dynamic EQ is applied"
            )


def _plan_decode_policy(stems: list[Stem]) -> str | None:
    """The decoder policy the stems agree on, or ``None`` when it is unknown.

    A stem that was loaded by the production loader knows how it was decoded.
    A stem that was built by hand does not, and the plan must not pretend it
    does: an untagged stem is not evidence of high precision, so a plan built
    from untagged stems records no policy at all and is replayed the legacy
    way.

    Evidence that disagrees is refused rather than guessed.  Two different
    policies in one session mean two different decoders ran over the same
    material, and picking either one would silently mis-decode the other
    stems on replay -- that is refused even when every source is a WAV.

    Partial evidence -- some stems tagged, some not -- is refused too, with
    one narrow exception: when *every* source is a WAV.  Both policies read a
    WAV through the very same parser, so the decoder choice cannot change
    what those files decode to; the plan is then built and records no policy,
    which keeps it replayable the legacy way and makes no claim about
    precision it cannot back up.  The exception is about the file format
    only -- it is not extended to FLAC, AIFF, MP3 or any other format that
    ffmpeg has to decode, and nothing about the audio itself is guessed.
    """
    tagged: list[str] = []
    for stem in stems:
        policy = stem.decode_policy
        if policy is None:
            continue
        if not isinstance(policy, str) or policy not in analyze.DECODE_POLICIES:
            raise PlanError(
                f"stem {stem.name!r} records an unknown decode policy "
                f"{policy!r}; expected one of "
                f"{', '.join(analyze.DECODE_POLICIES)}"
            )
        tagged.append(policy)

    known = set(tagged)
    if not known:
        return None
    if len(known) > 1:
        raise PlanError(
            "the stems were decoded with more than one decoder policy "
            f"({', '.join(sorted(known))}); load them again with one policy"
        )
    if len(tagged) == len(stems):
        return next(iter(known))

    # partial evidence: the untagged stems could have been decoded either way
    if all(stem.name.lower().endswith(".wav") for stem in stems):
        return None
    raise PlanError(
        "only some of the stems record how they were decoded; "
        "load them again with one policy so the plan can be replayed"
    )


def plan_decode_policy(plan: dict) -> str:
    """The decoder policy a saved plan has to be replayed with.

    A plan that does not record one was written before the field existed, so
    it is replayed with the legacy decoder -- the one it was built with.  A
    recorded policy is used exactly as recorded, whatever the current default
    is.
    """
    source = plan.get("source")
    if not isinstance(source, dict) or "decode_policy" not in source:
        return analyze.LEGACY_DECODE_POLICY
    policy = source["decode_policy"]
    if not isinstance(policy, str) or policy not in analyze.DECODE_POLICIES:
        raise PlanError(
            f"source.decode_policy must be one of "
            f"{', '.join(analyze.DECODE_POLICIES)}, not {policy!r}"
        )
    return policy


def build_plan(
    stems: list[Stem],
    preset: dict,
    preset_path: str | None = None,
    intensity: float = 1.0,
    balance_preserve: float | None = None,
    reference: dict | None = None,
    blend: float = 0.7,
    baseline: dict | None = None,
    reverb_match: bool = False,
    trims: dict | None = None,
    pans: dict | None = None,
    target_lufs: float | None = None,
    true_peak_db: float | None = None,
    source_balance_profile: str = "auto",
    capture_groups: list | None = None,
    _source_balance: dict | None = None,
    reverb_amount: float | None = None,
    group_trims: dict | None = None,
) -> dict:
    """Build a MixPlan.

    ``baseline`` optionally carries what the *rendered* mix already measures
    (``tone``, ``width``, ``crest``).  Reference mode corrects toward the
    reference from that baseline, so it must be the post-processing numbers --
    comparing a reference against the pre-processing sum blames the renderer's
    own low-end treatment on the reference and asks for a correction that is
    already being made.  :func:`automix.pipeline.mix` supplies this by
    rendering once before planning against a reference.

    ``reference`` is an analysed reference track (see :mod:`automix.reference`).
    When present, its measured targets are blended with the preset's, weighted
    by ``blend`` (0 = ignore it, 1 = follow it exactly).  The blend matters:
    copying a reference outright usually asks for a spectral correction that
    this material cannot deliver, and half of it is usually what people mean.

    ``capture_groups`` is the user's explicit declaration of which loaded
    tracks are the same physical source recorded by several microphones (see
    :mod:`automix.capture_groups`).  It is never inferred: a group is a fact
    about the session that only the user has.  A declared group's members keep
    their imported time, polarity and waveform relationship before the common
    bus/master, because their automatic per-member processing is bypassed and
    they share one pan.  Nothing else in the plan changes, and the tracks that
    are not in a group keep exactly the numbers they would have had.
    """
    reverb_amount = _manual_reverb_amount(reverb_amount, optional=True)
    if not stems:
        raise PlanError("no stems to plan")
    try:
        balance_preserve = balancemod.legacy_api_balance(balance_preserve)
        decision = _source_balance or balancemod.resolve(stems, reference, source_balance_profile, balance_preserve)
        balancemod.validate(decision, {analyze.track_id(st.name) for st in stems})
    except ValueError as exc:
        raise PlanError(str(exc)) from exc
    balance_preserve = decision["balance_preserve"]
    blend = float(max(0.0, min(1.0, blend)))
    if reference is not None and blend <= 0.0:
        reference = None

    guards = dict(preset.get("guards") or {})
    warnings: list[str] = []

    # analysis is normally done by the caller, but a plan must never be built
    # from missing metrics
    for s in stems:
        if not s.metrics:
            analyze.measure(s)

    # The decoder policy is read from the stems that were actually loaded,
    # before any of them is dropped.  A silent input must not be able to hide
    # the fact that two different decoders were used over the same session.
    decode_policy = _plan_decode_policy(stems)

    # Capture groups are declared, never inferred: which microphones recorded
    # one physical source is a fact about the session that only the user has.
    # The declaration is checked here -- before anything is planned, long
    # before anything is rendered -- so an invalid group is a clear error and
    # never a half-protected mix.

    # The channel facts are computed once and kept: the group validator needs
    # them, and the automatic pan rule below needs to know which loaded stems
    # really carry two different channels.
    group_facts = _capture_group_facts(stems)
    try:
        groups = capmod.normalize(capture_groups, group_facts)
    except capmod.CaptureGroupError as exc:
        raise PlanError(str(exc)) from exc
    grouped_ids = {member for group in groups for member in group["members"]}
    # Which declared group each member belongs to.  Only two members of the
    # *same* group may have their shared-performance warning suppressed.
    group_of = {member: group["id"] for group in groups for member in group["members"]}

    # A silent stem is not a quiet stem.  Pushing it up by the clamp limit just
    # amplifies dither (or literal zeros) and fills the track list with a row
    # that does nothing; dropping it with a note is the honest behaviour.
    excluded_stems = [{"id": analyze.track_id(s.name), "file": s.name, "reason": "silent"}
                      for s in stems if s.metrics.get("silent")]
    dropped = [s.name for s in stems if s.metrics.get("silent")]
    stems = [s for s in stems if not s.metrics.get("silent")]
    if dropped:
        warnings.append(
            f"dropped {len(dropped)} silent stem(s): {', '.join(dropped)}"
        )
    if not stems:
        raise PlanError("every stem is silent; nothing to mix")

    # A stem that is far shorter than the session is almost always a truncated
    # export, not an arrangement choice: it gets padded with silence and then
    # simply stops being heard.  Say so instead of mixing half a part.
    if len(stems) > 1:
        lengths = {s.name: float(getattr(s, "source_duration", None) or s.duration) for s in stems}
        longest = max(lengths.values())
        for name, length in lengths.items():
            if longest > 10.0 and length < longest * 0.5:
                warnings.append(
                    f"{name} is {length:.1f} s but the session is {longest:.1f} s; "
                    "check whether the export was cut short"
                )

    gains = _gain_stage(stems, preset, guards, warnings, intensity, balance_preserve)
    fitted = decision["proof"].get("relative_gain_db", {}) if decision["origin"] == "reference_fit" else {}
    fit_expected = {}
    for st in stems:
        tid = analyze.track_id(st.name)
        if tid not in fitted:
            continue
        wanted = gains[st.name] + fitted[tid]
        fit_expected[tid] = wanted
        gains[st.name] = round(max(-MAX_STAGE_DB, min(MAX_STAGE_DB, wanted)), 2)
        if abs(wanted-gains[st.name]) > .011:
            warnings.append(f"{tid}: reference balance gain target {wanted:+.2f} dB not reached; stage clamp {gains[st.name]:+.2f} dB")


    # Reverb matching (opt-in, deliberately timid).
    #
    # The idea was: measure the reference's post-transient tail and scale our
    # sends to match.  The measurement works, but it is *content dependent* --
    # a dry reference full of short hits reads 0.06 while our mix of a single
    # long-decay kick reads 0.41, so comparing the two numbers directly says
    # "you are far too wet" when nothing is wrong.  There is no way to separate
    # "the reference is dry" from "the reference has less sustain" using the
    # finished mix alone.
    #
    # So it is off by default, capped at +-20%, and only fires when both tails
    # are large enough to mean something and the disagreement is clear.  When
    # in doubt it changes nothing and says so.
    reverb_scale = 1.0
    reverb_note = None
    if reference is not None and baseline and reverb_match:
        ref_tail = float(reference.get("tail_ratio") or 0.0)
        our_tail = float(baseline.get("tail") or 0.0)
        if ref_tail < 0.1 or our_tail < 0.1:
            reverb_note = (
                "reverb match skipped: not enough transient evidence "
                f"(reference tail {ref_tail:.2f}, ours {our_tail:.2f})"
            )
        else:
            raw = ref_tail / our_tail
            if 0.8 <= raw <= 1.25:
                reverb_note = (
                    f"reverb match: already close (reference tail {ref_tail:.2f} "
                    f"vs ours {our_tail:.2f})"
                )
            else:
                target = max(0.85, min(1.2, raw))
                reverb_scale = (1.0 - blend) * 1.0 + blend * target
                reverb_note = (
                    f"reverb match (experimental): {'raised' if reverb_scale > 1 else 'lowered'} "
                    f"to ×{reverb_scale:.2f} — reference tail {ref_tail:.2f} vs ours {our_tail:.2f}"
                )

    # Spectral baseline: what this material already sounds like once the style
    # balance is applied.  The reference's tilt is a *correction* against this,
    # so a mix that is already bright does not get brightened twice.
    base_tone = None
    if reference is not None:
        if baseline and baseline.get("tone"):
            base_tone = list(baseline["tone"])
        else:
            n = max(s.audio.shape[0] for s in stems)
            acc = np.zeros((n, 2))
            for st in stems:
                a = st.audio
                if a.shape[0] < n:
                    a = np.vstack([a, np.zeros((n - a.shape[0], 2))])
                acc += a * dsp.db_to_lin(gains[st.name])
            base_tone = refmod.spectrum_db(acc, stems[0].sr)
    comp = dict(preset.get("compression") or {})
    sat = dict(preset.get("saturation") or {})
    reverb = dict(preset.get("reverb") or {})
    max_sat = float(guards.get("max_saturation", 1.0))
    max_width = float(guards.get("max_width", 2.0))
    min_crest = float(guards.get("min_crest_db", 0.0))
    max_gr = float(guards.get("max_gr_db", 12.0))
    width_style = float(preset.get("width", 1.0))

    tracks = []
    skipped_dense: list[str] = []
    # The facts about each imported source, kept apart from the decisions the
    # planner builds on top of them: a protected member keeps the facts and
    # loses the decisions.
    source_facts_by_id: dict[str, list[str]] = {}
    for s in stems:
        group = role_group(s.role)
        crest = float(s.metrics.get("crest_db", 12.0))

        # compression: scale the group amount by intensity, then respect the
        # two guards (predicted gain reduction, and don't squash an already
        # dense source further)
        amount = float(comp.get(group, comp.get(s.role, 0.0))) * intensity
        if crest < min_crest:
            if amount > 0.0:
                skipped_dense.append(s.name)
            amount = 0.0
        est_gr = amount * 10.0
        if est_gr > max_gr:
            capped = max_gr / 10.0
            warnings.append(
                f"{s.name}: compression amount {amount:.2f} capped to {capped:.2f} "
                f"(estimated {est_gr:.1f} dB GR > guard {max_gr:.1f} dB)"
            )
            amount = capped

        comp_block = _comp_params(group, amount) if amount > 0 else None
        comp_dial = None
        if comp_block and group == "drum":
            comp_block, comp_dial = _open_drum_track_comp(comp_block)

        saturation = float(sat.get(group, sat.get(s.role, 0.0))) * intensity
        if saturation > max_sat:
            warnings.append(f"{s.name}: saturation {saturation:.2f} capped to {max_sat:.2f}")
            saturation = max_sat

        wet = (
            float(reverb.get("amount", 0.0))
            * REVERB_WEIGHT.get(s.role, 0.3)
            * intensity
            * reverb_scale
        )
        width = width_style if group == "music" else 1.0
        if s.role in ("bass", "sub_808", "kick", "lead_vocal", "drums_full"):
            width = 1.0
        width = min(width, max_width)

        hpf = min(500.0, max(20.0, HPF_HZ.get(s.role, 50.0)))

        # A stem that is mostly empty is usually a mistaken export (a marker
        # track, a muted take).  Silence itself is handled elsewhere; this is
        # about the *proportion*, which nothing was looking at.
        silent_frac = float(s.metrics.get("silence_ratio", 0.0) or 0.0)
        if silent_frac > 0.7 and float(s.source_duration or 0.0) > 10.0:
            warnings.append(
                f"{s.name}: {silent_frac * 100:.0f}% of the file is silence "
                f"(only {s.metrics.get('active_s', 0.0):.1f} s of audio) -- "
                "check that this is the stem you meant to export"
            )

        clip_ratio = float(s.metrics.get("clipped_ratio", 0.0))
        if clip_ratio > 0.005:
            warnings.append(
                f"{s.name}: {clip_ratio * 100:.1f}% of samples are already at full "
                f"scale — the source file is clipped, that damage cannot be undone here"
            )
        if float(s.metrics.get("correlation", 0.0)) < -0.3:
            warnings.append(
                f"{s.name}: channels are largely out of phase "
                f"(correlation {s.metrics['correlation']:+.2f}); mono playback will lose it"
            )

        ride = []
        if s.role in RIDE_ROLES and intensity > 0.0:
            ride = _ride_curve(s, RIDE_MAX_DB * intensity)

        tid = analyze.track_id(s.name)
        confidence = 0.9 if s.role_source == "keyword" else 0.55
        # What is true of the imported audio: the role it was given and what
        # was measured on it.  A protected member keeps these facts; the
        # automatic decisions built on top of them are replaced by the
        # protection's own, so the facts are kept apart on purpose.
        source_facts = [
            f"role={s.role} from {s.role_source} (confidence {confidence:.2f})",
            f"stem lufs {s.metrics.get('lufs', 0):.1f}, "
            f"crest {crest:.1f}, centroid {s.metrics.get('centroid_hz', 0):.0f} Hz",
        ]
        track = {
            "id": tid,
            "role": s.role,
            "gain_db": gains[s.name],
            "pan": PAN.get(s.role, 0.0),
            "hpf_hz": hpf,
            "eq": _tonal_eq(s, preset, guards),
            "saturation": round(saturation, 3),
            "sends": {"plate": round(wet, 3)} if wet > 0 else {},
            "width": round(width, 3),
            "confidence": confidence,
            "evidence": source_facts + [
                f"style gain for {s.role} {preset['role_gains_db'].get(s.role, -10.0):+.1f} dB",
            ],
            "source_file": s.name,
        }
        source_facts_by_id[tid] = source_facts
        if track["id"] in fitted:
            track["evidence"].append(
                f"same-song rough mix relative correction {fitted[track['id']]:+.3f} dB included once in gain_db; "
                f"common fit gain {decision['proof']['common_gain_db']:+.3f} dB is unapplied; proof is before processing"
            )
        if comp_block:
            track["comp"] = comp_block
        if comp_dial:
            track["comp_style_dial"] = comp_dial
        hum = float(s.metrics.get("hum_hz", 0.0))
        if hum and s.role not in HUM_EXEMPT:
            # A sustained bass note at 50 Hz is indistinguishable from hum on
            # this material, so low-end roles are exempt; on anything else a
            # steady 50/60 Hz tone is almost certainly mains.
            track.setdefault("eq", []).append(
                {
                    "type": "bell",
                    "freq": hum,
                    "gain_db": HUM_NOTCH_DB,
                    "q": HUM_NOTCH_Q,
                    "enabled": True,
                    # a surgical repair, not tone shaping: it is allowed past
                    # max_eq_db, and the schema keeps the reason visible
                    "purpose": "hum_notch",
                }
            )
            track["evidence"].append(f"hum notch at {hum:.0f} Hz ({HUM_NOTCH_DB:.0f} dB, Q {HUM_NOTCH_Q:.0f})")
            if tid in grouped_ids:
                # The notch is planned exactly as before and the protection
                # throws it away.  Saying "notched" would be a claim about
                # audio that will not be in the mix.
                warnings.append(
                    f"{s.name}: {hum:.0f} Hz hum found, not notched "
                    "(capture group protection keeps the member's own EQ)"
                )
            else:
                warnings.append(f"{s.name}: {hum:.0f} Hz hum notched ({HUM_NOTCH_DB:.0f} dB)")

        if s.role in VOCAL_ROLES and float(s.metrics.get("sibilance", 0.0)) >= DEESS_THRESHOLD:
            amount = min(DEESS_MAX_AMOUNT, 0.3 + (s.metrics["sibilance"] - DEESS_THRESHOLD) * 10.0)
            amount = round(amount * intensity, 3)
            if amount > 0.05:
                track["deess"] = {"amount": amount, "freq_hz": 6000.0}
                track["evidence"].append(
                    f"de-esser {amount:.2f} above {DEESS_THRESHOLD:.2f} "
                    f"(sibilance index {s.metrics['sibilance']:.3f})"
                )

        if ride:
            # [[seconds, dB], ...] -- piecewise linear, applied by the renderer
            track["ride"] = ride
            span = max(g for _, g in ride) - min(g for _, g in ride)
            track["evidence"].append(
                f"ride ±{RIDE_MAX_DB * intensity:.1f} dB over {len(ride)} points "
                f"(spread {span:.1f} dB)"
            )
        tracks.append(track)

    # Sections: measured on the gained sum, so the boundaries follow the song
    # as it will actually be heard rather than one stem's arrangement.
    sec_list = []
    section_info = {"factor": 1.0, "reference_db": None}
    if intensity > 0.0:
        acc = np.zeros((max(s.audio.shape[0] for s in stems), 2))
        for st in stems:
            a = st.audio
            if a.shape[0] < acc.shape[0]:
                a = np.vstack([a, np.zeros((acc.shape[0] - a.shape[0], 2), dtype=a.dtype)])
            acc += a * dsp.db_to_lin(gains[st.name])
        sec_list = sections.detect(acc, stems[0].sr)
        del acc

        # How hard should the sections contrast?  Default is the preset's own
        # small move; a reference rescales it by how much *it* contrasts between
        # its own sections.  A flat reference (no sections) switches the whole
        # thing off -- inventing a chorus lift for a song that never lifts is
        # exactly the kind of "helpful" that sounds wrong.
        sec_factor = 1.0
        if reference is not None:
            ref_db = float(reference.get("section_contrast_db") or 0.0)
            ref_factor = min(1.8, max(0.4, ref_db / 5.0)) if ref_db > 0.5 else 0.0
            sec_factor = (1.0 - blend) * 1.0 + blend * ref_factor
            if sec_factor < 0.05:
                warnings.append(
                    "reference has no section contrast, so section variation is off"
                )

        # Match the *contour*: if the reference's choruses sit 4 dB above its
        # verses and ours only 2 dB, lift ours.  This is section-aligned in
        # spirit (by label, not by time), which is the part that survives a
        # different arrangement order.
        label_gain = {}
        if reference is not None and len(reference.get("sections") or []) > 1:
            ref_secs = reference["sections"]
            ref_mean = float(np.mean([x["level_db"] for x in ref_secs]))
            our_mean = float(np.mean([x["level_db"] for x in sec_list]))
            ref_by = {}
            our_by = {}
            for x in ref_secs:
                ref_by.setdefault(x["label"], []).append(x["level_db"] - ref_mean)
            for x in sec_list:
                our_by.setdefault(x["label"], []).append(x["level_db"] - our_mean)
            for lab in set(ref_by) & set(our_by):
                delta = float(np.mean(ref_by[lab])) - float(np.mean(our_by[lab]))
                delta = max(-2.0, min(2.0, delta)) * blend * intensity
                label_gain[lab] = round(delta, 2)

        for sec in sec_list:
            lab = sec["label"]
            sec["gain_db"] = label_gain.get(lab, 0.0)
            width = 1.0 + (SECTION_WIDTH.get(lab, 1.0) - 1.0) * intensity * sec_factor
            send = 1.0 + (SECTION_SEND.get(lab, 1.0) - 1.0) * intensity * sec_factor
            air = SECTION_AIR.get(lab, 0.0) * intensity * sec_factor
            sec["width_scale"] = round(width, 3)
            sec["send_scale"] = round(send, 3)
            sec["air_db"] = round(air, 2)
        section_info = {
            "factor": round(sec_factor, 3),
            "reference_db": reference.get("section_contrast_db") if reference else None,
        }
        if len(sec_list) > 1:
            chorus = next((x for x in sec_list if x["label"] == "chorus"), None)
            verse = next((x for x in sec_list if x["label"] == "verse"), None)
            detail = []
            if chorus:
                detail.append(f"width ×{chorus['width_scale']:.3f} on chorus")
            if verse:
                detail.append(f"send ×{verse['send_scale']:.2f} on verse")
            if abs(sec_factor - 1.0) > 1e-6:
                detail.append(f"reference contrast ×{sec_factor:.2f}")
            warnings.append(
                "sections: "
                + ", ".join(f"{x['label']} {x['start_s']:.0f}-{x['end_s']:.0f}s" for x in sec_list)
                + (f" ({'; '.join(detail)})" if detail else "")
            )
        else:
            sec_list = []

    # Manual trims are applied last, on top of every automatic decision, and are
    # recorded as evidence so the receipt explains itself even when the user was
    # the one who moved the fader.
    trims = trims or {}
    for t in tracks:
        delta = float(trims.get(t["id"], 0.0))
        if abs(delta) < 1e-6:
            continue
        before = float(t["gain_db"])
        t["gain_db"] = round(max(-MAX_STAGE_DB, min(MAX_STAGE_DB, before + delta)), 2)
        applied = t["gain_db"] - before
        t["trim_db"] = round(delta, 2)
        note = f"manual trim {delta:+.1f} dB ({before:+.1f} -> {t['gain_db']:+.1f})"
        if abs(applied - delta) > 0.01:
            note += f", clamped to {applied:+.1f} dB by the stage limit"
        t.setdefault("evidence", []).append(note)

    # Loudness normalisation can invert the relationship a producer recorded on
    # purpose: a quiet backing vocal gets a big boost and ends up in front of
    # the lead.  Keep the hierarchy explicit -- doubles sit behind the lead.
    lead = next((t for t in tracks if t["role"] == "lead_vocal"), None)
    source_levels = {t["id"]: float(s.metrics["lufs"]) for s, t in zip(stems, tracks)}
    if lead is not None:
        for t in tracks:
            if t["role"] != "backing_vocal":
                continue
            if t["id"] in grouped_ids:
                # a protected member keeps its imported relative level, and the
                # hidden hierarchy compensation is exactly the per-mic level
                # change the group exists to prevent
                continue
            ceiling = source_levels[lead["id"]] + float(lead["gain_db"]) - 4.0
            planned_level = source_levels[t["id"]] + float(t["gain_db"])
            if planned_level > ceiling:
                before = float(t["gain_db"])
                wanted = ceiling - source_levels[t["id"]]
                t["gain_db"] = round(max(-MAX_STAGE_DB, wanted), 2)
                actual_gap = (source_levels[lead["id"]] + float(lead["gain_db"]) -
                              source_levels[t["id"]] - float(t["gain_db"]))
                detail = (
                    f"gain wanted {wanted:+.2f} dB, applied {t['gain_db']:+.2f} dB "
                    f"(was {before:+.2f}); planned lead-minus-backing gap {actual_gap:+.2f} dB "
                    "(hierarchy target >= +4.00 dB)"
                )
                if wanted < -MAX_STAGE_DB:
                    status = "hierarchy target NOT reached: stage clamp; " + detail
                else:
                    status = "below the lead vocal after adjustment; " + detail
                t["evidence"].append(
                    status
                )
                warnings.append(
                    f"{t['id']}: backing vocal {status}"
                )

    for t in tracks:
        if t["id"] in fit_expected:
            if t["id"] in grouped_ids:
                # The protection replaces this track's gain with the user's
                # trim on a 0 dB base, so the fit is superseded rather than
                # missed: saying "target not reached" would blame a guard that
                # was never in the way.
                continue
            wanted = fit_expected[t["id"]] + float((trims or {}).get(t["id"], 0.0))
            if abs(t["gain_db"]-wanted) > .011:
                warnings.append(
                    f"{t['id']}: reference balance target plus manual trim {wanted:+.2f} dB not reached; "
                    f"planned {t['gain_db']:+.2f} dB after stage/manual/hierarchy guards. "
                    "The raw-source fit does not prove the rendered balance."
                )

    stems_by_id = {t["id"]: s.metrics for s, t in zip(stems, tracks)}
    # The helper only orders the automatic members it is about to place, so
    # the plan's own track order -- and the pairing with the stems -- is
    # untouched.
    _spread_duplicate_pans(tracks)

    # A declared "balanced stems" source at full balance preservation is the
    # user saying: this material is already a balanced stereo print.  A role
    # offset, a spread slot or a side read out of a file name is an L/R
    # offset on top of that print, and nobody asked for it -- so the
    # automatic placement is dropped for the stems that really carry two
    # different channels.  The user's own pan and the declared capture group
    # are applied after this and still have the last word.
    #
    # Only the loaded channels are read here: a stem that is mono, or a stereo
    # file whose two channels are identical, is one source and not a stereo
    # print, so it keeps the placement it always had.
    if decision["requested"] == "balanced_stems" and float(balance_preserve) == 1.0:
        differing_channels = {f["id"] for f in group_facts if not f["dual_mono"]}
        for t in tracks:
            if t["id"] not in differing_channels:
                continue
            before = float(t.get("pan", 0.0))
            if before == 0.0:
                continue
            t["pan"] = 0.0
            t.setdefault("evidence", []).append(
                f"automatic placement suppressed (pan {before:+.2f}): the source is a "
                "declared balanced stereo at 100% balance preservation, so no extra "
                "L/R offset is added"
            )

    # An explicit pan from the user is the final authority: it is applied after
    # every automatic placement decision (role defaults, duplicate-role
    # spreading, L/R file names) so that nothing downstream can overwrite it.
    # The value is clamped and rounded exactly as the direct API always did.
    for t in tracks:
        if pans and t["id"] in pans:
            before = float(t.get("pan", 0.0))
            t["pan"] = round(max(-1.0, min(1.0, float(pans[t["id"]]))), 2)
            t.setdefault("evidence", []).append(
                f"manual pan {t['pan']:+.2f} (was {before:+.2f})"
            )

    _duplicate_warnings(tracks, stems_by_id, warnings, group_of)
    _low_end_carve(tracks, stems_by_id, guards, warnings, grouped_ids)
    unmask_summary = unmask.build(stems, tracks, intensity, guards)

    # Fader gain alone cannot preserve balance: EQ, compression and saturation
    # change the level differently for each role.  Freeze the source window
    # and source level in the plan; the renderer executes this bounded matching rule
    # just as it executes the compressor's declared make-up rule.  Old plans
    # without this block keep their original signal path.
    for s, t in zip(stems, tracks):
        start, end = analyze.active_span(s.audio, s.sr)
        t["level_match"] = {
            "mode": "active_lufs_blend",
            "source_lufs": float(s.metrics["lufs"]),
            "amount": 1.0,
            "max_gain_db": 6.0,
            "start_frame": int(start),
            "end_frame": int(end),
        }
        t.setdefault("evidence", []).append(
            f"post-track loudness target = source LUFS + current gain, source window "
            f"{start}:{end}, correction capped at +/-6 dB (pre-bus)"
        )

    # The protection is applied last, after every automatic decision and after
    # the automatic fan, but before the receipt is written -- so the numbers the
    # renderer reads are the protected ones.  A declared member loses its
    # automatic ride, high-pass, EQ, de-esser, compressor, saturation, width
    # and loudness matching, its gain becomes the user's explicit trim on top
    # of 0 dB, and it takes the group's one common pan.  The tracks that are
    # not in a group are not touched, and nothing here reaches the common
    # bus/master.
    receipt = None
    if groups:
        try:
            receipt, group_warnings = capmod.apply_protection(
                tracks, groups, pans=pans, max_gain_db=MAX_STAGE_DB,
                source_facts=source_facts_by_id,
            )
        except capmod.CaptureGroupError as exc:
            raise PlanError(str(exc)) from exc
        warnings.extend(group_warnings)
        _sync_unmask_targets(unmask_summary, tracks)

    if skipped_dense:
        warnings.append(
            f"compression skipped on {len(skipped_dense)} already-dense stem(s) "
            f"(crest below {min_crest:.1f} dB): {', '.join(skipped_dense)}"
        )

    # buses: the style's group amounts again, at bus level
    def _bus_level_db(group: str) -> float | None:
        """Approximate the summed level of a bus from its tracks' own numbers.

        Without this, a bus's compressor threshold was a formula in the
        abstract: on the demo session it sat above the music bus's actual
        level, so "bus glue" measured -0.00 dB of gain reduction on music and
        vocal -- present in the plan, absent in the audio.
        """
        power = 0.0
        for t in tracks:
            if ROLE_GROUP.get(t["role"], "music") != group:
                continue
            metrics = stems_by_id.get(t["id"], {})
            lufs = float(metrics.get("lufs", -70.0))
            if lufs <= -70.0:
                continue
            power += 10.0 ** ((lufs + float(t.get("gain_db", 0.0))) / 10.0)
        if power <= 0.0:
            return None
        return 10.0 * float(np.log10(power))

    def _bus(group: str) -> dict:
        amt = float(comp.get(group, 0.0)) * intensity * 0.75  # buses compress less
        # Width is a per-track decision here (see the track loop); letting the
        # music bus apply the style's width a second time squared it.  Measured
        # impact was small (~3% of side/mid) but two owners for one number is
        # how the *next* bug happens.
        bus = {"width": 1.0}
        if amt > 0:
            spec = _comp_params("bus", amt)
            if group == "drum":
                was = {"law": DRUM_OPEN_LAW, "ratio": spec["ratio"], "attack_ms": spec["attack_ms"],
                       "threshold_db": spec["threshold_db"]}
                spec["ratio"] = min(float(spec["ratio"]), DRUM_BUS_MAX_RATIO)
                spec["attack_ms"] = max(float(spec["attack_ms"]), DRUM_BUS_MIN_ATTACK_MS)
            level = _bus_level_db(group)
            if level is not None:
                # The estimate is only a hint.  The renderer places the threshold
                # relative to the bus's *measured* level: an RMS sum of per-track
                # loudness is a poor guess of a real bus's peaks, and the first
                # attempt left bus glue at -0.04 dB of gain reduction.
                offset = round(-1.0 - 6.0 * amt, 1)
                if group == "drum":
                    # R117: glue, not a lid (DRUM_BUS_THRESHOLD_OFFSET_DB)
                    was["threshold_offset_db"] = offset
                    was["threshold_db"] = min(COMP_THRESHOLD_DB_RANGE[1],
                                              max(COMP_THRESHOLD_DB_RANGE[0], round(level + offset, 1)))
                    offset = DRUM_BUS_THRESHOLD_OFFSET_DB
                spec["threshold_offset_db"] = offset
                # The estimate is bounded to the schema's legal range so a
                # loud bus (two protected guitar mics at a legal +18 dB trim
                # estimate above 0 dB) cannot make the saved plan invalid.
                # Only the estimate is bounded: the offset is what the
                # renderer uses, so the audio does not move.
                low, high = COMP_THRESHOLD_DB_RANGE
                spec["threshold_db"] = min(high, max(low, round(level + offset, 1)))
            bus["comp"] = spec
            if group == "drum":
                bus["comp_style_dial"] = was
        if group in ("music",):
            bus["saturation"] = round(min(float(sat.get("music", 0.0)) * 0.5, max_sat), 3)
        return bus

    buses = {g: _bus(g) for g in ("drum", "bass", "music", "vocal", "fx")}
    if any(t["role"] in VOCAL_ROLES for t in tracks):
        style_db = (VOCAL_VAR_PRO_MEDIAN_DB
                    + float(preset.get("vocal_forward_db", 0.0)) * float(intensity)
                    + float((preset.get("techniques") or {}).get("vocal_lift_db", 0.0)) * float(intensity))
        balance = {
            "law": VOCAL_BALANCE_LAW,
            "target_db": round(style_db, 2),
            "max_boost_db": VOCAL_BALANCE_MAX_BOOST_DB,
            "max_cut_db": VOCAL_BALANCE_MAX_CUT_DB,
            "style_db": round(style_db, 2),
            "meter": BALANCE_METER_LAW,
            "sends": BALANCE_SEND_LAW,
        }
        # The source's own vocal balance only counts when somebody vouched
        # for it (proven stems, a declared profile, an explicit amount).  An
        # unproven session is raw recording levels, and keeping half of a
        # quietly recorded vocal is how every vocal came out too quiet.
        k_vocal = 0.0 if decision["origin"] == "default" else float(balance_preserve)
        if k_vocal > 0.0:
            source_db = _source_vocal_balance(stems, fitted)
            if source_db is not None:
                balance["source_db"] = round(source_db, 2)
                balance["target_db"] = round(
                    min(6.0, max(-12.0, k_vocal * source_db + (1.0 - k_vocal) * style_db)), 2
                )
        buses["vocal"]["balance"] = balance
    # The band's own balance: drums and bass against the rest of the band,
    # where they play (R114, reported 2026-11-04: "鼓組和Bass都會太大").  Only for
    # an unproven session: stems somebody vouched for keep their balance.
    if decision["origin"] == "default" and float(intensity) > 0.0:
        tech = dict(preset.get("techniques") or {})
        for group, median, key in (("drum", DRUM_VAR_PRO_MEDIAN_DB, "drum_forward_db"),
                                   ("bass", BASS_VAR_PRO_MEDIAN_DB, "bass_forward_db")):
            if median is not None and any(ROLE_GROUP.get(t["role"], "music") == group for t in tracks):
                target = median + float(tech.get(key, 0.0)) * float(intensity)
                buses[group]["balance"] = {"law": PART_BALANCE_LAW, "target_db": round(target, 2),
                                           "max_boost_db": PART_BALANCE_MAX_DB, "max_cut_db": PART_BALANCE_MAX_CUT_DB,
                                           "meter": BALANCE_METER_LAW, "sends": BALANCE_SEND_LAW}
    # one tempo estimate for the delay and the reverb timing
    tempo_memo: dict = {}

    def tempo_of() -> dict | None:
        if "tempo" not in tempo_memo:
            tempo_memo["tempo"] = _tempo_for_delay(stems, tracks)
        return tempo_memo["tempo"]

    delay_block = _mix_techniques(preset, intensity, stems, tracks, buses, grouped_ids, warnings, tempo_of)
    reverb_timing = None
    if (preset.get("techniques") or {}).get("reverb_tempo_lock"):
        reverb_timing = _reverb_timing(reverb, tempo_of())
    # the preset's tonal tilt is a *master* curve: it describes the finished
    # mix's balance, so it is written into the master EQ rather than spread
    # across the tracks (which would apply it several times over).
    tilt = dict(preset.get("tonal_tilt_db") or {})
    max_eq = float(guards.get("max_eq_db", 6.0))

    corrective = None
    tone_indices = []
    post_style_baseline = bool(baseline and baseline.get("tone"))
    if reference is not None and base_tone is not None:
        tone_indices = refmod.common_support(reference["tone"], base_tone,
                                             reference_sr=reference.get("sample_rate"),
                                             base_sr=stems[0].sr)
        corrective = refmod.corrective_tilt(reference["tone"], base_tone, indices=tone_indices)
        for k in tilt:
            if post_style_baseline:
                # The baseline already contains the style EQ.  A correction
                # against it must be added to that EQ, not replace it.
                tilt[k] = float(tilt.get(k, 0.0)) + blend * corrective.get(k, 0.0)
            else:
                tilt[k] = (1.0 - blend) * float(tilt.get(k, 0.0)) + blend * corrective.get(k, 0.0)
        if len(tone_indices) < 2:
            warnings.append("reference tone: insufficient common supported bands; no relative tone correction")

    master_eq = []
    for key, kind, freq, q in (
        ("low", "low_shelf", 120.0, 0.7),
        ("low_mid", "bell", 300.0, 0.9),
        ("mid", "bell", 1000.0, 0.8),
        ("presence", "bell", 3500.0, 0.9),
        ("air", "high_shelf", 9000.0, 0.7),
    ):
        g = float(tilt.get(key, 0.0)) * intensity
        if abs(g) < 0.05:
            continue
        if abs(g) > max_eq:
            warnings.append(f"master tilt {key} {g:+.1f} dB capped to {max_eq:+.1f} dB")
            g = max_eq if g > 0 else -max_eq
        master_eq.append(
            {"type": kind, "freq": freq, "gain_db": round(g, 2), "q": q, "enabled": True}
        )

    for st in stems:
        src = getattr(st, "resampled_from", None)
        if src:
            warnings.append(
                f"{st.name}: {src} Hz resampled to {st.sr} Hz so it lines up with the session"
            )

    target_override = target_lufs
    true_peak_override = true_peak_db

    # An explicit target overrides the style's own.  Streaming platforms
    # normalise to roughly -14 LUFS, so "make it a streaming master" is a
    # routine request that should not require editing a preset.
    if target_override is not None:
        t = float(target_override)
        if not (-24.0 <= t <= -6.0):
            raise PlanError(f"target_lufs {t} is outside the supported -24..-6 range")
        target_lufs = t
    if true_peak_override is not None:
        tp = float(true_peak_override)
        if not (-6.0 <= tp <= 0.0):
            raise PlanError(f"true_peak {tp} is outside the supported -6..0 range")
        target_tp = tp
    else:
        target_tp = float(preset["true_peak_db"])
    if target_override is None:
        target_lufs = float(preset["target_lufs"])
    mono_below = float(preset.get("mono_below_hz", 0.0))
    bus_amt = float(comp.get("bus", 0.0)) * intensity

    if reference is not None:
        if target_override is not None:
            warnings.append(
                f"explicit target {target_override:+.1f} LUFS overrides the reference's "
                f"{float(reference['lufs']):+.1f} LUFS"
            )
        else:
            target_lufs = (1.0 - blend) * target_lufs + blend * float(reference["lufs"])
        # The true-peak ceiling belongs to the style/platform, not to the
        # reference.  Taking min() with the reference's own peak looked
        # prudent but is wrong in both directions: a quiet reference (-20 dBTP)
        # then acts as a ceiling and pins the mix ~4 dB under its loudness
        # target, while a hot one gains nothing because the preset is already
        # the stricter of the two.
        if float(reference["true_peak_db"]) > target_tp:
            warnings.append(
                f"reference true peak {reference['true_peak_db']:+.1f} dBTP is above the "
                f"{target_tp:+.1f} dBTP style ceiling; keeping the ceiling"
            )

        # width: match the reference's side/mid ratio, bounded by the guard
        # width is recorded as an absolute side/mid target, which the renderer
        # then matches by measurement -- a "multiplier" here would be a guess
        base_width = float(baseline["width"]) if baseline and "width" in baseline else _baseline_width(stems)
        width_target = (1.0 - blend) * base_width + blend * float(reference["width"])


        # dynamics: if the reference is denser than this material can be at the
        # style's settings, ask the master bus for more compression.  This is a
        # heuristic, and it says so: crest is an outcome, not a control.
        base_crest = (
            float(baseline["crest"]) if baseline and "crest" in baseline else _baseline_crest(stems)
        )
        if base_crest is not None and base_crest > 0:
            deficit = base_crest - float(reference["crest_db"])
            if deficit > 0:
                extra = min(0.6, blend * deficit / 14.0)
                bus_amt = min(0.8, bus_amt + extra)
                warnings.append(
                    f"reference is {deficit:.1f} dB denser; master compression raised "
                    f"to {bus_amt:.2f} to approach it"
                )

        # low end: a reference whose sub band is already mono is a strong hint
        if reference.get("low_correlation", 0.0) > 0.9:
            mono_below = max(mono_below, 120.0)

    master = {
        "eq": master_eq,
        "target_lufs": round(target_lufs, 2),
        "true_peak_db": round(target_tp, 2),
        # Only the ceiling is real: the true-peak limiter uses a fixed ~2 ms
        # lookahead and a fast release so the intersample guarantee holds.
        # "lookahead_ms" and "release_ms" used to be written here and read by
        # nothing -- another field that looked like a control and was not.
        # R114: the release law (a fast limiter flutters and sounds clipped)
        # R115: never more than 3 dB off the loudest 5% to reach the number
        "limiter": {"ceiling_db": round(target_tp, 2), "law": "tp_release_v1",
                    "max_gr_p95_db": 3.0, "max_backoff_db": 1.5},
        # R114: a wider top end from the side signal only (mono unchanged)
        **({"side_air_db": round(float((preset.get("techniques") or {}).get("master_side_air_db", 0.0))
                                * float(intensity), 2)}
           if float((preset.get("techniques") or {}).get("master_side_air_db", 0.0)) * float(intensity) > 0.05
           else {}),
        "mono_below_hz": mono_below,
        "width": 1.0,
        "width_target": round(width_target, 3) if reference is not None else None,
        "width_limit": float(guards.get("max_width", 2.0)),
    }
    if bus_amt > 0:
        master["comp"] = _comp_params("bus", bus_amt)
    buses["master"] = master

    sr = stems[0].sr
    duration = max(s.duration for s in stems)

    # R118: the user's group faders, on top of every automatic decision
    present = {ROLE_GROUP.get(t["role"], "music") for t in tracks}
    for name, value in _user_trims(group_trims).items():
        if name == "fx" or name in present:
            buses[name]["user_trim_db"] = value
        else:
            warnings.append(f"group fader {name} {value:+.1f} dB ignored: the session has no {name} track")

    plan = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "utuvo-automix 0.1.0",
        "source": {
            "sample_rate": sr,
            "frames": max(len(s.audio) for s in stems),
            "duration_s": round(duration, 3),
            "excluded_stems": excluded_stems,
            "stems": [
                {
                    "id": t["id"],
                    "file": s.name,
                    "role": s.role,
                    "confidence": t["confidence"],
                    "sample_rate": s.sr,
                    "channels": int(s.audio.shape[1]),
                    "duration_s": round(s.duration, 3),
                }
                for s, t in zip(stems, tracks)
            ],
        },
        "style": {
            "id": preset["id"],
            "version": int(preset.get("version", 1)),
            "intensity": round(float(intensity), 3),
            "balance_preserve": float(balance_preserve),
            "source_balance": json.loads(json.dumps(decision)),
            "preset_file": os.path.basename(preset_path) if preset_path else None,
            "name": preset.get("name"),
            "blend": round(blend, 3) if reference is not None else 0.0,
            # carried so the receipt can compare the result against what the
            # style was aiming for, without the reader having to open the preset
            "crest_target_db": preset.get("crest_target_db"),
            "mono_below_hz": preset.get("mono_below_hz"),
        },
        "tracks": tracks,
        "buses": buses,
        "unmask": unmask_summary,
        "sections": sec_list,
        "section_contrast": section_info,
        # The reverb's shape, as the renderer will actually build it.  Without
        # this the plan could not be replayed: the renderer had no way to know
        # the style's decay, pre-delay and high cut.
        "reverb": {
            "amount": round(float(reverb.get("amount", 0.0)), 3),
            "decay_s": round(float(reverb.get("decay_s", 1.6)), 3),
            "pre_delay_ms": round(float(reverb.get("pre_delay_ms", 20.0)), 2),
            "high_cut_hz": round(float(reverb.get("high_cut_hz", 8000.0)), 1),
            "scale": round(reverb_scale, 3),
            # R114: band-dependent decay, early reflections, no low end in
            # the return (dsp.make_ir_plate_v2)
            "ir_law": "plate_v2",
        },
        "warnings": warnings,
        # R117: saturation follows each passage's own level, oversampled
        # (dsp.SATURATION_LAWS); plans without it saturate as before
        "saturation_law": SATURATION_LAW,
        "compressor_law": COMPRESSOR_LAW,
    }
    if delay_block is not None:
        plan["delay"] = delay_block
    duck_depth = min(REVERB_DUCK_MAX_DB,
                     float((preset.get("techniques") or {}).get("vocal_reverb_duck_db", 0.0)) * float(intensity))
    sings_into_reverb = any(t["role"] in analyze.VOCAL_ROLES and t["id"] not in grouped_ids
                            and float((t.get("sends") or {}).get("plate", 0.0)) > 0.0 for t in tracks)
    if duck_depth >= 0.5 and sings_into_reverb:
        tempo = tempo_of()
        if tempo is not None:
            release = min(max(30000.0 / float(tempo["bpm"]), REVERB_DUCK_RELEASE_MS[0]), REVERB_DUCK_RELEASE_MS[1])
        else:
            release = REVERB_DUCK_DEFAULT_RELEASE_MS
        # makeup = depth: as wet as before while the voice sings (the owner:
        # "vocal a bit dry", "the image must be big"), the tail blooms by the
        # depth in the gaps -- the space grows without clouding the words
        plan["reverb"]["duck"] = {"law": REVERB_DUCK_LAW, "depth_db": round(duck_depth, 2),
                                  "makeup_db": round(duck_depth, 2),
                                  "attack_ms": 20.0, "release_ms": round(release, 1)}
    if reverb_timing is not None:
        # the renderer reads decay_s / pre_delay_ms; timing is the receipt
        plan["reverb"]["decay_s"] = reverb_timing["decay_s"]
        plan["reverb"]["pre_delay_ms"] = reverb_timing["pre_delay_ms"]
        plan["reverb"]["timing"] = reverb_timing
    if decode_policy is not None:
        # Only recorded when it is known.  A plan without it was built from
        # material whose origin nobody can vouch for, and it is replayed the
        # legacy way -- claiming float32 there would be a claim, not a fact.
        plan["source"]["decode_policy"] = decode_policy
    if reverb_amount is not None:
        plan["reverb"]["manual_amount"] = reverb_amount
    if receipt is not None:
        # the receipt is optional metadata: it says what was protected and
        # where the protection stops.  The renderer never reads it -- the
        # numbers it carries are already in the tracks.
        plan["capture_groups"] = receipt
    if preset.get("intent") is not None:
        plan["style"]["intent"] = json.loads(json.dumps(preset["intent"]))
    if reverb_note:
        warnings.append(reverb_note)

    if reference is not None:
        block = refmod.to_plan_block(reference, blend)
        if base_tone is not None:
            block["tone_comparison"] = {
                "method": "rms_centered_common_third_octave_db",
                "indices": tone_indices,
                "support_floor_db": 30.0,
                "centers_hz": [refmod.THIRD_OCTAVE_CENTERS[i] for i in tone_indices],
                "baseline_tone": base_tone,
                "baseline_kind": "post_style_render" if post_style_baseline else "raw_gain_sum_approximation",
                "before_db": refmod.tone_distance(reference["tone"], base_tone, tone_indices),
            }
        if corrective is not None:
            block["notes"] += (
                " · corrective tilt "
                + ", ".join(f"{k} {v:+.1f}" for k, v in corrective.items())
            )
        plan["style"]["reference"] = block

    return plan


def _receipt_source_ids(plan: dict, bridge_legacy: bool = False) -> set[str]:
    """Complete source graph, independent of the receipt being checked.

    Legacy builders omitted dropped sources; only their existing unambiguous
    dropped-silent warning supplies that metadata. No waveform/filesystem read.
    """
    source = plan["source"]
    survivors = source.get("stems")
    if not isinstance(survivors, list):
        raise PlanError("source.stems must describe the analyzed source graph")
    if "excluded_stems" in source:
        excluded = source["excluded_stems"]
        if not isinstance(excluded, list):
            raise PlanError("source.excluded_stems must be a list")
    else:
        excluded = []
        if bridge_legacy:
            matches = [re.fullmatch(r"dropped ([1-9][0-9]*) silent stem\(s\): (.+)", warning, re.DOTALL)
                       for warning in (plan.get("warnings") or []) if isinstance(warning, str)]
            matches = [match for match in matches if match is not None]
            if len(matches) > 1:
                raise PlanError("legacy silent-source evidence is ambiguous; remove style.source_balance receipt or re-plan")
            if matches:
                count, names = int(matches[0][1]), matches[0][2]
                files = [names] if count == 1 else names.split(", ")
                if len(files) != count or any(not name for name in files) or len(set(files)) != count:
                    raise PlanError("legacy silent-source count/split is ambiguous; remove style.source_balance receipt or re-plan")
                excluded = [{"id": analyze.track_id(name), "file": name, "reason": "silent"} for name in files]
    ids, files = set(), set()
    for record, dropped in [(record, False) for record in survivors] + [(record, True) for record in excluded]:
        if (not isinstance(record, dict) or not isinstance(record.get("id"), str)
                or not isinstance(record.get("file"), str) or not record["id"] or not record["file"]
                or record["id"] != analyze.track_id(record["file"])):
            raise PlanError("source graph requires matching file/track_id records")
        if dropped and (set(record) != {"id", "file", "reason"} or record["reason"] != "silent"):
            raise PlanError("excluded source graph requires only id/file/reason=silent")
        if record["id"] in ids or record["file"] in files:
            raise PlanError("source graph IDs/files must be unique and survivors/excluded disjoint")
        ids.add(record["id"]); files.add(record["file"])
    return ids


#: the canonical plan schema's file name.  It is shipped with the App bundle,
#: so a packaged App checks plans exactly like the source tree does.
SCHEMA_FILENAME = "mix_plan.schema.json"


def default_schema_path() -> str:
    """Where the canonical schema lives.

    In the source tree it sits at the repository root, next to ``app/``; in
    the packaged App it is bundled as data and lands in
    ``sys._MEIPASS/schemas``.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return os.path.join(meipass, "schemas", SCHEMA_FILENAME)
    package_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(os.path.dirname(package_dir))
    return os.path.join(repo_dir, "schemas", SCHEMA_FILENAME)


def _assert_finite_numbers(value, path: str = "$") -> None:
    """Refuse NaN or Infinity anywhere in a plan.

    JSON has no NaN and no Infinity, but Python's parser accepts both
    (``json.loads('{"gain_db": NaN}')``) and a plan built in memory can
    carry one.  JSON Schema does not catch it: to jsonschema a NaN *is* a
    number, and every range check on it passes silently.  So this guard is
    separate from the schema and runs on its own.

    Integers are finite by definition and are never converted to float: an
    integer with thousands of digits is a number, not an overflow.
    """
    if isinstance(value, bool) or isinstance(value, numbers.Integral):
        return
    if isinstance(value, numbers.Number):
        try:
            number = float(value)
        except (OverflowError, ValueError):
            return  # too large to be a float, but finite as a number
        if not math.isfinite(number):
            raise PlanError(
                f"plan has a non-finite number at {path}: {value!r} "
                "(JSON has no NaN and no Infinity)"
            )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_numbers(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_numbers(item, f"{path}[{index}]")


def _load_validator_packages():
    """Import the validator stack, or say exactly which part is missing.

    A missing dependency is a broken installation, not a bad plan: the
    message says so, so a user's valid plan is never refused because of it
    and never waved through either.
    """
    try:
        import jsonschema
    except ImportError as exc:
        raise PlanError(
            "the plan validator is not installed: jsonschema is required to "
            f"check a plan against {SCHEMA_FILENAME} ({exc})"
        ) from exc
    try:
        import jsonschema_specifications  # noqa: F401  (the metaschemas)
    except ImportError as exc:
        raise PlanError(
            "the plan validator is not installed: jsonschema_specifications "
            f"(the JSON Schema metaschemas) is required ({exc})"
        ) from exc
    try:
        from referencing import Registry, Resource
        from referencing.exceptions import Unresolvable
        from referencing.jsonschema import DRAFT202012
    except ImportError as exc:
        raise PlanError(
            "the plan validator is not installed: referencing is required to "
            f"resolve the schema's own references ({exc})"
        ) from exc
    return jsonschema, Registry, Resource, DRAFT202012, Unresolvable


def _read_schema(schema_path: str) -> dict:
    """Read a schema file, or say why it cannot be used."""
    try:
        with open(schema_path, encoding="utf-8") as fh:
            raw = fh.read()
    except UnicodeDecodeError as exc:
        raise PlanError(
            f"the plan schema is not UTF-8 text ({schema_path}): {exc}"
        ) from exc
    except OSError as exc:
        raise PlanError(f"the plan schema cannot be read ({schema_path}): {exc}") from exc
    try:
        schema = json.loads(raw)
    except ValueError as exc:
        raise PlanError(f"the plan schema is not valid JSON ({schema_path}): {exc}") from exc
    if not isinstance(schema, dict):
        raise PlanError(f"the plan schema must be a JSON object ({schema_path})")
    return schema


def _compile_schema(jsonschema, Registry, Resource, DRAFT202012, schema, schema_path):
    """Turn a schema into a validator, or say why it cannot be used.

    The registry has no retrieval function, so a reference that is not inside
    the schema cannot be resolved -- and nothing goes out to the network to
    resolve it.  A schema that *mentions* a URL as data is not a reference
    and is not touched by this at all.
    """
    try:
        validator_cls = jsonschema.validators.validator_for(
            schema, default=jsonschema.validators.Draft202012Validator
        )
        validator_cls.check_schema(schema)
    except jsonschema.exceptions.SchemaError as exc:
        raise PlanError(
            f"the plan schema is not a valid JSON Schema ({schema_path}): {exc.message}"
        ) from exc
    except Exception as exc:
        raise PlanError(
            f"the plan schema could not be prepared ({schema_path}): {exc}"
        ) from exc
    try:
        resource = Resource.from_contents(schema, default_specification=DRAFT202012)
    except Exception as exc:
        raise PlanError(
            f"the plan schema could not be prepared ({schema_path}): {exc}"
        ) from exc
    # A registry with no retrieval function: a reference that is not inside
    # the schema is unresolvable, and the validator says so instead of going
    # out to the network.
    registry = Registry()
    uri = schema.get("$id")
    if isinstance(uri, str) and uri:
        registry = registry.with_resource(uri=uri, resource=resource)
    try:
        return validator_cls(schema, registry=registry)
    except Exception as exc:
        raise PlanError(
            f"the plan schema could not be prepared ({schema_path}): {exc}"
        ) from exc


def _check_against_schema(plan: dict, schema_path: str) -> None:
    """Check a plan against a schema with the real JSON Schema validator.

    A plan that does not satisfy the schema raises jsonschema's own
    ``ValidationError``: its message says what is wrong and its json_path
    says where.  Anything else -- a validator that cannot run, a reference
    that cannot be resolved -- is an installation problem and is reported as
    a ``PlanError`` that says so.
    """
    jsonschema, Registry, Resource, DRAFT202012, Unresolvable = _load_validator_packages()
    schema = _read_schema(schema_path)
    validator = _compile_schema(jsonschema, Registry, Resource, DRAFT202012, schema, schema_path)
    try:
        validator.validate(plan)
    except jsonschema.ValidationError:
        raise
    except Unresolvable as exc:
        raise PlanError(
            f"the plan schema ({schema_path}) needs a reference that cannot be "
            f"resolved offline: {exc}. The validator resolves references from "
            "the schema itself and never fetches one over the network."
        ) from exc
    except Exception as exc:
        raise PlanError(
            f"the plan could not be checked against {schema_path}: {exc}"
        ) from exc


def _validate_stereo_pair_receipt(group: dict, gid: str) -> None:
    """A pair receipt has to describe the whole pair.

    A receipt that names one member of a pair, names the same member twice,
    or protects a different pair than it declares claims a stereo image it
    does not have.  That is a false claim about the audio, so it is refused
    instead of being repaired into something that sounds right.
    """
    trim = group.get("trim_db")
    if isinstance(trim, bool) or not isinstance(trim, (int, float)) or not np.isfinite(trim):
        raise PlanError(f"capture group {gid!r} stereo pair needs a finite 'trim_db'")
    if not -12.0 <= float(trim) <= 12.0:
        raise PlanError(
            f"capture group {gid!r} stereo pair 'trim_db' must be between -12 and 12 dB"
        )
    pan = group.get("pan")
    if isinstance(pan, bool) or not isinstance(pan, (int, float)) or not np.isfinite(pan):
        raise PlanError(f"capture group {gid!r} stereo pair needs a finite 'pan'")
    if not -1.0 <= float(pan) <= 1.0:
        raise PlanError(f"capture group {gid!r} stereo pair 'pan' must be between -1 and 1")
    members = group.get("members")
    if not isinstance(members, list) or len(members) != 2 or len(set(members)) != 2:
        raise PlanError(
            f"capture group {gid!r} stereo pair must name exactly two different members"
        )
    protected = group.get("protected_tracks")
    if not isinstance(protected, list) or len(protected) != 2 or len(set(protected)) != 2:
        raise PlanError(
            f"capture group {gid!r} stereo pair must protect exactly two different tracks"
        )
    if list(protected) != list(members):
        raise PlanError(
            f"capture group {gid!r} stereo pair protects {', '.join(str(t) for t in protected)} "
            f"but declares {', '.join(str(t) for t in members)}; a pair protects the same "
            "left and right members, in the same order"
        )


def validate(plan: dict, schema_path: str | None = None) -> None:
    """Check a plan against the shipped schema and the engine's invariants.

    The schema is mandatory.  A missing validator dependency, an unreadable,
    corrupt or invalid schema is a hard stop with a message that names the
    installation problem -- a valid plan is never blamed for it, and it is
    never waved through either.

    The schema runs first, so a plan whose containers are the wrong shape is
    refused before the engine's own checks can trip over them.
    """
    if not isinstance(plan, dict):
        raise PlanError(f"plan must be a JSON object, not {type(plan).__name__}")
    _assert_finite_numbers(plan)
    if schema_path is None:
        schema_path = default_schema_path()
    _check_against_schema(plan, schema_path)
    try:
        _engine_invariants(plan)
    except PlanError:
        raise
    except (TypeError, KeyError, AttributeError, IndexError) as exc:
        # The schema cannot describe every field the engine reads, so the
        # engine's own reads are guarded too: a plan with a container of the
        # wrong shape is refused, never a traceback.
        raise PlanError(f"plan structure is invalid: {exc!r}") from exc


def _engine_invariants(plan: dict) -> None:
    """The invariants the engine itself relies on (the plan is schema-valid)."""
    try:
        rev = dict(plan.get("reverb") or {})
    except (TypeError, ValueError):
        # The legacy schema did not constrain this container. Only inspect
        # new manual settings here; rendering retains its original dict() rules.
        rev = {}
    if "manual_amount" in rev:
        _manual_reverb_amount(rev["manual_amount"])

    for key in ("schema_version", "source", "style", "tracks", "buses"):
        if key not in plan:
            raise PlanError(f"plan missing required key {key!r}")
    for name, bus in (plan["buses"] or {}).items():
        # only the renderer's vocal, drum and bass steps read it; anywhere
        # else it would be a setting that silently does nothing
        if name not in ("vocal", "drum", "bass") and isinstance(bus, dict) and "balance" in bus:
            raise PlanError(f"buses.{name}.balance: only the vocal, drum and bass buses have a balance target")
        if isinstance(bus, dict) and isinstance(bus.get("balance"), dict):
            wanted = "vocal_var_v1" if name == "vocal" else "part_var_v1"
            if bus["balance"].get("law") != wanted:
                raise PlanError(f"buses.{name}.balance: law must be {wanted}")
        # the same for the other bus-only moves: each has one bus that reads it
        for key, owner in (("duck", "bass"), ("parallel", "drum"), ("character", "vocal")):
            if name != owner and isinstance(bus, dict) and key in bus:
                raise PlanError(f"buses.{name}.{key}: only the {owner} bus has one")
    if not plan["tracks"]:
        raise PlanError("plan has no tracks")
    # Each move must have something to act on; otherwise it is a setting that
    # silently does nothing (the renderer only runs buses that have tracks).
    roles = {t.get("id"): t.get("role") for t in plan["tracks"] if isinstance(t, dict)}
    groups = {ROLE_GROUP.get(r, "music") for r in roles.values()}
    for name in ("drum", "bass"):
        if ((plan["buses"] or {}).get(name) or {}).get("balance") is not None and name not in groups:
            raise PlanError(f"buses.{name}.balance: the plan has no {name} track")
    vocal_bus = (plan["buses"] or {}).get("vocal") or {}
    if (vocal_bus.get("balance") is not None or vocal_bus.get("character") is not None) and "vocal" not in groups:
        raise PlanError("buses.vocal: balance / character need a vocal track")
    buses_in = plan["buses"] or {}
    duck = (buses_in.get("bass") or {}).get("duck")
    if duck is not None:
        keys = duck.get("key_ids") or []
        missing = [k for k in keys if k not in roles]
        if not keys or missing:
            raise PlanError(f"buses.bass.duck: key tracks not in the plan: {missing or 'none given'}")
        wrong = [k for k in keys if roles[k] not in ("kick", "drums_full")]
        if wrong:
            raise PlanError(f"buses.bass.duck: a key must be a kick or a whole kit, not {wrong}")
        if any(roles[k] == "drums_full" for k in keys) and not duck.get("key_lowpass_hz"):
            raise PlanError("buses.bass.duck: a whole-kit key needs key_lowpass_hz (its kick is in the low end)")
        if "bass" not in groups:
            raise PlanError("buses.bass.duck: the plan has no bass track to duck")
    if (buses_in.get("drum") or {}).get("parallel") is not None and "drum" not in groups:
        raise PlanError("buses.drum.parallel: the plan has no drum track")
    # the renderer's own threshold: a send it would skip is no send
    sending = any(float(((t.get("sends") or {}).get("delay") or 0.0)) > SEND_MIN
                  for t in plan["tracks"] if isinstance(t, dict))
    if plan.get("delay") is None and sending:
        raise PlanError("tracks send to a delay but the plan has no delay block")
    if plan.get("delay") is not None and not sending:
        raise PlanError("the plan has a delay block but no track sends to it; remove the block or give a track sends.delay")
    # The schema says which decoder policies exist; this says it again, so a
    # plan checked against some other, more permissive schema still cannot
    # smuggle an unknown decoder past the engine.
    source = plan["source"]
    if isinstance(source, dict) and "decode_policy" in source:
        policy = source["decode_policy"]
        if not isinstance(policy, str) or policy not in analyze.DECODE_POLICIES:
            raise PlanError(
                "source.decode_policy must be one of "
                f"{', '.join(analyze.DECODE_POLICIES)}, not {policy!r}"
            )
    # an unknown ratio law would otherwise be rendered some guessed way
    comps = [t.get("comp") for t in plan["tracks"]]
    comps += [b.get("comp") for b in plan["buses"].values() if isinstance(b, dict)]
    for comp in comps:
        if isinstance(comp, dict) and "ratio_law" in comp:
            law = comp["ratio_law"]
            if not isinstance(law, str) or law not in COMP_RATIO_LAWS:
                raise PlanError(
                    f"comp.ratio_law must be one of {', '.join(COMP_RATIO_LAWS)}, not {law!r}"
                )
    if plan["style"].get("intent") is not None:
        _validate_style_intent(plan["style"]["intent"])
    decision = plan["style"].get("source_balance")
    if decision is not None:
        evidence = decision.get("proof") if isinstance(decision, dict) else None
        is_fit = isinstance(evidence, dict) and evidence.get("method") in fitmod.SUPPORTED_METHODS
        known_inputs = _receipt_source_ids(plan, bridge_legacy=is_fit)
        try:
            balancemod.validate(decision, known_inputs,
                                int(plan["source"]["sample_rate"]), plan["source"].get("frames"))
        except ValueError as exc:
            remedy = "; remove style.source_balance receipt or re-plan with complete source inputs" if is_fit else ""
            raise PlanError(str(exc) + remedy) from exc
        if decision["balance_preserve"] != plan["style"].get("balance_preserve"):
            raise PlanError("source_balance disagrees with style.balance_preserve")
    if decision is None and "excluded_stems" in plan["source"]:
        _receipt_source_ids(plan)
    reference = (plan.get('style') or {}).get('reference') or {}
    comparison = reference.get('tone_comparison')
    if comparison is not None:
        if not isinstance(comparison, dict) or comparison.get('method') != 'rms_centered_common_third_octave_db':
            raise PlanError('unsupported reference tone comparison method')
        indices = comparison.get('indices')
        if (not isinstance(indices, list) or any(type(i) is not int or not 0 <= i < 31 for i in indices)
                or indices != sorted(set(indices))):
            raise PlanError('reference tone comparison indices must be ordered unique bands0..30')
        for tone in (reference.get('tone'), comparison.get('baseline_tone')):
            if not isinstance(tone, list) or len(tone) != 31 or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in tone):
                raise PlanError('reference tone comparison requires31 finite bands')
        if comparison.get('centers_hz') != [refmod.THIRD_OCTAVE_CENTERS[i] for i in indices]:
            raise PlanError('reference tone comparison centers disagree with fixed indices')
        before = comparison.get('before_db')
        if before is not None and (not isinstance(before, (int, float)) or not np.isfinite(before) or before < 0):
            raise PlanError('reference before distance must be finite/nonnegative or null')

    if sum(t.get("dynamic_eq") is not None for t in plan["tracks"]) > 2:
        raise PlanError("dynamic vocal unmask supports at most2 targets")
    known_ids = {t["id"] for t in plan["tracks"] if t["role"] == "lead_vocal"}
    for track in plan["tracks"]:
        if track.get("dynamic_eq") is not None:
            try:
                unmask.validate(track["dynamic_eq"], track["role"], int(plan["source"]["sample_rate"]),
                                float(plan["source"]["duration_s"]), known_ids)
            except (ValueError, TypeError, KeyError) as exc:
                raise PlanError(f"invalid dynamic_eq: {exc}") from exc
        match = track.get("level_match")
        if match is None:
            continue
        if not isinstance(match, dict) or match.get("mode") != "active_lufs_blend":
            raise PlanError("unsupported track level_match mode")
        if set(match) != {"mode", "source_lufs", "amount", "max_gain_db", "start_frame", "end_frame"}:
            raise PlanError("level_match has missing or unsupported settings")
        for key, low, high in (("source_lufs", -100.0, 30.0),
                               ("amount", 0.0, 1.0), ("max_gain_db", 0.0, 6.0)):
            value = match.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not low <= value <= high:
                raise PlanError(f"level_match.{key} must be finite and within {low}..{high}")
        start, end = match.get("start_frame"), match.get("end_frame")
        if type(start) is not int or type(end) is not int or start < 0 or end <= start:
            raise PlanError("level_match source window must have integer 0 <= start_frame < end_frame")
        frames = plan["source"].get("frames")
        if frames is not None and (type(frames) is not int or frames <= 0 or end > frames):
            raise PlanError("level_match source window exceeds or has invalid source frames")

    # The capture-group receipt is optional metadata -- a plan without it is a
    # normal plan -- but a receipt that claims protection for a track the mix
    # does not contain is a false claim.  Refuse it instead of shipping it.
    receipt = plan.get("capture_groups")
    if receipt is not None:
        if not isinstance(receipt, dict) or receipt.get("mode") != "declared_capture_group":
            raise PlanError("unsupported capture group receipt mode")
        groups = receipt.get("groups")
        if not isinstance(groups, list):
            raise PlanError("capture_groups.groups must be a list")
        known_ids = {t["id"] for t in plan["tracks"]}
        for group in groups:
            if not isinstance(group, dict):
                raise PlanError("every capture group receipt must be an object")
            gid = group.get("id")
            if not isinstance(gid, str) or not gid.strip():
                raise PlanError("every capture group receipt needs an id")
            mode = group.get("mode")
            if "mode" in group and group["mode"] != "stereo_pair":
                raise PlanError(
                    f"capture group {gid!r} has an unsupported mode {group['mode']!r}"
                )
            if "trim_db" in group and "mode" not in group:
                raise PlanError(
                    f"capture group {gid!r} has a shared 'trim_db' but no mode; a shared "
                    "trim only means something on a stereo pair"
                )
            protected = group.get("protected_tracks")
            if not isinstance(protected, list) or not protected:
                raise PlanError(f"capture group {gid!r} receipt names no protected track")
            missing = [tid for tid in protected if tid not in known_ids]
            if missing:
                raise PlanError(
                    f"capture group {gid!r} claims protection for track(s) "
                    f"{', '.join(missing)} that are not in this mix"
                )
            if mode == "stereo_pair":
                _validate_stereo_pair_receipt(group, gid)

    # Two fields describe the same ceiling.  The renderer reads the limiter's,
    # so editing only `true_peak_db` used to look like it worked while the audio
    # stayed at the limiter's value.  Refuse the ambiguity instead of guessing.
    master = (plan.get("buses") or {}).get("master") or {}
    limiter = master.get("limiter") or {}
    if limiter.get("ceiling_db") is not None and master.get("true_peak_db") is not None:
        if abs(float(limiter["ceiling_db"]) - float(master["true_peak_db"])) > 0.01:
            raise PlanError(
                "master.true_peak_db "
                f"({master['true_peak_db']}) and master.limiter.ceiling_db "
                f"({limiter['ceiling_db']}) disagree; the renderer uses the "
                "limiter's value -- set both, or edit the limiter block"
            )


def write_plan(plan: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
