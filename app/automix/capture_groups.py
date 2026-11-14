"""Declared capture groups: the user says which microphones recorded one source.

A capture group is an explicit statement -- "these imported tracks are the same
physical source, captured by several microphones".  A protected group keeps the
IMPORTED audio's relative time, polarity and waveform relationship before the
common bus/master, by bypassing the automatic per-member processing and giving
every member one common pan.

What this module deliberately does *not* do:

* it never infers groups from similar names or roles -- the user declares them;
* it never aligns time, flips polarity, removes bleed or "fixes" a recorded
  phase problem -- the imported relationship is kept exactly as imported;
* it never claims a better mix, and it never touches the common bus/master.

It only (a) validates an explicit declaration and (b) computes the protected
numbers for the members.  Everything else -- where in the plan the protection
lands, what the renderer does with it -- belongs to the caller.

The canonical declaration is a list of objects::

    [{"id": "room pair", "members": ["mic_l", "mic_r"], "pan": -0.2}]

``pan`` is optional and defaults to 0.  Member names are canonical track IDs
(what ``analyze.track_id`` produces); a file path is never read as a member.

A second declaration shape is the explicit stereo pair::

    [{"id": "pair", "members": ["mic_l", "mic_r"], "mode": "stereo_pair",
      "pan": 0, "trim_db": 0}]

There the two members are the declared left and the declared right channel:
the order of ``members`` IS the channel identity, and it is never sorted or
inferred from the file names.  A pair keeps the two channels apart instead
of putting both members on one common pan, and it takes a shared trim that
moves both members together.  A declaration without ``mode`` keeps exactly
the legacy behaviour and exactly the legacy fields.
"""

from __future__ import annotations

import math
import numbers

#: a group id is a human label, not an identifier we look up anywhere
GROUP_ID_MAX_CHARS = 64
#: how many groups one mix may declare
MAX_GROUPS = 32
#: a group needs at least two members to mean anything, and 128 is generous
MIN_MEMBERS = 2
MAX_MEMBERS = 128
PAN_MIN = -1.0
PAN_MAX = 1.0

#: the only new declaration mode: an explicit ordered stereo pair
MODE_STEREO_PAIR = "stereo_pair"
#: a stereo pair holds exactly two members -- the declared left, the declared right
PAIR_MEMBER_COUNT = 2
#: the shared pair trim range, in dB
TRIM_MIN = -12.0
TRIM_MAX = 12.0
#: The renderer's pan law is constant-power (45 degrees at centre) with a
#: sqrt(2) endpoint normalisation, so a member sitting on an endpoint is
#: +3.0103 dB hot.  Both members of a pair take this compensation, so a
#: centred pair sums back to the imported level.  The value is kept at full
#: precision: rounding it to -3.01 moves the pair by ~3e-5.
PAIR_BASE_GAIN_DB = -3.010299956639812

#: the fields a protected member carries instead of its automatic values
PROTECTED_VALUES = {
    "gain_db": 0.0,
    "hpf_hz": 20.0,
    "eq": [],
    "saturation": 0.0,
    "width": 1.0,
    "sends": {"plate": 0.0},
}

#: the fields a protected member must not carry at all
#: (R132: a ride's band-following receipt goes with the ride)
PROTECTED_ABSENT = ("ride", "ride_follow", "comp", "deess", "dynamic_eq", "level_match")

#: the boundary, written where a reader (and the user) will actually see it
BOUNDARY = (
    "Protection applies to the members before the common bus/master: no per-member "
    "ride, high-pass, EQ, de-esser, compression, saturation, width or loudness "
    "matching, and one common pan. It does not align time, flip polarity or remove "
    "bleed, and it does not change the common bus/master processing, so the "
    "finished mix is not guaranteed to be identical to the imported waveforms."
)

#: the same boundary, for a declared stereo pair
STEREO_PAIR_BOUNDARY = (
    "A stereo pair applies to its members before the common bus/master: no per-member "
    "ride, high-pass, EQ, de-esser, compression, saturation, width or loudness "
    "matching, and no shared pan position -- the declared left and the declared right "
    "member keep their own pan and their own level. The shared position moves the pair "
    "toward one side and narrows the image -- at the extremes both members land on the "
    "same side -- so it is a stereo image positioning control, not a balance control. "
    "The shared trim moves both members together. The pair is not aligned in time, not "
    "polarity inverted, not MS-decoded and not de-bleeded, and the common bus/master "
    "still changes the finished mix."
)

#: the same paragraph, scoped to the same-source groups, for a plan that also
#: holds a stereo pair: the pair has no common pan, and the sentence must not
#: be read as if it had one
BOUNDARY_WITH_GROUP = (
    "A same-source capture group applies to its members before the common bus/master: "
    "no per-member ride, high-pass, EQ, de-esser, compression, saturation, width or "
    "loudness matching, and one common pan. It does not align time, flip polarity or "
    "remove bleed, and it does not change the common bus/master processing, so the "
    "finished mix is not guaranteed to be identical to the imported waveforms."
)


class CaptureGroupError(ValueError):
    """A capture group declaration that cannot be honoured.

    Raised *before* anything is rendered, so an invalid declaration never
    leaves a half-protected mix behind.
    """


def _finite(value, label: str) -> float:
    """A number we can vouch for.

    A value that cannot even be turned into a float -- an integer too large
    to represent -- is not a number this tool can use, and it must not escape
    as an OverflowError: the caller asked for a clear group validation error.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise CaptureGroupError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError, TypeError):
        raise CaptureGroupError(f"{label} must be a finite number") from None
    if not math.isfinite(number):
        raise CaptureGroupError(f"{label} must be a finite number")
    return number


def normalize(declarations, tracks) -> list[dict]:
    """Validate a declaration list against the loaded tracks.

    ``declarations`` is the canonical list (or ``None``/empty for "no groups").
    ``tracks`` is a list of facts about the *loaded* tracks, each a dict with:

    ``id``           canonical track ID
    ``role``         the role the analysis gave the track
    ``sample_rate``  the sample rate the file was imported at
    ``duration_s``   the duration of the imported file
    ``dual_mono``    True when the array is mono or dual-mono (L/R exactly equal)
    ``silent``       True when the analysis reads the track as silent, which
                    is the same test the planner filters on: such a track is
                    not in the mix, so it cannot be protected

    Returns the normalized groups: ``[{"id", "members", "pan"}, ...]``.
    Raises :class:`CaptureGroupError` with a message meant for a person.
    """
    if declarations is None:
        return []
    if not isinstance(declarations, list):
        raise CaptureGroupError("capture groups must be a list of group objects")
    if len(declarations) > MAX_GROUPS:
        raise CaptureGroupError(
            f"at most {MAX_GROUPS} capture groups are supported (got {len(declarations)})"
        )

    by_id: dict[str, list[dict]] = {}
    for fact in tracks:
        by_id.setdefault(fact["id"], []).append(fact)

    groups: list[dict] = []
    seen_group_ids: set[str] = set()
    claimed: dict[str, str] = {}

    for index, raw in enumerate(declarations):
        label = f"capture group #{index + 1}"
        if not isinstance(raw, dict):
            raise CaptureGroupError(f"{label} must be an object")
        mode = raw.get("mode")
        if "mode" in raw and raw["mode"] != MODE_STEREO_PAIR:
            raise CaptureGroupError(
                f"{label}: unsupported mode {raw['mode']!r}; the only supported mode is "
                f"{MODE_STEREO_PAIR!r}"
            )
        is_pair = "mode" in raw and raw["mode"] == MODE_STEREO_PAIR
        allowed = {"id", "members", "pan"}
        if is_pair:
            allowed |= {"mode", "trim_db"}
        unknown = set(raw) - allowed
        if unknown:
            raise CaptureGroupError(
                f"{label} has unknown field(s): {', '.join(sorted(unknown))} "
                f"(allowed: {', '.join(sorted(allowed))})"
            )
        if "trim_db" in raw and not is_pair:
            raise CaptureGroupError(
                f"{label}: 'trim_db' needs mode {MODE_STEREO_PAIR!r}; a same-source "
                "capture group has no shared pair trim"
            )
        if "id" not in raw or "members" not in raw:
            raise CaptureGroupError(f"{label} needs both 'id' and 'members'")

        gid = raw["id"]
        if not isinstance(gid, str) or not gid.strip():
            raise CaptureGroupError(f"{label} needs a nonempty 'id' string")
        if len(gid) > GROUP_ID_MAX_CHARS:
            raise CaptureGroupError(
                f"{label} id is {len(gid)} characters; the limit is {GROUP_ID_MAX_CHARS}"
            )
        gid = gid.strip()
        if gid in seen_group_ids:
            raise CaptureGroupError(f"capture group id {gid!r} is declared twice")
        seen_group_ids.add(gid)

        members = raw["members"]
        if not isinstance(members, list):
            raise CaptureGroupError(f"capture group {gid!r}: 'members' must be a list")
        if len(members) < MIN_MEMBERS:
            raise CaptureGroupError(
                f"capture group {gid!r} needs at least {MIN_MEMBERS} members "
                f"(got {len(members)})"
            )
        if len(members) > MAX_MEMBERS:
            raise CaptureGroupError(
                f"capture group {gid!r} has {len(members)} members; the limit is {MAX_MEMBERS}"
            )
        if is_pair and len(members) != PAIR_MEMBER_COUNT:
            raise CaptureGroupError(
                f"capture group {gid!r}: a stereo pair needs exactly {PAIR_MEMBER_COUNT} "
                "members -- the declared left channel and the declared right channel "
                f"(got {len(members)}); the order of the two members IS the channel "
                "identity and is never sorted or inferred"
            )
        seen_members: set[str] = set()
        for member in members:
            if not isinstance(member, str) or not member.strip():
                raise CaptureGroupError(
                    f"capture group {gid!r}: every member must be a nonempty track id string"
                )
            if member in seen_members:
                raise CaptureGroupError(
                    f"capture group {gid!r} lists member {member!r} twice"
                )
            seen_members.add(member)

        pan = raw.get("pan", 0.0)
        pan = _finite(pan, f"capture group {gid!r}: 'pan'")
        if not PAN_MIN <= pan <= PAN_MAX:
            raise CaptureGroupError(
                f"capture group {gid!r}: 'pan' must be within {PAN_MIN}..{PAN_MAX} (got {pan})"
            )

        # the shared pair trim: one number, both members, bounded
        trim_db = 0.0
        if is_pair:
            trim_db = _finite(raw.get("trim_db", 0.0), f"capture group {gid!r}: 'trim_db'")
            if not TRIM_MIN <= trim_db <= TRIM_MAX:
                raise CaptureGroupError(
                    f"capture group {gid!r}: 'trim_db' must be within {TRIM_MIN}..{TRIM_MAX} "
                    f"(got {trim_db})"
                )

        # members must resolve to exactly one loaded track each
        facts = []
        for member in members:
            matches = by_id.get(member)
            if not matches:
                raise CaptureGroupError(
                    f"capture group {gid!r}: member {member!r} is not one of the loaded tracks"
                )
            if len(matches) > 1:
                raise CaptureGroupError(
                    f"capture group {gid!r}: member {member!r} is ambiguous "
                    f"({len(matches)} loaded tracks share that id)"
                )
            facts.append(matches[0])

        # a member may belong to one group only
        for member in members:
            owner = claimed.get(member)
            if owner is not None:
                raise CaptureGroupError(
                    f"track {member!r} is in capture groups {owner!r} and {gid!r}; "
                    "a track can only be in one group"
                )
            claimed[member] = gid

        # one physical source, one role: a group that mixes roles is a mistake
        roles = {fact["role"] for fact in facts}
        if len(roles) > 1:
            raise CaptureGroupError(
                f"capture group {gid!r} mixes roles ({', '.join(sorted(roles))}); "
                "a group holds tracks of one role"
            )

        # the imported files must agree on rate and length, or "the same
        # source" is not something we can vouch for
        rates = {fact["sample_rate"] for fact in facts}
        if len(rates) > 1:
            raise CaptureGroupError(
                f"capture group {gid!r} mixes imported sample rates "
                f"({', '.join(str(r) for r in sorted(rates))})"
            )
        # Sample precision, not milliseconds: two imports that differ by one
        # sample are not the same take, and rounding them onto each other
        # would let a mismatched pair into a group that claims to hold their
        # relation together.
        lengths: dict[int, float] = {}
        for fact in facts:
            frames = int(round(float(fact["duration_s"]) * float(fact["sample_rate"])))
            lengths[frames] = float(fact["duration_s"])
        if len(lengths) > 1:
            detail = ", ".join(
                f"{frames} samples ({seconds:.6f} s)"
                for frames, seconds in sorted(lengths.items())
            )
            raise CaptureGroupError(
                f"capture group {gid!r} mixes imported durations ({detail})"
            )

        # only mono / dual-mono material: a real stereo pair is two sources,
        # not one source seen twice
        for fact in facts:
            if not fact.get("dual_mono", True):
                raise CaptureGroupError(
                    f"capture group {gid!r}: track {fact['id']!r} is a stereo file; "
                    "only mono or dual-mono (identical L/R) tracks can be grouped"
                )

        # A member the planner would filter out of the mix cannot be
        # protected: the audio it promises to hold is not in the mix.  Quiet
        # audio is not the same thing as digital silence -- it can sit far
        # below the level floor and still be a real take -- so the message
        # says what the analysis found, not what the file "is".
        for fact in facts:
            if fact.get("silent", False):
                raise CaptureGroupError(
                    f"capture group {gid!r}: member {fact['id']!r} reads as silent "
                    "(no usable audio above the level floor), so the planner would drop "
                    "it from the mix; a group cannot protect audio that is not in the "
                    "mix -- remove that member or use the real take"
                )

        normalized = {
            "id": gid,
            "members": list(members),
            "pan": pan,
        }
        if is_pair:
            # the mode and the shared trim travel with the declaration; a
            # same-source group keeps exactly the three fields it always had
            normalized["mode"] = MODE_STEREO_PAIR
            normalized["trim_db"] = trim_db
        groups.append(normalized)

    return groups


def _clamp_gain(value: float, max_gain_db: float) -> float:
    return round(max(-max_gain_db, min(max_gain_db, value)), 2)


def _clamp_gain_exact(value: float, max_gain_db: float) -> float:
    """Clamp without rounding.

    The pair's numbers are the matrix: a rounded compensation or a rounded
    shared trim would move the image by a fraction of a percent and the
    centred identity would stop holding.
    """
    return max(-max_gain_db, min(max_gain_db, value))


def _exact_db(value: float) -> str:
    """A gain written as the number it actually is.

    The compensation is a full-precision number, so a rounded version of it
    in the evidence would be a different number than the one the renderer
    uses.
    """
    text = repr(float(value))
    return text if text.startswith("-") else "+" + text


def _pair_pan_targets(pan: float) -> tuple[float, float]:
    """The mapped pan of the declared left member and of the declared right one.

    At the centre the pair keeps the imported matrix: the left member sits on
    the left endpoint and the right member on the right endpoint.  Moving the
    shared position walks one member across the image while the other stays
    on its endpoint, so the pair narrows toward the target side instead of
    balancing left against right.
    """
    if pan >= 0.0:
        return 2.0 * pan - 1.0, 1.0
    return -1.0, 2.0 * pan + 1.0


def _apply_stereo_pair(by_id, group, *, pans, max_gain_db, source_facts, warnings):
    """Apply one declared stereo pair.

    Two members, two channels: the pair is not summed onto one common pan,
    each member keeps its own pan and its own level, and the shared trim
    moves both members together.
    """
    gid = group["id"]
    pan = float(group["pan"])
    shared_trim = float(group.get("trim_db", 0.0) or 0.0)
    left, right = group["members"][0], group["members"][1]
    left_pan, right_pan = _pair_pan_targets(pan)
    protected: list[str] = []
    override_notes: list[str] = []
    final_pans: dict[str, float] = {}
    final_gains: dict[str, float] = {}

    for member, target_pan, side in ((left, left_pan, "left"), (right, right_pan, "right")):
        track = by_id.get(member)
        if track is None:
            # The member is not in the mix.  Claiming a pair for the rest
            # would be a claim about audio that is not there.
            raise CaptureGroupError(
                f"capture group {gid!r}: member {member!r} is not in the mix; "
                "a group is never silently reduced"
            )
        protected.append(member)

        trim = float(track.get("trim_db", 0.0) or 0.0)
        wanted = PAIR_BASE_GAIN_DB + shared_trim + trim
        final_gain = _clamp_gain_exact(wanted, max_gain_db)
        track["gain_db"] = final_gain
        final_gains[member] = final_gain

        # the same neutral per-member chain a same-source group gets
        track["hpf_hz"] = PROTECTED_VALUES["hpf_hz"]
        track["eq"] = []
        track["saturation"] = PROTECTED_VALUES["saturation"]
        track["width"] = PROTECTED_VALUES["width"]
        track["sends"] = dict(PROTECTED_VALUES["sends"])
        for field in PROTECTED_ABSENT:
            track.pop(field, None)

        explicit = pans.get(member) if pans else None
        if explicit is not None:
            # the explicit pan is the last word, exactly as it always was
            manual = round(max(PAN_MIN, min(PAN_MAX, float(explicit))), 2)
            if abs(manual - target_pan) > 1e-9:
                warnings.append(
                    f"capture group {gid!r}: the {side} member {member} keeps its own pan "
                    f"{manual:+.2f} instead of the mapped {target_pan:+.2f}; the pair no "
                    "longer sits as one stereo image"
                )
                override_notes.append(
                    f"{side} member {member} keeps its own pan {manual:+.2f} "
                    f"(mapped {target_pan:+.2f})"
                )
            track["pan"] = manual
            final_pans[member] = manual
        else:
            track["pan"] = target_pan
            final_pans[member] = target_pan

        clamped = abs(final_gain - wanted) > 1e-9
        if abs(trim) > 1e-6:
            warnings.append(
                f"capture group {gid!r}: the {side} member {member} has a manual trim of "
                f"{trim:+.2f} dB; that member alone moves, the other member keeps the "
                "pair's level"
            )
        if clamped:
            warnings.append(
                f"capture group {gid!r}: the {side} member {member}'s final gain "
                f"{_exact_db(final_gain)} dB is clamped to the stage limit "
                f"+/-{max_gain_db:g} dB"
            )

        if source_facts is None:
            facts = list(track.get("evidence", []))
        else:
            facts = list(source_facts.get(member, []))
        pan_line = (
            f"capture group {gid!r}: stereo pair -- the shared position {pan:+.2f} maps "
            f"the {side} member to pan {target_pan:+.4f}"
        )
        if explicit is not None:
            pan_line += f", overridden by the explicit pan {final_pans[member]:+.2f}"
        gain_line = (
            f"capture group {gid!r}: stereo pair -- final gain {_exact_db(final_gain)} dB "
            f"(base compensation {_exact_db(PAIR_BASE_GAIN_DB)} dB"
        )
        if abs(shared_trim) > 1e-9:
            gain_line += f" + shared trim {_exact_db(shared_trim)} dB"
        if abs(trim) > 1e-6:
            gain_line += f" + manual trim {_exact_db(trim)} dB"
        if clamped:
            gain_line += f", clamped to the stage limit +/-{max_gain_db:g} dB"
        gain_line += ")"
        evidence = facts + [
            f"capture group {gid!r}: stereo pair -- per-member processing bypassed -- no "
            "automatic gain, high-pass, EQ, de-esser, compressor, ride, saturation, width "
            "or loudness matching; this member keeps its own channel",
            pan_line,
            gain_line,
        ]
        track["evidence"] = evidence

    notes = [
        f"stereo pair: the declared left member {left} and the declared right member "
        f"{right} keep their own channels; the order of the members IS the channel identity",
        f"stereo pair: shared position {pan:+.2f} maps left {left} to pan "
        f"{final_pans[left]:+.4f} and right {right} to pan {final_pans[right]:+.4f}",
        f"stereo pair: shared trim {_exact_db(shared_trim)} dB with base compensation "
        f"{_exact_db(PAIR_BASE_GAIN_DB)} dB per member; final gains {left} "
        f"{_exact_db(final_gains[left])} dB, {right} {_exact_db(final_gains[right])} dB",
    ] + override_notes

    return {
        "id": gid,
        "members": [left, right],
        "protected_tracks": protected,
        "pan": pan,
        "common_send": float(PROTECTED_VALUES["sends"]["plate"]),
        "notes": notes,
        "mode": MODE_STEREO_PAIR,
        "trim_db": shared_trim,
    }


def apply_protection(tracks: list[dict], groups: list[dict], *, pans=None,
                     max_gain_db: float = 18.0,
                     source_facts: dict | None = None) -> tuple[dict, list[str]]:
    """Rewrite the protected members' numeric fields.

    ``tracks`` are the plan's track dicts (mutated in place).  ``groups`` are
    the normalized groups.  ``pans`` is the user's explicit per-track pan map,
    if any -- an explicit pan keeps its priority and is reported as an override.

    ``source_facts`` maps a member id to the evidence lines that describe the
    *source* (the role it was given, what was measured on it).  When it is
    given, the member's evidence is rebuilt from those facts plus the
    protection's own lines: a protected member's evidence has to describe
    what the finished mix actually does with it, not the automatic proposals
    the protection threw away.  Without it the member's existing evidence is
    kept and the protection's lines are appended to it.

    Returns ``(receipt, warnings)``.  The receipt is what goes into the plan;
    the warnings are the honest statements about what the protection does and
    does not cover.
    """
    warnings: list[str] = []
    by_id = {t["id"]: t for t in tracks}
    receipt_groups = []

    for group in groups:
        gid = group["id"]
        if group.get("mode") == MODE_STEREO_PAIR:
            # a declared stereo pair is a different thing from a same-source
            # group: two channels, not one common pan
            receipt_groups.append(
                _apply_stereo_pair(
                    by_id,
                    group,
                    pans=pans,
                    max_gain_db=max_gain_db,
                    source_facts=source_facts,
                    warnings=warnings,
                )
            )
            continue
        pan = float(group["pan"])
        protected: list[str] = []
        notes: list[str] = []

        for member in group["members"]:
            track = by_id.get(member)
            if track is None:
                # The member is not in the mix.  Claiming protection for the
                # rest would be a claim about audio that is not there, so the
                # group is refused instead of quietly reduced.
                raise CaptureGroupError(
                    f"capture group {gid!r}: member {member!r} is not in the mix; "
                    "a group is never silently reduced"
                )
            protected.append(member)

            trim = float(track.get("trim_db", 0.0) or 0.0)
            final_gain = _clamp_gain(trim, max_gain_db)
            track["gain_db"] = final_gain

            track["hpf_hz"] = PROTECTED_VALUES["hpf_hz"]
            track["eq"] = []
            track["saturation"] = PROTECTED_VALUES["saturation"]
            track["width"] = PROTECTED_VALUES["width"]
            track["sends"] = dict(PROTECTED_VALUES["sends"])
            for field in PROTECTED_ABSENT:
                track.pop(field, None)

            explicit = pans.get(member) if pans else None
            if explicit is not None:
                manual = round(max(PAN_MIN, min(PAN_MAX, float(explicit))), 2)
                if abs(manual - pan) > 1e-9:
                    warnings.append(
                        f"capture group {gid!r}: {member} keeps its own pan {manual:+.2f} "
                        f"instead of the group's {pan:+.2f}; the group no longer sits as "
                        "one image"
                    )
                    notes.append(f"{member} keeps its own pan {manual:+.2f}")
                # the explicit pan is the last word, and it is recorded as one
                track["pan"] = manual
            else:
                track["pan"] = pan

            if abs(trim) > 1e-6:
                warnings.append(
                    f"capture group {gid!r}: {member} has a manual trim of {trim:+.2f} dB; "
                    "the members no longer keep their imported relative level"
                )

            # The final evidence of a protected member describes the mix that
            # comes out of it: the source facts, what the protection bypassed,
            # and the numbers the renderer will actually use.  The automatic
            # proposals it dropped (style gain, ride, vocal unmask, loudness
            # matching, the automatic placement) are not part of that mix, so
            # they are not part of the story either -- saying "ride applied"
            # here would be a claim about audio nobody will hear.
            if source_facts is None:
                facts = list(track.get("evidence", []))
            else:
                facts = list(source_facts.get(member, []))
            evidence = facts + [
                f"capture group {gid!r}: per-member processing bypassed -- no automatic "
                "gain, high-pass, EQ, de-esser, compressor, ride, saturation, width or "
                "loudness matching; one common pan"
            ]
            if abs(trim) > 1e-6:
                clamped = abs(final_gain - trim) > 0.005
                evidence.append(
                    f"capture group {gid!r}: final gain {final_gain:+.2f} dB "
                    f"(0 dB base + manual trim {trim:+.2f} dB"
                    + (f", clamped to {final_gain:+.2f} dB by the stage limit)"
                       if clamped else ")")
                )
            else:
                evidence.append(
                    f"capture group {gid!r}: final gain {final_gain:+.2f} dB "
                    "(0 dB base, no manual trim)"
                )
            if explicit is not None:
                evidence.append(
                    f"capture group {gid!r}: explicit pan {manual:+.2f} overrides the "
                    f"group's common pan {pan:+.2f}"
                )
            else:
                evidence.append(f"capture group {gid!r}: common pan {pan:+.2f}")
            track["evidence"] = evidence

        # A member that is not in the mix was refused above, and normalize()
        # only hands over groups whose members are all loaded, non-silent
        # tracks -- so the group always keeps every member it declared.  There
        # is no partially protected group to report.

        receipt_groups.append(
            {
                "id": gid,
                "members": list(group["members"]),
                "protected_tracks": protected,
                "pan": pan,
                "common_send": float(PROTECTED_VALUES["sends"]["plate"]),
                "notes": notes,
            }
        )

    has_pair = any(g.get("mode") == MODE_STEREO_PAIR for g in receipt_groups)
    has_group = any(g.get("mode") != MODE_STEREO_PAIR for g in receipt_groups)
    if has_pair:
        # The boundary describes the groups this plan actually holds: a pair
        # has no common pan, so the same-source sentence is scoped to the
        # same-source groups instead of being left to cover the pair too.
        parts = []
        if has_group:
            parts.append(BOUNDARY_WITH_GROUP)
        parts.append(STEREO_PAIR_BOUNDARY)
        boundary = "\n\n".join(parts)
    else:
        boundary = BOUNDARY
    receipt = {
        "mode": "declared_capture_group",
        "groups": receipt_groups,
        "boundary": boundary,
    }
    return receipt, warnings
