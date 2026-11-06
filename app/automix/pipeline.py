"""The whole job, in one call: stems + style (+ optional reference) -> mix.

This exists so the command line, the tests and (later) the app all run the
*same* sequence, and so the two-pass part of reference mode lives in one place
instead of being re-implemented per caller.

Why two passes?  Reference mode corrects our result *toward* the reference.
To know what to correct, it has to know what we currently produce -- after the
renderer's own HPFs, mono-below, bus compression and limiting.  Measuring the
raw stem sum instead would blame those decisions on the reference.  So: render
once with the preset, measure that, then re-plan with those measurements as the
baseline and render again.  One extra render (under a second here) buys a
correction that is actually about the reference.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from . import analyze, dsp, plan as planner, reference as refmod, render, source_balance as balancemod


def note_reference_tone(plan: dict, audio, report: dict) -> None:
    """Measure the result on the plan's frozen comparison set; no DSP decisions."""
    reference = (plan.get("style") or {}).get("reference") or {}
    comparison = reference.get("tone_comparison")
    if not comparison:
        return  # old plans have no new comparison contract
    indices = comparison["indices"]
    after_tone = refmod.spectrum_db(audio, int(plan["source"]["sample_rate"]))
    before = comparison["before_db"]
    after = refmod.tone_distance(reference["tone"], after_tone, indices)
    if after is None:
        diagnostic = "insufficient common support: fewer than 2 bands; no relative tone claim"
    elif after <= 0.5:
        diagnostic = "within 0.5 dB RMS relative-tone distance on the fixed common bands"
    elif before is not None and after < before:
        diagnostic = "distance decreased; remaining mismatch under the bounded five-region EQ"
    else:
        diagnostic = "reference tone not reached; distance did not decrease under the current constraints"
    report["reference_tone"] = {
        "method": comparison["method"], "analysis_method": dict(refmod.TONE_METHOD),
        "indices": list(indices), "centers_hz": comparison["centers_hz"],
        "band_count": len(indices), "before_db": before, "after_db": after,
        "baseline_kind": comparison["baseline_kind"], "diagnostic": diagnostic,
        "after_tone": after_tone,
    }


def _mono_context(report: dict) -> tuple[int | None, float | None]:
    """The sample rate and stereo loudness the mono comparison needs.

    Both come from the report, because both sides of the comparison have to
    be measured the same way: the stereo loudness was taken at the render's
    native rate, so the fold-down has to be taken there too.  The meter
    resamples to 48 kHz by itself, but only when it is told the real rate.
    Handed a 96/192 kHz signal at a 48 kHz rate it skips the resample and
    reads several dB low -- enough to invent a phase problem that is not in
    the mix.  An unknown rate is therefore not a licence to assume 48 kHz:
    the diagnostic reports itself unavailable instead.
    """
    sr = None
    raw = report.get("sample_rate")
    if raw is not None and not isinstance(raw, bool):
        try:
            value = int(raw)
            if float(raw) == value and value > 0:
                sr = value
        except (TypeError, ValueError, OverflowError):
            sr = None
    stereo = None
    raw = report.get("lufs")
    if raw is not None and not isinstance(raw, bool):
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            value = None
        if value is not None and math.isfinite(value):
            stereo = value
    return sr, stereo


def _note_mono(report: dict, audio, warnings: list) -> None:
    """Check the mix survives a mono fold-down.

    Phones, club PAs and most single-speaker listening are mono.  A mix that
    only sounds right in stereo is a real-world bug, and it is cheap to spot:
    fold to mono and compare loudness.  A 3 dB drop is the point where most
    engineers would start looking for a phase problem.
    """
    try:
        if audio.ndim != 2 or audio.shape[1] != 2:
            return
        # a number from an earlier measurement must never outlive this one,
        # whatever happens below
        report.pop("mono_loss_db", None)
        report.pop("mono_diagnostic", None)
        sr, stereo = _mono_context(report)
        if sr is None or stereo is None:
            report["mono_diagnostic"] = {
                "available": False,
                "reason": "sample rate or stereo loudness is not available in the report",
            }
            return
        if stereo <= -70.0:
            # under the BS.1770 gating floor there is nothing to compare
            report["mono_diagnostic"] = {
                "available": False,
                "reason": "stereo loudness is below the gating floor; mono fold-down is not measurable",
            }
            return
        mono_lufs = dsp.loudness_lufs(audio.mean(axis=1, keepdims=True), sr)
        # BS.1770 sums channels, so the same signal in one channel reads 3 dB
        # lower than in two.  Without this correction every dual-mono mix looks
        # like it "loses 3 dB in mono", which is exactly what the first version
        # reported.
        loss = stereo - (mono_lufs + 3.0103)
        report["mono_loss_db"] = round(loss, 2)
        report["mono_diagnostic"] = {
            "available": True,
            "sample_rate": sr,
            "stereo_lufs": round(stereo, 2),
            "mono_lufs": round(mono_lufs, 2),
            "loss_db": round(loss, 2),
        }
        if loss > 3.0:
            warnings.append(
                f"mono fold-down loses {loss:.1f} dB; check for phase problems in wide parts"
            )
    except Exception:  # noqa: BLE001 - a diagnostic must never fail a mix
        return


def _note_width(plan: dict, report: dict, warnings: list) -> None:
    """Say when the requested width could not be reached.

    The renderer widens by measuring the side/mid ratio it actually has, then
    caps the multiplier with the style guard.  A reference that is much wider
    than the guard allows therefore lands short of its target -- which is the
    right behaviour, but the receipt has to say so instead of quietly reporting
    a different number.
    """
    target = (plan.get("buses") or {}).get("master", {}).get("width_target")
    if not target:
        return
    achieved = float(report.get("width", 0.0))
    if achieved < float(target) * 0.9:
        warnings.append(
            f"width target {float(target):.3f} could not be reached "
            f"(got {achieved:.3f}); the width guard limits how far the side can be pushed"
        )


def _note_balance(plan: dict, report: dict, warnings: list) -> None:
    """Say when the lead vocal is buried, using the measured contributions.

    "The vocal is too quiet" is the single most common note on a rough mix, and
    it is measurable: compare the vocal's contribution to the sum of everything
    else.  A 12 dB gap is wide enough that most people would call it buried.
    """
    tracks = report.get("tracks") or []
    vocal = next((t for t in tracks if t.get("role") == "lead_vocal"), None)
    if not vocal:
        return
    others = [t for t in tracks if t.get("role") != "lead_vocal" and t.get("contrib_db", -120) > -70]
    if not others:
        return
    # power sum of the others, in dB
    power = sum(10 ** (t["contrib_db"] / 10.0) for t in others)
    others_db = 10 * np.log10(power + 1e-30)
    gap = float(vocal["contrib_db"]) - float(others_db)
    if gap < -12.0:
        warnings.append(
            f"lead vocal sits {abs(gap):.1f} dB under everything else; "
            "try a positive trim on it"
        )


def _note_crest(plan: dict, preset: dict, report: dict) -> None:
    """Report output crest's gap below the style reference.

    This compares the output to ``crest_target_db``, not to the input.  The
    reference gap does not measure dynamics lost or identify their cause;
    it can prompt a comparison at equal loudness.
    """
    want = preset.get("crest_target_db")
    if want is None:
        return
    got = float(report.get("crest_db", 0.0))
    miss = got - float(want)
    if miss < -4.0:
        plan.setdefault("warnings", []).append(
            f"dynamics: output crest {got:.1f} dB is {abs(miss):.1f} dB below the "
            f"{float(want):.1f} dB style reference. "
            "This reference gap does not measure a loss of dynamics or establish its cause. "
            "Compare at equal loudness to assess the result."
        )


def _timed_reverb(reverb_kwargs: dict, caller_keys: set, plan: dict) -> dict:
    """A tempo-locked plan (reverb.timing) renders with its own decay and
    pre-delay -- the numbers a replay of the saved plan uses -- unless the
    caller asked for others.  Without timing nothing changes."""
    rev = plan.get("reverb") or {}
    if not isinstance(rev.get("timing"), dict):
        return reverb_kwargs
    out = dict(reverb_kwargs)
    # exactly what the renderer reads from a plan when no kwargs are given
    # (render.render), so the first render and a later replay agree
    if "decay_s" not in caller_keys:
        out["decay_s"] = float(rev.get("decay_s", 1.6))
    if "predelay_ms" not in caller_keys:
        out["predelay_ms"] = float(rev.get("predelay_ms", rev.get("pre_delay_ms", 20.0)))
    return out


def _try_headroom(stems, plan, audio, report, reverb_kwargs, collect):
    """Keep a bounded retry only when it meets both mastering gates."""
    buses = report.get("buses", {})
    gr = report.get("limiter", {}).get("gain_reduction_p95_db")
    if gr is None or gr >= -3.0:
        return plan, audio, report
    target = report["target_lufs"]
    # A tiny miss can come from the guard's own convergence tolerance. For
    # deeper misses, try a small fader correction before limiting harder.
    drum = (report.get("part_balance") or {}).get("drum", {})
    if gr >= -3.05:
        trims = {}
    elif not (buses.get("drum") and buses.get("bass")):
        return plan, audio, report
    elif (buses.get("music") and drum.get("applied_db", 0) <= -12
          and drum["result_db"] > drum["target_db"]):
        # Keep the drum-to-music fader relationship intact on this retry.
        trims = {"drum": -3.0, "music": -3.0}
    else:
        trims = {"drum": -1.0}

    candidate = copy.deepcopy(plan)
    candidate["headroom_trim_db"] = trims
    # The existing guard tolerates 0.05 dB of overshoot. Leave room for it
    # while checking the measured result against the original 3 dB limit.
    candidate["buses"]["master"]["limiter"]["max_gr_p95_db"] = 2.94
    if "music" in trims:
        candidate["buses"]["master"]["limiter"]["max_backoff_db"] = 1.70
    planner.validate(candidate)
    trial_collect = {} if collect is not None else None
    trial_audio, trial = render.render(stems, candidate, reverb_kwargs, collect=trial_collect)
    trial_gr = trial["limiter"].get("gain_reduction_p95_db")
    if (trial_gr is None or trial_gr < -3.0 or not target - 1.0 <= trial["lufs"] <= target + 1.0
            or trial["correlation"] < 0.3 or trial["true_peak_db"] > report["ceiling_db"]):
        return plan, audio, report
    _note_mono(trial, trial_audio, [])
    mono = trial.get("mono_loss_db")
    if mono is None or mono > 3.0:
        return plan, audio, report
    if collect is not None:
        collect.clear()
        collect.update(trial_collect)
    return candidate, trial_audio, trial


def mix(
    stems: list,
    preset: dict,
    preset_path: str | None = None,
    intensity: float = 1.0,
    balance_preserve: float | None = None,
    reference: dict | None = None,
    blend: float = 0.7,
    reverb_kwargs: dict | None = None,
    reverb_match: bool = False,
    trims: dict | None = None,
    pans: dict | None = None,
    target_lufs: float | None = None,
    true_peak_db: float | None = None,
    collect: dict | None = None,
    source_balance_profile: str = "auto",
    reverb_amount: float | None = None,
    capture_groups: list | None = None,
    group_trims: dict | None = None,
) -> tuple[dict, "object", dict]:
    """Return ``(plan, audio, report)``."""
    reverb_amount = planner._manual_reverb_amount(reverb_amount, optional=True)
    reference_pass = reference is not None and blend > 0.0
    try:
        balance_preserve = balancemod.legacy_api_balance(balance_preserve)
        decision = balancemod.resolve(stems, reference, source_balance_profile, balance_preserve)
    except ValueError as exc:
        raise planner.PlanError(str(exc)) from exc
    if reverb_kwargs is None:
        reverb_kwargs = {}
    # Presets carry the reverb's shape per style (trap 15 ms, cinematic 35 ms).
    # Until now nothing copied it across, so every style got the same 20 ms and
    # `pre_delay_ms` was a knob that did nothing.  Note the name change: the
    # preset says pre_delay_ms, the IR builder says predelay_ms.
    rev_style = preset.get("reverb") or {}
    caller_reverb = set(reverb_kwargs)
    reverb_kwargs.setdefault("decay_s", float(rev_style.get("decay_s", 1.6)))
    reverb_kwargs.setdefault("predelay_ms", float(rev_style.get("pre_delay_ms", 20.0)))

    # Pass 1 is deliberately *without* the reference: it has to measure what
    # the preset alone produces.  Feeding the reference in here as well makes
    # pass 2 compare the reference against an already-(wrongly)-corrected mix,
    # which is how the +6 dB low shelf kept re-deriving itself.
    plan = planner.build_plan(
        stems,
        preset,
        preset_path=preset_path,
        intensity=intensity,
        balance_preserve=decision["balance_preserve"],
        _source_balance=decision,
        trims=trims,
        group_trims=group_trims,
        pans=pans,
        target_lufs=target_lufs,
        true_peak_db=true_peak_db,
        reverb_amount=None if reference_pass else reverb_amount,
        # both reference passes must see the same declaration: a group that
        # only existed in one of them would be protected in one measurement
        # and not in the other
        capture_groups=capture_groups,
    )
    # The plan is checked before the renderer ever sees it: a plan that does
    # not satisfy the schema must not produce audio.  This is the first
    # reference pass -- the second one is checked the same way below.
    planner.validate(plan)
    reverb_kwargs = _timed_reverb(reverb_kwargs, caller_reverb, plan)
    audio, report = render.render(stems, plan, reverb_kwargs, collect=collect)
    if (reference is None and not (trims or pans or group_trims)
            and target_lufs is None and true_peak_db is None):
        plan, audio, report = _try_headroom(stems, plan, audio, report, reverb_kwargs, collect)
    _note_crest(plan, preset, report)
    _note_balance(plan, report, plan.setdefault("warnings", []))
    _note_width(plan, report, plan.setdefault("warnings", []))
    if "headroom_trim_db" not in plan:
        _note_mono(report, audio, plan.setdefault("warnings", []))

    if reference is None or blend <= 0.0:
        return plan, audio, report

    sr = int(plan["source"]["sample_rate"])
    baseline = {
        "tone": refmod.spectrum_db(audio, sr),
        "width": dsp.width_ratio(audio),
        "crest": dsp.crest_db(audio),
        "tail": dsp.tail_ratio(audio, sr),
    }
    # Pass two only needs these measurements, not the first full render.
    # A collector, when requested, continues to own its exported pieces.
    del audio, report
    plan = planner.build_plan(
        stems,
        preset,
        preset_path=preset_path,
        intensity=intensity,
        balance_preserve=decision["balance_preserve"],
        _source_balance=decision,
        reference=reference,
        blend=blend,
        baseline=baseline,
        reverb_match=reverb_match,
        reverb_amount=reverb_amount,
        trims=trims,
        group_trims=group_trims,
        pans=pans,
        target_lufs=target_lufs,
        true_peak_db=true_peak_db,
        capture_groups=capture_groups,
    )
    # The second reference pass is checked the same way: a plan that does not
    # satisfy the schema must not produce audio either.
    planner.validate(plan)
    reverb_kwargs = _timed_reverb(reverb_kwargs, caller_reverb, plan)
    audio, report = render.render(stems, plan, reverb_kwargs, collect=collect)
    _note_crest(plan, preset, report)
    _note_balance(plan, report, plan.setdefault("warnings", []))
    _note_width(plan, report, plan.setdefault("warnings", []))
    _note_mono(report, audio, plan.setdefault("warnings", []))
    report["reference_passes"] = 2
    note_reference_tone(plan, audio, report)
    return plan, audio, report
