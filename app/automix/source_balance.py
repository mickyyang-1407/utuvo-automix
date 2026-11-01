"""Resolve input balance intent once, before either planning pass.

This is provenance evidence, not a sound-quality score. No raw audio or shared
cache is retained. Rendering only consumes the resulting numeric plan.
"""
from __future__ import annotations

import math

import numpy as np

from . import analyze, dsp, reference_balance as fitmod

PROFILES = ("auto", "balanced_stems", "raw_tracks", "reference_balance")
METHOD = "aligned_stereo_gain_fit_v1"
LIMITS = {"max_residual": 0.08, "min_correlation": 0.99,
          "min_active_s": 8.0, "max_duration_difference_s": 0.05}


def legacy_api_balance(balance_preserve):
    """Keep finite clamping at the existing planner/pipeline Python APIs.

    CLI/HTTP and resolve() deliberately use strict inputs(). Clamp only after
    rejecting booleans and nonfinite/non-numeric values, never converting NaN
    into an endpoint. Metadata records the effective explicit value.
    """
    if balance_preserve is None:
        return None
    if (isinstance(balance_preserve, bool) or not isinstance(balance_preserve, (int, float))
            or not math.isfinite(balance_preserve)):
        raise ValueError("balance_preserve must be a finite number")
    return min(1.0, max(0.0, float(balance_preserve)))


def inputs(profile="auto", balance_preserve=None):
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError("source_balance_profile must be auto, balanced_stems, raw_tracks or reference_balance")
    if balance_preserve is not None:
        if (isinstance(balance_preserve, bool) or not isinstance(balance_preserve, (int, float))
                or not math.isfinite(balance_preserve) or not 0 <= balance_preserve <= 1):
            raise ValueError("balance_preserve must be a finite number within 0..1")
        balance_preserve = float(balance_preserve)
    return profile, balance_preserve


def _stereo(audio):
    x = np.asarray(audio)
    if x.ndim == 1:
        return np.repeat(x[:, None], 2, axis=1)
    if x.ndim == 2 and x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("source balance proof requires mono or stereo audio")
    return x


def _source_decode_policy(stems):
    """Decode the reference the way the sources were decoded.

    The proof compares the source sum with the reference sample by sample, so
    both sides must come out of the same decoder: under ``float32_v1`` a mono
    FLAC reference is 3.01 dB lower than under ``float32_v2``.  Stems that do
    not record a policy, or disagree, leave the loader's default.
    """
    policies = {getattr(s, "decode_policy", None) for s in stems}
    if len(policies) == 1:
        policy = policies.pop()
        if policy in analyze.DECODE_POLICIES:
            return policy
    return analyze.DEFAULT_DECODE_POLICY


def proof(stems, reference, *, _decoded=None):
    """Fit one positive stereo gain on the unprocessed, aligned source sum.

    Zero-lag only: no search for an offset, channel remapping or per-stem gain.
    All comparable samples contribute to residual/correlation. Coverage uses
    250 ms blocks where BOTH source and reference exceed -60 dBFS and their
    own peak block power minus40 dB. Durations must match within50 ms before
    comparing the common samples; this does not search or correct time offsets.
    """
    result = {"method": METHOD, "passed": False, "limits": dict(LIMITS)}
    def fail(reason):
        result["reason"] = reason
        return result
    if not reference or not reference.get("path"):
        return fail("no_reference_audio")
    if not stems:
        return fail("no_source_audio")
    sr = int(stems[0].sr)
    if any(s.sr != sr for s in stems):
        return fail("source_sample_rates_not_aligned")
    try:
        ref, ref_sr = (_decoded if _decoded is not None else
                       analyze.load_stem_file(reference["path"], _source_decode_policy(stems)))
        ref = _stereo(ref)
        source_frames = max(len(s.audio) for s in stems)
        ref_duration = len(ref) / float(ref_sr)
        duration = source_frames / float(sr)
        result.update(source_duration_s=duration, reference_duration_s=ref_duration,
                      source_sample_rate=sr, reference_sample_rate=int(ref_sr),
                      sample_rate_treatment="fft_resample_reference" if ref_sr != sr else "native",
                      duration_difference_s=abs(duration - ref_duration))
        if abs(duration - ref_duration) > LIMITS["max_duration_difference_s"]:
            return fail("duration_mismatch")
        if min(duration, ref_duration) < LIMITS["min_active_s"]:
            return fail("shorter_than_minimum_coverage")
        if ref_sr != sr:
            ref = dsp.resample_fft(ref, ref_sr, sr)
        n = min(source_frames, len(ref))
        block = max(1, int(round(sr * 0.25)))
        energies = []
        xx = yy = xy = 0.0
        for start in range(0, n, block):
            end = min(n, start + block)
            x = np.zeros((end - start, 2), dtype=np.float64)
            for stem in stems:
                available = min(end, len(stem.audio)) - start
                if available > 0:
                    x[:available] += _stereo(stem.audio[start:start + available])
            y = ref[start:end]
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                return fail("nonfinite_audio")
            a = float(np.sum(x * x)); b = float(np.sum(y * y))
            cross = float(np.sum(x * y))
            if not all(math.isfinite(v) for v in (a, b, cross, xx + a, yy + b, xy + cross)):
                return fail("nonfinite_audio_energy")
            xx += a; yy += b; xy += cross
            energies.append((a / x.size, b / y.size, (end - start) / sr))
        if xx <= 1e-20 or yy <= 1e-20:
            return fail("silent_source_or_reference")
        xgate = max(1e-6, max(a for a, _, _ in energies) * 1e-4)
        ygate = max(1e-6, max(b for _, b, _ in energies) * 1e-4)
        active = sum(seconds for a, b, seconds in energies if a >= xgate and b >= ygate)
        correlation = xy / math.sqrt(xx * yy)
        gain = xy / xx
        residual = math.sqrt(max(0.0, yy - xy * xy / xx) / yy)
        result.update(comparable_duration_s=n / sr, active_coverage_s=active,
                      active_fraction=active / (n / sr), gain=gain,
                      gain_db=20 * math.log10(gain) if gain > 0 else None,
                      relative_rms_residual=residual, correlation=correlation)
        if active < LIMITS["min_active_s"]:
            return fail("insufficient_active_coverage")
        if correlation < LIMITS["min_correlation"] or gain <= 0:
            return fail("incoherent_or_inverted_audio")
        if residual > LIMITS["max_residual"]:
            return fail("reconstruction_residual_too_large")
        result.update(passed=True, reason="coherent_same_content_reconstruction")
        return result
    except (OSError, analyze.StemError, ValueError) as exc:
        return fail("reference_audio_unavailable: " + str(exc))


def resolve(stems, reference=None, profile="auto", balance_preserve=None):
    profile, requested_k = inputs(profile, balance_preserve)
    evidence = {"method": METHOD, "passed": False, "reason": "not_required_for_declared_profile"}
    resolved = profile if profile not in ("auto", "reference_balance") else "unknown"
    if requested_k is not None:
        k, origin = requested_k, "explicit"
        evidence["reason"] = "not_required_for_explicit_balance_preserve"
    elif profile == "reference_balance":
        # Only this explicit mode may infer per-stem corrections. Decode once
        # for the scalar proof and the optional richer fit, local to this call.
        decoded = None
        if reference and reference.get("path"):
            try:
                decoded = analyze.load_stem_file(reference["path"], _source_decode_policy(stems))
            except (OSError, analyze.StemError, ValueError) as exc:
                evidence = {"method": METHOD, "passed": False,
                            "reason": "reference_audio_unavailable: " + str(exc)}
        else:
            evidence = proof(stems, reference)
        if decoded is not None:
            evidence = proof(stems, reference, _decoded=decoded)
            if not evidence["passed"]:
                scalar = evidence
                evidence = fitmod.evaluate(stems, *decoded)
                evidence["scalar_proof"] = scalar
        if evidence["passed"]:
            k = 1.0
            if evidence["method"] == METHOD:
                resolved, origin = "balanced_stems", "proof"
            else:
                resolved, origin = "reference_balance", "reference_fit"
        else:
            k, origin = 0.5, "default"
    elif profile == "balanced_stems":
        k, origin = 1.0, "profile"
    elif profile == "raw_tracks":
        k, origin = 0.5, "default"
    else:
        evidence = proof(stems, reference)
        if evidence["passed"]:
            resolved, k, origin = "balanced_stems", 1.0, "proof"
        else:
            k, origin = 0.5, "default"
    return {"requested": profile, "resolved": resolved, "requested_balance_preserve": requested_k,
            "balance_preserve": k, "origin": origin, "proof": evidence}


def validate(decision, known_ids=None, source_sr=None, source_frames=None):
    if not isinstance(decision, dict):
        raise ValueError("source_balance must be an object")
    inputs(decision.get("requested"), decision.get("requested_balance_preserve"))
    inputs("auto", decision.get("balance_preserve"))
    if decision.get("balance_preserve") is None:
        raise ValueError("source_balance requires a resolved balance_preserve")
    if decision.get("resolved") not in ("unknown", "balanced_stems", "raw_tracks", "reference_balance"):
        raise ValueError("unsupported resolved source balance profile")
    if decision.get("origin") not in ("explicit", "default", "profile", "proof", "reference_fit"):
        raise ValueError("unsupported source balance origin")
    evidence = decision.get("proof")
    if (not isinstance(evidence, dict) or evidence.get("method") not in (METHOD, *fitmod.SUPPORTED_METHODS)
            or type(evidence.get("passed")) is not bool or not isinstance(evidence.get("reason"), str)):
        raise ValueError("invalid source balance proof")
    def finite_tree(value):
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError("source balance proof numbers must be finite")
        if isinstance(value, dict):
            for child in value.values(): finite_tree(child)
        elif isinstance(value, list):
            for child in value: finite_tree(child)
    finite_tree(evidence)
    if evidence["method"] in fitmod.SUPPORTED_METHODS:
        if decision["requested"] != "reference_balance" or decision["requested_balance_preserve"] is not None:
            raise ValueError("reference fit requires its explicit profile and no explicit k")
        fitmod.validate_receipt(evidence, known_ids, source_sr, source_frames)
        if evidence["passed"]:
            if (decision["resolved"] != "reference_balance" or decision["origin"] != "reference_fit"
                    or decision["balance_preserve"] != 1):
                raise ValueError("accepted reference fit decision is inconsistent")
        elif decision["resolved"] != "unknown" or decision["origin"] != "default" or decision["balance_preserve"] != .5:
            raise ValueError("rejected reference fit must use the unknown default")
    elif decision["origin"] == "reference_fit" or decision["resolved"] == "reference_balance":
        raise ValueError("reference balance requires an accepted fit proof")
