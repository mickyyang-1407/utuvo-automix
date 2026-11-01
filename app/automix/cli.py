"""Command line entry point for the automix MVP.

    python3 -m automix.cli --stems demo_stems --style pop_2020s --out out/pop
    python3 -m automix.cli --stems demo_stems --list-styles
    python3 -m automix.cli --render-plan out/pop/mixplan.json --out out/again

The first form is the whole product: drop stems in a folder, pick a style, get
a mix.  The second form is what the receipt points at -- a plan is data, so it
can be re-rendered or hand-edited without re-running the analysis.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import sys
import time

from . import analyze, dsp, plan as planner, pipeline, reference as refmod, render, source_balance as balancemod


def _styles(preset_dir: str) -> list[str]:
    if not os.path.isdir(preset_dir):
        return []
    return sorted(
        os.path.splitext(n)[0] for n in os.listdir(preset_dir) if n.endswith(".json")
    )


def _default_preset_dir() -> str:
    """Where the style presets live when the caller does not say.

    In the source tree they sit next to ``app/``; in a packaged App they are
    bundled as data and land in ``sys._MEIPASS/presets``.  An explicit
    ``--preset-dir`` still wins -- this is only the default.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return os.path.join(str(meipass), "presets")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(os.path.dirname(here), "presets")


def _plan_summary(p: dict) -> str:
    """A compact, human-readable view of what the planner decided."""
    m = (p.get("buses") or {}).get("master") or {}
    lines = [
        f"target     {m.get('target_lufs')} LUFS, ceiling {m.get('true_peak_db')} dBTP",
        "",
        f"{'track':<18}{'role':<17}{'gain':>8}{'pan':>7}  decisions",
    ]
    for t in p.get("tracks", []):
        marks = []
        if t.get("trim_db"):
            marks.append(f"trim{t['trim_db']:+.1f}")
        if t.get("deess"):
            marks.append("deess")
        if t.get("ride"):
            marks.append("ride")
        if t.get("eq"):
            marks.append(f"{len(t['eq'])} eq")
        lines.append(
            f"{t['id']:<18}{t['role']:<17}{t['gain_db']:>+7.2f}{t['pan']:>+7.2f}  "
            + (", ".join(marks) or "-")
        )
    if p.get("style", {}).get("source_balance"):
        b = p["style"]["source_balance"]
        lines += ["", f"source balance {b['requested']} -> {b['resolved']} · k={b['balance_preserve']} ({b['origin']})",
                  "  " + b["proof"]["reason"]]
        if b["origin"] == "reference_fit":
            lines += [f"  source-only gain fit; common {b['proof']['common_gain_db']:+.3f} dB unapplied"]
            lines += [f"  {id}: relative correction {gain:+.3f} dB" for id,gain in b["proof"]["relative_gain_db"].items()]
        elif b["requested"] == "reference_balance":
            lines += ["  manual k skips fitting" if b["origin"] == "explicit" else
                      "  scalar preservation; no new per-stem correction" if b["origin"] == "proof" else
                      "  fit unavailable; existing unknown-input balance 0.5 retained"]
    if p.get("sections"):
        lines += ["", "sections   " + ", ".join(
            f"{s['label']} {s['start_s']:.0f}-{s['end_s']:.0f}s" for s in p["sections"]
        )]
    if (p.get("capture_groups") or {}).get("groups"):
        lines += ["", "capture groups (protected before the common bus/master):"]
        for g in p["capture_groups"]["groups"]:
            if g.get("mode") == "stereo_pair":
                # A pair is not one common pan: the two members keep their own
                # channel and their own pan.
                members = g.get("members") or []
                left = members[0] if len(members) > 0 else ""
                right = members[1] if len(members) > 1 else ""
                lines.append(
                    f"  {g['id']}: stereo pair, left {left}, right {right} · "
                    f"pair position {g['pan']:+.2f} · "
                    f"shared gain {float(g.get('trim_db', 0.0)):+.2f} dB"
                )
            else:
                # The receipt never carries silent members: normalize() refuses a
                # group whose member the planner would drop, so a protected group
                # always holds every member it declared.
                lines.append(
                    f"  {g['id']}: {', '.join(g['protected_tracks'])} pan {g['pan']:+.2f}"
                )
        if any(g.get("mode") == "stereo_pair" for g in p["capture_groups"]["groups"]):
            lines.append(
                "  stereo pair: the two members keep their own channel and their own pan. "
                "At position 0 the imported left/right relationship is kept before the "
                "common bus/master; moving the position narrows the image and both members "
                "meet on the same side at the extremes; the shared trim moves both members "
                "together. Stereo image position -- not a balance control, not an engineer's "
                "recipe, not a quality guarantee."
            )
    if p.get("warnings"):
        lines += ["", "warnings:"]
        lines += [f"  - {w}" for w in p["warnings"]]
    return "\n".join(lines)


def _validation_detail(exc: Exception) -> str:
    """One honest line about why a plan was refused."""
    message = getattr(exc, "message", None) or str(exc)
    where = getattr(exc, "json_path", None)
    if where:
        message = f"{message} (at {where})"
    return " ".join(str(message).split())


def _load_saved_plan(path: str) -> dict:
    """Read a saved plan and refuse it when it does not hold up.

    The plan is checked before a single source is loaded, a sample is rendered
    or an output file is written, so a broken plan cannot leave a half-mixed
    directory behind.  Every refusal says what actually went wrong: a file
    that cannot be read, text that is not JSON, JSON that is not an object, or
    a plan that does not satisfy the contract the engine renders from.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            p = json.load(fh)
    except OSError as exc:
        raise planner.PlanError(f"--render-plan: cannot read {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise planner.PlanError(f"--render-plan: {path} is not UTF-8 text: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise planner.PlanError(f"--render-plan: {path} is not valid JSON: {exc}") from exc
    except ValueError as exc:
        # Well-formed JSON the interpreter refuses to convert -- an integer
        # longer than Python's digit limit, for one.  The file is not the
        # problem, the limit is; say which one instead of raising it.
        raise planner.PlanError(f"--render-plan: {path} could not be parsed: {exc}") from exc
    if not isinstance(p, dict):
        raise planner.PlanError(
            f"--render-plan: {path} must hold a JSON object, not {type(p).__name__}"
        )
    source = p.get("source")
    if source is not None and not isinstance(source, dict):
        raise planner.PlanError(
            f"--render-plan: {path}: source must be an object, not {type(source).__name__}"
        )
    stems_dir = (source or {}).get("stems_dir")
    if stems_dir is not None and not isinstance(stems_dir, str):
        # The schema does not constrain this field, but the CLI walks it as a
        # path: a list or an object here blows up on the first stat().
        raise planner.PlanError(
            f"--render-plan: {path}: source.stems_dir must be a path string, "
            f"not {type(stems_dir).__name__}"
        )
    try:
        planner.validate(p)
    except planner.PlanError as exc:
        raise planner.PlanError(f"--render-plan: {path}: {exc}") from exc
    except Exception as exc:
        # The validator's own failures (jsonschema, or a broken plan that
        # trips an invariant it reads structurally) are reported as they are
        # instead of as a traceback.
        raise planner.PlanError(
            f"--render-plan: {path} is not a valid mix plan: {_validation_detail(exc)}"
        ) from exc
    return p


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog=prog or "automix",
        description="Drop stems, pick a style, get a mix.",
    )
    ap.add_argument("--stems", help="folder of stem files")
    ap.add_argument("--style", default="pop_2020s", help="style preset id")
    ap.add_argument("--out", default="automix_out", help="output folder")
    ap.add_argument("--intensity", type=float, default=1.0, help="0..1 style strength")
    ap.add_argument(
        "--balance-preserve",
        type=float,
        default=None,
        help="0..1 explicit override: how much of the source's own balance to keep (0 = the style decides)",
    )
    ap.add_argument("--source-balance-profile", choices=balancemod.PROFILES, default="auto",
                    help="input intent: auto, balanced_stems, raw_tracks, or reference_balance (same-song aligned rough mix, gain ratios only)")
    ap.add_argument("--preset-dir", default=None, help="override the presets folder")
    ap.add_argument(
        "--reference",
        default=None,
        help="audio file whose loudness / tone / dynamics / width to aim at",
    )
    ap.add_argument(
        "--blend",
        type=float,
        default=0.7,
        help="0..1 weight of the reference against the style preset",
    )
    ap.add_argument(
        "--target-lufs",
        type=float,
        default=None,
        help="override the style's loudness target (e.g. -14 for streaming)",
    )
    ap.add_argument(
        "--true-peak",
        type=float,
        default=None,
        help="override the style's true-peak ceiling in dBTP (e.g. -1.0)",
    )
    ap.add_argument(
        "--reverb-match",
        action="store_true",
        help="experimental: nudge reverb toward the reference's measured tail (bounded +-20%%)",
    )
    ap.add_argument(
        "--export-stems",
        default=None,
        metavar="DIR",
        help="also write the processed stems (and the reverb return) for a DAW",
    )
    def reverb_amount_arg(text):
        try:
            return planner._manual_reverb_amount(float(text))
        except (ValueError, planner.PlanError) as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
    ap.add_argument("--reverb-amount", type=reverb_amount_arg, default=None,
                    help="0..1 amount of this tool's added reverb; 0 keeps source room intact")
    ap.add_argument(
        "--plan-only",
        action="store_true",
        help="build and write the plan (and receipt) without rendering audio",
    )
    ap.add_argument("--render-plan", default=None, help="re-render an existing mixplan.json")
    ap.add_argument(
        "--capture-groups",
        default=None,
        metavar="PATH",
        help=(
            "JSON file with the declared capture groups: which loaded tracks are the "
            "same physical source recorded by several microphones "
            '([{"id": "room pair", "members": ["mic_l", "mic_r"], "pan": -0.2}])'
        ),
    )
    ap.add_argument("--list-styles", action="store_true", help="show available styles")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.plan_only and args.export_stems:
        ap.error("--plan-only does not render, so it cannot --export-stems")
    if args.capture_groups and args.render_plan:
        # a saved plan is replayed as it is: its numbers are the mix.  Fitting
        # a group into it here would silently change what the plan says.
        ap.error(
            "--capture-groups cannot be combined with --render-plan: "
            "the saved plan's numbers are replayed as they are"
        )

    capture_groups = None
    if args.capture_groups:
        try:
            with open(args.capture_groups, encoding="utf-8") as fh:
                capture_groups = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            ap.error(f"--capture-groups: {exc}")

    preset_dir = args.preset_dir or _default_preset_dir()

    if args.list_styles:
        english = planner.style_text_en(preset_dir)
        for name in _styles(preset_dir):
            p = planner.load_preset(name, preset_dir)
            # the presets are written in Chinese; print the English text when there is one
            text = {**(p.get("intent") or {}), "description": p.get("description", ""),
                    **(english.get(p["id"]) or {})}
            print(f"{name:<16} {p.get('name', ''):<26} {text['description']}")
            for key, label in (("applicability", "fits"), ("caveat", "caveat"), ("numeric_basis", "numbers")):
                if text.get(key):
                    print(f"  {label}: {text[key]}")
        return 0

    def say(msg: str) -> None:
        if not args.quiet:
            print(msg, flush=True)

    intensity = max(0.0, min(1.0, args.intensity))
    try:
        _, balance_preserve = balancemod.inputs(args.source_balance_profile, args.balance_preserve)
    except ValueError as exc:
        ap.error(str(exc))
    # a dict is filled in place by render(); None means "do not collect"
    collected: dict | None = {} if args.export_stems else None

    # A saved plan is checked before anything is loaded, rendered or written:
    # an invalid plan has to fail without leaving an output directory behind.
    saved_plan = None
    if args.render_plan:
        try:
            saved_plan = _load_saved_plan(args.render_plan)
        except planner.PlanError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    os.makedirs(args.out, exist_ok=True)
    mix_path = os.path.join(args.out, "mix.wav")
    plan_path = os.path.join(args.out, "mixplan.json")
    report_path = os.path.join(args.out, "report.md")

    try:
        if args.render_plan:
            p = saved_plan
            if args.reverb_amount is not None:
                p["reverb"] = dict(p.get("reverb") or {})
                p["reverb"]["manual_amount"] = args.reverb_amount
            rev = dict(p.get("reverb") or {})
            if "manual_amount" in rev:
                planner._manual_reverb_amount(rev["manual_amount"])
            stems_dir = (
                args.stems if args.stems is not None
                else p.get("source", {}).get("stems_dir")
            )
            if not stems_dir or not os.path.isdir(stems_dir):
                raise planner.PlanError(
                    f"unusable source.stems_dir: {stems_dir!r}; "
                    "use --stems to select the source folder"
                )
            policy = planner.plan_decode_policy(p)
            say(f"loading stems from plan: {stems_dir} ({policy})")
            # The plan decides how its own sources are decoded.  A plan that
            # records a policy is replayed with exactly that policy; a plan
            # that records none was built with the legacy decoder, so it is
            # replayed that way -- whatever the current default happens to be.
            stems = analyze.load_stems(stems_dir, policy)
            available = {analyze.track_id(s.name) for s in stems}
            missing = [t["id"] for t in p["tracks"] if t["id"] not in available]
            if missing:
                raise planner.PlanError(
                    "missing source track(s) required by saved plan: " + ", ".join(missing)
                )
            target_sr = int(p["source"]["sample_rate"])
            for s in stems:
                if s.sr != target_sr:
                    loaded_sr = s.sr
                    s.audio = dsp.resample_fft(s.audio, loaded_sr, target_sr)
                    s.resampled_from = getattr(s, "resampled_from", None) or loaded_sr
                    s.sr = target_sr
                    warning = (
                        f"{s.name}: loaded {loaded_sr} Hz resampled to {target_sr} Hz "
                        "for saved plan replay"
                    )
                    warnings = p.setdefault("warnings", [])
                    if warning not in warnings:
                        warnings.append(warning)
            by_file = {s.name: s for s in stems}
            # trust the plan's roles rather than re-guessing them
            for entry in p["source"]["stems"]:
                s = by_file.get(entry["file"])
                if s is not None:
                    s.role = entry["role"]
            analyze.align(stems)
            for s in stems:
                analyze.measure(s)
        else:
            if not args.stems:
                ap.error("--stems is required (or use --render-plan)")
            t0 = time.time()
            say(f"loading {args.stems}")
            stems = analyze.load_stems(args.stems)
            analyze.detect_roles(stems)
            analyze.align(stems)
            for s in stems:
                analyze.measure(s)
            say(f"  {len(stems)} stems, {stems[0].duration:.1f} s, {stems[0].sr} Hz")
            need = analyze.estimate_memory_mb(stems)
            total = analyze.system_memory_mb()
            if total:
                # leave room for the OS, the browser and the DAW that made the stems
                analyze.check_memory(stems, budget_mb=total * 0.55)
                say(f"  working set ~{need / 1000:.1f} GB of {total / 1000:.1f} GB RAM")
            for s in stems:
                say(
                    f"  {s.name:<22} {s.role:<15} {s.metrics['lufs']:>7.2f} LUFS  "
                    f"crest {s.metrics['crest_db']:>5.2f}  ({s.role_source})"
                )

            preset = planner.load_preset(args.style, preset_dir)
            say(f"style: {preset.get('name', preset['id'])} (intensity {intensity:.2f})")

            ref = None
            if args.reference:
                ref = refmod.analyse(args.reference)
                blend = max(0.0, min(1.0, args.blend))
                say(
                    f"reference: {ref['file']} -> {ref['lufs']:.2f} LUFS, "
                    f"{ref['true_peak_db']:.2f} dBTP, crest {ref['crest_db']:.1f}, "
                    f"width {ref['width']:.2f} (blend {blend:.2f})"
                )
                say(
                    "  target tone: "
                    + ", ".join(f"{k} {v:+.1f}" for k, v in ref["tone_tilt"].items())
                )

            say(f"analysed in {time.time() - t0:.1f} s")

        if args.plan_only and not args.render_plan:
            p = planner.build_plan(
                stems,
                preset,
                preset_path=args.style,
                intensity=intensity,
                balance_preserve=balance_preserve,
                source_balance_profile=args.source_balance_profile,
                reference=ref,
                blend=args.blend,
                target_lufs=args.target_lufs,
                true_peak_db=args.true_peak,
                reverb_amount=args.reverb_amount,
                capture_groups=capture_groups,
            )
            if args.stems:
                p["source"]["stems_dir"] = os.path.abspath(args.stems)
            planner.validate(p)
            planner.write_plan(p, plan_path)
            say(f"plan written: {plan_path} (--plan-only: no audio rendered)")
            say("")
            say(_plan_summary(p))
            return 0

        t1 = time.time()
        if args.render_plan:
            mix, report = render.render(stems, p, collect=collected)
        else:
            # pipeline.mix re-plans against the rendered baseline in reference
            # mode, so it owns planning from here on
            p, mix, report = pipeline.mix(
                stems,
                preset,
                preset_path=args.style,
                intensity=intensity,
                balance_preserve=balance_preserve,
                source_balance_profile=args.source_balance_profile,
                reference=ref,
                blend=args.blend,
                reverb_match=args.reverb_match,
                reverb_amount=args.reverb_amount,
                target_lufs=args.target_lufs,
                true_peak_db=args.true_peak,
                collect=collected,
                capture_groups=capture_groups,
            )
        render.write_wav24(mix_path, mix, int(p["source"]["sample_rate"]))
        if args.render_plan:
            pipeline.note_reference_tone(p, mix, report)
        # Always record the plan that was actually rendered.  In --render-plan
        # mode the previous behaviour skipped this, so the output directory kept
        # no record of what produced the WAV -- and the "plan <path>" line below
        # pointed at a file that did not exist.
        if args.stems:
            p["source"]["stems_dir"] = os.path.abspath(args.stems)
        if args.render_plan and os.path.abspath(plan_path) == os.path.abspath(args.render_plan):
            # re-rendering a plan into its own folder must not overwrite it
            plan_path = os.path.join(args.out, "mixplan.rendered.json")
        planner.validate(p)
        planner.write_plan(p, plan_path)
        say(f"plan written: {plan_path}")
        render.write_report(report_path, p, report, stems)

        if args.export_stems and collected is not None:
            exported = render.write_stem_export(
                args.export_stems, p, collected, report, mix
            )
            say(f"stems exported: {len(exported)} file(s) -> {args.export_stems}")

        say(f"rendered in {time.time() - t1:.1f} s")
        say("")
        say(f"  mix    {mix_path}")
        say(f"  plan   {plan_path}")
        say(f"  report {report_path}")
        say("")
        say(
            f"  {report['lufs']} LUFS (target {report['target_lufs']})   "
            f"{report['true_peak_db']} dBTP (ceiling {report['ceiling_db']})"
        )
        for w in list(p.get("warnings") or []) + list(report.get("warnings") or []):
            say(f"  ! {w}")

        off = abs(report["lufs"] - report["target_lufs"])
        if off > 1.0:
            print(
                f"warning: loudness landed {off:.1f} dB off target "
                f"({report['lufs']} vs {report['target_lufs']})",
                file=sys.stderr,
            )
        if report["true_peak_db"] > report["ceiling_db"] + 0.1:
            print(
                f"warning: true peak {report['true_peak_db']} exceeds ceiling "
                f"{report['ceiling_db']}",
                file=sys.stderr,
            )
        return 0

    except (
        analyze.StemError,
        planner.PlanError,
        render.RenderError,
        refmod.ReferenceError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
