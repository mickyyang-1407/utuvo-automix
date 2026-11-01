"""Smoke test for the packaged App: it must answer, and it must refuse.

The App is asked two things.  First the three plans it has always checked:
a valid one, the same plan with a field out of the schema's range, and the
same plan with a NaN in it.  The first must exit 0, the other two exit 2.
Then its own command line: ``--cli --help`` and ``--cli --list-styles`` must
both exit 0 and print what they promise, not a server and not a traceback.

No server is started, no browser is opened, no audio is rendered and nothing
outside the fixture folder is written.
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from automix import analyze, plan as planner  # noqa: E402  (the path is set above)

SR = 48000


def _tone(name: str, dur: float = 2.0, freq: float = 220.0):
    """A short synthetic tone -- ours, not borrowed from anywhere."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = 0.2 * np.sin(2 * np.pi * freq * t)
    st = analyze.Stem(
        name=name,
        path=name,
        audio=np.stack([x, x], axis=1).astype(np.float32),
        sr=SR,
    )
    analyze.detect_roles([st])
    analyze.measure(st)
    return st


def build_fixtures(directory: str):
    """Write the three plans into ``directory``; return (path, expected exit)."""
    stems = [_tone("vocal_lead.wav", 2.0, 220.0), _tone("guitar_elec.wav", 2.0, 110.0)]
    analyze.align(stems)
    for st in stems:
        analyze.measure(st)
    plan = planner.build_plan(stems, planner.load_preset("pop_2020s"))

    valid = os.path.join(directory, "valid-plan.json")
    out_of_range = os.path.join(directory, "out-of-range-plan.json")
    nan_plan = os.path.join(directory, "nan-plan.json")

    planner.write_plan(plan, valid)

    # the same plan, one field outside the schema's range
    broken = copy.deepcopy(plan)
    broken["tracks"][0]["pan"] = 5.0
    planner.write_plan(broken, out_of_range)

    # the same plan again, with a NaN in a number (JSON has no NaN, but the
    # parser accepts it -- which is exactly why the guard exists)
    broken = copy.deepcopy(plan)
    broken["tracks"][0]["gain_db"] = float("nan")
    planner.write_plan(broken, nan_plan)

    return [(valid, 0), (out_of_range, 2), (nan_plan, 2)]


def run_checks(binary: str, cases) -> list:
    """Run the App against the fixtures; return the list of failures."""
    failures = []
    for path, expected in cases:
        label = os.path.basename(path)
        try:
            proc = subprocess.run(
                [binary, "--validate-plan", path],
                capture_output=True,
                text=True,
                timeout=300,
            )
        except OSError as exc:
            failures.append(f"{label}: cannot run {binary}: {exc}")
            continue
        print(f"  {label}: exit {proc.returncode} (expected {expected})")
        if proc.returncode != expected:
            failures.append(
                f"{label}: exit {proc.returncode}, expected {expected}\n"
                f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
            )
        output = f"{proc.stdout}\n{proc.stderr}"
        if output.strip() and "Traceback" in output:
            failures.append(f"{label}: the App printed a traceback:\n{output}")
    return failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="build smoke test for Automix.app")
    ap.add_argument("--dir", required=True, help="folder the fixtures are written to")
    ap.add_argument("--binary", default=None, help="the App to smoke test")
    args = ap.parse_args(argv)

    os.makedirs(args.dir, exist_ok=True)
    cases = build_fixtures(args.dir)
    if not args.binary:
        for path, _ in cases:
            print(f"  fixture: {path}")
        return 0

    print(f"smoke testing {args.binary} ...")
    failures = run_checks(args.binary, cases)
    failures += run_cli_checks(args.binary)
    if failures:
        print("smoke test failed:", file=sys.stderr)
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    print(
        "smoke test ok: valid plan accepted, out-of-range and NaN plans refused, "
        "the packaged command line answers"
    )
    return 0


def run_cli_checks(binary: str) -> list:
    """Ask the packaged App for its own command line; return the failures.

    The App has to answer the same command line the python module answers:
    ``--cli --help`` prints the usage with the App's own name in it, and
    ``--cli --list-styles`` prints the styles the bundle carries.  Both must
    exit 0.  A server that never answers, or a traceback, is a failure.
    """
    failures = []
    checks = [
        (["--cli", "--help"], "usage: Automix --cli"),
        (["--cli", "--list-styles"], "pop_2020s"),
    ]
    for args, expected in checks:
        label = " ".join(args)
        try:
            proc = subprocess.run(
                [binary, *args],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            failures.append(
                f"{label}: no answer within 120 s -- the App started a server "
                "instead of answering"
            )
            continue
        except OSError as exc:
            failures.append(f"{label}: cannot run {binary}: {exc}")
            continue
        print(f"  {label}: exit {proc.returncode} (expected 0)")
        if proc.returncode != 0:
            failures.append(
                f"{label}: exit {proc.returncode}, expected 0\n"
                f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
            )
        if expected not in proc.stdout:
            failures.append(
                f"{label}: the App did not print {expected!r}\n"
                f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
            )
        output = f"{proc.stdout}\n{proc.stderr}"
        if "Traceback" in output:
            failures.append(f"{label}: the App printed a traceback:\n{output}")
    return failures


if __name__ == "__main__":
    sys.exit(main())
