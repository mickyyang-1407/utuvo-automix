"""The App's own command line entry: ``Automix --cli ...``.

The App has always been the GUI, and the python module has always been the
command line.  This covers the third door: the same command line, reached
through the App's own entry point, with the exit codes, the refusals and the
preflight it has always had.

Everything here is synthetic.  The plans are built from two short tones made
here, the output folders belong to the test, and the packaged App is never
built or launched: the entry point is exercised as a python process, which is
the same file the bundle runs.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import app  # noqa: E402  (the path is set above)
import build_smoke  # noqa: E402  (the path is set above)
from automix import cli as climod  # noqa: E402  (the path is set above)

#: the repository root: the folder that holds ``presets/`` and ``schemas/``
REPO = os.path.dirname(HERE)


# ---------------------------------------------------------------- helpers


def run_app(*args, cwd=HERE):
    """Run the App's entry point the way the bundle runs it."""
    return subprocess.run(
        [sys.executable, "-B", os.path.join(HERE, "app.py"), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
    )


def _capture_fd(fd: int, action) -> str:
    """Run ``action`` with ``fd`` pointed at a pipe; return what it wrote.

    This is the only honest way to see what the CLI wrote while
    ``sys.stdout`` is None: the stream objects are gone, the descriptor is
    not, and the descriptor is what the caller reads.
    """
    read_fd, write_fd = os.pipe()
    saved = os.dup(fd)
    try:
        os.dup2(write_fd, fd)
        action()
    finally:
        os.dup2(saved, fd)
        os.close(saved)
        os.close(write_fd)
    chunks = []
    while True:
        data = os.read(read_fd, 65536)
        if not data:
            break
        chunks.append(data)
    os.close(read_fd)
    return b"".join(chunks).decode("utf-8", "replace")


@pytest.fixture
def served(monkeypatch):
    """Record every call to the GUI's server instead of starting one."""
    calls = []

    def fake_serve(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(app, "serve", fake_serve)
    return calls


# ------------------------------------------------------- the public entry


def test_bare_cli_shows_the_command_line_help_and_exits_zero(capsys, served):
    assert app.main(["--cli"]) == 0
    out = capsys.readouterr().out
    assert "usage: Automix --cli" in out
    assert served == []


def test_cli_help_names_the_app_entry_and_exits_zero(capsys, served):
    assert app.main(["--cli", "--help"]) == 0
    out = capsys.readouterr().out
    assert "usage: Automix --cli" in out
    assert "--stems" in out and "--render-plan" in out
    assert served == []


def test_cli_list_styles_prints_the_styles_and_exits_zero(capsys, served):
    assert app.main(["--cli", "--list-styles"]) == 0
    assert "pop_2020s" in capsys.readouterr().out
    assert served == []


def test_an_unknown_cli_option_is_refused_with_exit_two(capsys, served):
    assert app.main(["--cli", "--not-a-flag"]) == 2
    assert "unrecognized arguments" in capsys.readouterr().err
    assert served == []


def test_an_unknown_outer_argument_is_still_refused_with_exit_two(capsys, served):
    assert app.main(["--not-a-flag"]) == 2
    assert "unknown argument" in capsys.readouterr().err
    assert served == []


def test_the_frozen_worker_flag_is_not_swallowed_by_the_cli(capsys, served):
    """A user's own argv is not the machinery talking to itself.

    ``--multiprocessing-fork`` in a user's arguments must be refused like any
    other unknown option, not taken for the frozen worker and answered with
    a silent zero.
    """
    assert app.main(["--cli", "--stems", "somewhere", "--multiprocessing-fork"]) == 2
    assert served == []


def test_the_frozen_worker_still_dispatches_without_serving(served):
    assert app.main(["--multiprocessing-fork"]) == 0
    assert served == []


def test_no_arguments_still_serves_the_gui(monkeypatch, served):
    monkeypatch.delenv("AUTOMIX_NO_BROWSER", raising=False)
    assert app.main([]) == 0
    assert len(served) == 1
    assert served[0][1].get("open_browser") is True


def test_the_build_check_can_still_suppress_the_browser(monkeypatch, served):
    monkeypatch.setenv("AUTOMIX_NO_BROWSER", "1")
    assert app.main([]) == 0
    assert served[0][1].get("open_browser") is False


def test_validate_plan_still_checks_a_saved_plan(capsys, tmp_path, served):
    cases = dict((path, expected) for path, expected in build_smoke.build_fixtures(str(tmp_path)))
    valid = next(p for p in cases if cases[p] == 0)
    broken = next(p for p in cases if cases[p] == 2)

    assert app.main(["--validate-plan", valid]) == 0
    assert "plan is valid" in capsys.readouterr().out

    assert app.main(["--validate-plan", broken]) == 2
    assert "plan check failed" in capsys.readouterr().err
    assert served == []


# ------------------------------------------------- the packaged entry point


def test_the_app_entry_point_runs_the_command_line_as_a_process():
    proc = run_app("--cli", "--help")
    assert proc.returncode == 0, proc.stderr[-800:]
    assert "usage: Automix --cli" in proc.stdout
    assert "Traceback" not in proc.stdout + proc.stderr


def test_the_app_entry_point_refuses_an_unknown_cli_option():
    proc = run_app("--cli", "--not-a-flag")
    assert proc.returncode == 2
    assert "Traceback" not in proc.stdout + proc.stderr


def test_the_app_entry_point_still_refuses_an_unknown_outer_argument():
    proc = run_app("--not-a-flag")
    assert proc.returncode == 2
    assert "unknown argument" in proc.stderr


def test_the_app_entry_point_still_checks_a_saved_plan(tmp_path):
    cases = dict((path, expected) for path, expected in build_smoke.build_fixtures(str(tmp_path)))
    valid = next(p for p in cases if cases[p] == 0)
    proc = run_app("--validate-plan", valid)
    assert proc.returncode == 0, proc.stderr[-800:]
    assert "plan is valid" in proc.stdout


# ------------------------------------------------------------- the streams


def test_the_cli_restores_the_caller_s_streams_and_leaves_the_descriptors_open(capsys):
    """The CLI borrows the streams; it does not keep or close them."""
    out, err = io.StringIO(), io.StringIO()
    previous_out, previous_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        assert app.main(["--cli", "--list-styles"]) == 0
    finally:
        sys.stdout, sys.stderr = previous_out, previous_err

    assert "pop_2020s" in out.getvalue()
    assert sys.stdout is previous_out and sys.stderr is previous_err
    # the caller's own streams still work after the CLI has been through them
    out.write("still usable")
    out.flush()
    assert out.getvalue().endswith("still usable")
    # and so do the descriptors behind them
    os.fstat(1)
    os.fstat(2)


def test_the_cli_writes_to_the_descriptor_when_the_stream_is_missing(capsys):
    """A windowed bundle can have no stdout while fd 1 is still there."""
    previous_out, previous_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = None, None
    try:
        written = _capture_fd(1, lambda: app.main(["--cli", "--list-styles"]))
    finally:
        sys.stdout, sys.stderr = previous_out, previous_err

    assert "pop_2020s" in written
    assert sys.stdout is previous_out and sys.stderr is previous_err
    os.fstat(1)


def test_a_cli_refusal_reaches_the_error_descriptor_when_the_stream_is_missing(capsys):
    previous_out, previous_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = None, None
    try:
        written = _capture_fd(2, lambda: app.main(["--cli", "--not-a-flag"]))
    finally:
        sys.stdout, sys.stderr = previous_out, previous_err

    assert "unrecognized arguments" in written
    os.fstat(2)


CLOSED_DESCRIPTOR_SCRIPT = """
import os
import sys

os.close(1)
os.close(2)
sys.stdout = None
sys.stderr = None
sys.path.insert(0, {mvp!r})
import app

raise SystemExit(app.main({argv!r}))
"""


def _run_without_descriptors(argv):
    script = CLOSED_DESCRIPTOR_SCRIPT.format(mvp=HERE, argv=argv)
    return subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=HERE,
        capture_output=True,
        timeout=600,
    )


def test_a_missing_descriptor_leaves_the_exit_code_intact():
    """No descriptor is not a usable descriptor.

    The output cannot go anywhere, so only the exit code survives -- and it
    has to be the CLI's own: a refusal still refuses, and nothing escapes as
    a traceback (which would exit 1 instead).
    """
    proc = _run_without_descriptors(["--cli", "--not-a-flag"])
    assert proc.returncode == 2, proc.stdout + proc.stderr


def test_a_missing_descriptor_does_not_break_the_command_line():
    proc = _run_without_descriptors(["--cli", "--list-styles"])
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ------------------------------------------------------- the saved plan


def _plan_cases(tmp_path):
    """The three plans the build smoke builds: valid, out of range, NaN."""
    cases = build_smoke.build_fixtures(str(tmp_path))
    by_expectation = {expected: path for path, expected in cases}
    return by_expectation[0], by_expectation[2]


def test_an_invalid_saved_plan_is_refused_before_any_output_is_written(capsys, tmp_path):
    valid, broken = _plan_cases(tmp_path / "plans")
    out_dir = tmp_path / "out"

    assert app.main(["--cli", "--render-plan", broken, "--out", str(out_dir)]) == 2
    assert "error:" in capsys.readouterr().err
    # nothing was created: the refusal came before the output folder
    assert not out_dir.exists()


def test_an_invalid_saved_plan_keeps_the_outputs_that_were_there(capsys, tmp_path):
    valid, broken = _plan_cases(tmp_path / "plans")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    sentinel = out_dir / "mix.wav"
    sentinel.write_bytes(b"the previous mix")

    assert app.main(["--cli", "--render-plan", broken, "--out", str(out_dir)]) == 2
    assert sentinel.read_bytes() == b"the previous mix"
    assert sorted(p.name for p in out_dir.iterdir()) == ["mix.wav"]


def test_a_plan_that_is_not_json_is_refused(capsys, tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("this is not json", encoding="utf-8")
    out_dir = tmp_path / "out"

    assert app.main(["--cli", "--render-plan", str(broken), "--out", str(out_dir)]) == 2
    assert "error:" in capsys.readouterr().err
    assert not out_dir.exists()


def test_a_plan_without_a_schema_is_refused(capsys, tmp_path):
    broken = tmp_path / "no-schema.json"
    broken.write_text(json.dumps({"tracks": []}), encoding="utf-8")
    out_dir = tmp_path / "out"

    assert app.main(["--cli", "--render-plan", str(broken), "--out", str(out_dir)]) == 2
    assert "error:" in capsys.readouterr().err
    assert not out_dir.exists()


# ------------------------------------------------------------ the presets


def test_the_source_tree_default_preset_dir_still_finds_the_shipped_presets():
    assert not hasattr(sys, "_MEIPASS")
    default = climod._default_preset_dir()
    assert os.path.isfile(os.path.join(default, "pop_2020s.json"))


def test_a_packaged_app_lists_the_presets_bundled_next_to_it(monkeypatch, capsys):
    monkeypatch.setattr(sys, "_MEIPASS", REPO, raising=False)
    assert app.main(["--cli", "--list-styles"]) == 0
    assert "pop_2020s" in capsys.readouterr().out


def test_an_explicit_preset_dir_still_wins(tmp_path, capsys):
    empty = tmp_path / "no-presets-here"
    empty.mkdir()
    assert app.main(["--cli", "--list-styles", "--preset-dir", str(empty)]) == 0
    # the given folder is the one that was read: it has no styles in it
    assert capsys.readouterr().out == ""


# ------------------------------------------- the module's own command line


def test_the_module_command_line_still_calls_itself_automix(capsys):
    with pytest.raises(SystemExit) as raised:
        climod.main(["--help"])
    assert raised.value.code == 0
    assert "usage: automix" in capsys.readouterr().out


def test_the_module_command_line_can_be_given_another_name(capsys):
    with pytest.raises(SystemExit) as raised:
        climod.main(["--help"], prog="Automix --cli")
    assert raised.value.code == 0
    assert "usage: Automix --cli" in capsys.readouterr().out
