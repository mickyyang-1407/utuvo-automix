"""Double-clickable entry point for the packaged app.

`serve()` blocks forever and opens the default browser once the socket is up,
which is exactly what a windowed bundle needs: no console, no arguments.

Two things that are easy to get wrong and both silently break the bundle:
* ``multiprocessing.freeze_support()`` must run before anything spawns a child,
  otherwise a frozen render child re-executes this whole file and starts a
  second server instead of the worker ("render process died without reporting").
* ``AUTOMIX_NO_BROWSER=1`` suppresses the browser for the build check.

On top of the normal App there are two developer modes: ``--validate-plan PATH``
checks a saved plan with the same preflight the command line uses and exits,
and ``--cli ...`` hands the rest of the arguments to the real command line.
"""

import multiprocessing
import os
import signal
import sys
import threading
import time

# Before any argument is looked at and before anything can spawn: a frozen
# child re-executes this file, and freeze_support is what tells it to behave
# like a worker instead of starting a second server.
multiprocessing.freeze_support()

from automix import plan as planner
from automix.cli import _load_saved_plan
from automix.server import serve

USAGE = "usage: Automix [--validate-plan PATH]"


def _say(text: str, error: bool = False) -> None:
    """Print a line, even in a windowed bundle where stdout is None."""
    stream = sys.stderr if error else sys.stdout
    line = f"{text}\n"
    if stream is None:
        try:
            os.write(2 if error else 1, line.encode("utf-8", "replace"))
        except OSError:
            pass
        return
    stream.write(line)
    stream.flush()


def _validate_plan(path: str) -> int:
    """Check a saved plan and say what is wrong with it.

    This is the command line's saved-plan preflight, unchanged: the file has
    to be readable JSON holding an object, and the plan itself has to satisfy
    the shipped schema and the engine's invariants.  Nothing is rendered, no
    server is started and nothing is written.

    The refusal is reported with the reason the central validator gave, as
    it is: a plan that breaks the schema says so, and an installation that
    cannot run the validator says *that* -- a valid plan is never called
    invalid because the machine is missing a package.
    """
    try:
        saved = _load_saved_plan(path)
    except planner.PlanError as exc:
        message = str(exc).replace("--render-plan:", "--validate-plan:", 1)
        _say(f"plan check failed: {message}", error=True)
        return 2
    except Exception as exc:  # noqa: BLE001 - a refusal is a refusal
        _say(f"plan check failed: {exc}", error=True)
        return 2
    tracks = saved.get("tracks") or []
    _say(f"plan is valid: {path}")
    _say(
        f"  schema_version {saved.get('schema_version')}, "
        f"{len(tracks)} track(s), style {(saved.get('style') or {}).get('id')}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # The command line is reached only when --cli is the FIRST argument: the
    # rest of the arguments then belong to the real CLI, parser and all.
    # It is checked before the frozen worker's flag below, so a user's own
    # arguments are never taken for the machinery talking to itself.
    if argv and argv[0] == "--cli":
        return _run_cli(argv[1:])
    # A multiprocessing helper re-executes this binary with its own flags.
    # They are the machinery talking to itself, not a user's arguments: they
    # must not start a second server and must not be rejected as unknown.
    if "--multiprocessing-fork" in argv:
        return 0
    if not argv:
        _attach_to_shell()
        serve(open_browser=os.environ.get("AUTOMIX_NO_BROWSER") != "1")
        return 0
    if len(argv) == 2 and argv[0] == "--validate-plan":
        return _validate_plan(argv[1])
    _say(f"unknown argument: {' '.join(argv)}\n{USAGE}", error=True)
    return 2


def _attach_to_shell() -> None:
    """Belong to the native window that started this engine, if one did.

    The window (macos-shell/main.swift) sets AUTOMIX_SHELL_PID, reads the
    address the server prints, and starts us as the leader of our own process
    group.  Two things follow from that:

    * stdout must be line-buffered: a windowed bundle can have no sys.stdout
      at all, and a pipe is block-buffered, so the address would never arrive.
    * when the window goes away -- quit, crash or Force Quit -- the server and
      every render child it started must go too.  Quit sends SIGTERM; a crash
      or Force Quit is noticed by polling the parent.  Either way a running
      publication is allowed to finish, then the whole group is ended.

    Without AUTOMIX_SHELL_PID nothing changes.
    """
    shell = os.environ.get("AUTOMIX_SHELL_PID", "")
    if not shell.isdigit():
        return
    shell_pid = int(shell)
    try:
        sys.stdout = os.fdopen(os.dup(1), "w", encoding="utf-8", errors="replace", buffering=1)
    except OSError:
        pass

    def end() -> None:
        # a publication or a project copy that is running finishes first
        # (automix.server.begin_shutdown); render children are just stopped
        from automix import server as srvmod
        try:
            srvmod.begin_shutdown(timeout=60.0)
        finally:
            try:
                if os.getpgrp() == os.getpid():
                    os.killpg(os.getpgrp(), signal.SIGKILL)
            finally:
                os._exit(0)

    def watch() -> None:
        while os.getppid() == shell_pid:
            time.sleep(1.0)
        end()

    # Quit in the window sends SIGTERM to this process only
    signal.signal(signal.SIGTERM, lambda _signum, _frame: end())
    threading.Thread(target=watch, name="shell-watch", daemon=True).start()


class _DiscardStream:
    """A text stream with nowhere to go.

    A windowed bundle can have neither a stream nor a usable descriptor.
    The command line still runs and still reports its exit code -- it just
    cannot print.
    """

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True


def _text_stream(current, fd: int):
    """Return a usable UTF-8 text stream for ``fd``.

    A windowed bundle can leave sys.stdout set to None while the process
    still owns fd 1 -- that is what happens when the App is started with
    pipes.  The caller's descriptor is never closed: it is duplicated
    first and only the duplicate is wrapped.
    """
    if current is not None and hasattr(current, "write"):
        return current
    try:
        duplicate = os.dup(fd)
    except OSError:
        return _DiscardStream()
    try:
        return os.fdopen(duplicate, "w", encoding="utf-8", errors="replace")
    except OSError:
        os.close(duplicate)
        return _DiscardStream()


def _run_cli(argv: list[str]) -> int:
    """Run the existing command line and hand back its exit code.

    Nothing is re-implemented here: the parser, the preflight and the audio
    logic all stay in automix.cli, so the packaged App and the python module
    answer the same way.  What this adds is a place to stand for a windowed
    bundle, where sys.stdout and sys.stderr can be None even though the
    process still owns usable file descriptors.
    """
    from automix import cli as climod

    # A bare --cli shows the command line's own help and stops: no server,
    # no browser, no output folder.
    args = list(argv) if argv else ["--help"]
    previous_out, previous_err = sys.stdout, sys.stderr
    out_stream = _text_stream(previous_out, 1)
    err_stream = _text_stream(previous_err, 2)
    sys.stdout, sys.stderr = out_stream, err_stream
    try:
        return climod.main(args, prog="Automix --cli")
    except SystemExit as exc:
        # argparse ends the process itself for --help and for a bad option.
        # Those codes are the command line's own and have to survive.
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        _say(str(code), error=True)
        return 2
    finally:
        sys.stdout, sys.stderr = previous_out, previous_err
        for stream, previous in ((out_stream, previous_out), (err_stream, previous_err)):
            if stream is previous:
                continue
            try:
                stream.close()
            except (OSError, ValueError):
                pass


if __name__ == "__main__":
    sys.exit(main())
