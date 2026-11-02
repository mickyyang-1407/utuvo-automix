"""Quitting the native window must not cut a publication in half.

The window sends the engine SIGTERM; app.py then calls begin_shutdown(), which
has to wait for a publication or project copy that is running and refuse to
start a new one.  Nothing here touches a real session: the guarded function
is a stand-in.
"""

from __future__ import annotations

import threading
import time

import pytest

from automix import server as srvmod


@pytest.fixture(autouse=True)
def fresh_guard(monkeypatch):
    monkeypatch.setattr(srvmod, "_guarded_steps", 0)
    monkeypatch.setattr(srvmod, "_shutting_down", False)
    monkeypatch.setattr(srvmod, "_guarded_cond", threading.Condition())


def test_shutdown_waits_for_a_running_step_and_refuses_new_ones():
    started, release, finished = threading.Event(), threading.Event(), []

    @srvmod._guarded
    def publish():
        started.set()
        release.wait(5)
        finished.append(time.monotonic())

    worker = threading.Thread(target=publish)
    worker.start()
    assert started.wait(5)

    done = []
    waiter = threading.Thread(target=lambda: done.append((srvmod.begin_shutdown(5), time.monotonic())))
    waiter.start()
    time.sleep(0.2)
    assert not done, "shutdown returned while a publication was still running"

    release.set()
    worker.join(5)
    waiter.join(5)
    assert done and done[0][0] is True
    assert done[0][1] >= finished[0]

    with pytest.raises(srvmod.ShuttingDown):
        publish()


def test_shutdown_gives_up_after_its_timeout():
    started, release = threading.Event(), threading.Event()

    @srvmod._guarded
    def stuck():
        started.set()
        release.wait(5)

    worker = threading.Thread(target=stuck)
    worker.start()
    assert started.wait(5)
    try:
        assert srvmod.begin_shutdown(0.2) is False
    finally:
        release.set()
        worker.join(5)


def test_the_three_publishing_steps_are_guarded():
    for name in ("_publish_render", "_save_project", "_open_project"):
        assert getattr(srvmod, name).__wrapped__ is not None, name


def test_a_quit_render_leaves_nothing_once_it_is_old_enough(tmp_path):
    import os

    sess = tmp_path / "abc123"
    (sess / "stems").mkdir(parents=True)
    old_stage = sess / (srvmod.STAGE_PREFIX + "old")
    young_stage = sess / (srvmod.STAGE_PREFIX + "young")
    backup = sess / (srvmod.BACKUP_PREFIX + "keep")
    for d in (old_stage, young_stage, backup):
        d.mkdir()
        (d / "mix.wav").write_bytes(b"x")
    old_part = sess / "stems" / "vocal.wav.1234abcd.part"
    old_part.write_bytes(b"x")
    stem = sess / "stems" / "vocal.wav"
    stem.write_bytes(b"x")
    long_ago = time.time() - 3600
    for p in (old_stage, old_part, backup, stem):
        os.utime(p, (long_ago, long_ago))

    assert srvmod._sweep_interrupted(str(tmp_path)) == 2
    assert not old_stage.exists() and not old_part.exists()
    # a render that may still be running, a rollback copy and a real stem stay
    assert young_stage.exists() and backup.exists() and stem.exists()


def test_a_manifest_is_replaced_whole_or_not_at_all(tmp_path, monkeypatch):
    import json

    path = tmp_path / "session.json"
    path.write_text(json.dumps({"old": True}))

    def broken_replace(_src, _dst):
        raise OSError("disk full")

    monkeypatch.setattr(srvmod.os, "replace", broken_replace)
    with pytest.raises(OSError):
        srvmod._atomic_write(str(path), b'{"new": true}')
    assert json.loads(path.read_text()) == {"old": True}
    assert [p.name for p in tmp_path.iterdir()] == ["session.json"]
