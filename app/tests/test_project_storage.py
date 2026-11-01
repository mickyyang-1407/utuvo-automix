"""Explicit project copies survive disposable sessions; synthetic, local only."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
import wave
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from automix import server as srv


CONTROLS = {
    "style": "rock", "intensity": 0.8,
    "source_balance_profile": "balanced_stems", "balance_preserve": 0.35,
    "blend": 0.7, "target_lufs": -20.0, "export_stems": True,
    "reverb_amount": 0.5,
}
DRAFT = dict(CONTROLS, intensity=0.0, balance_preserve=None, blend=0.0,
             target_lufs=None, reverb_amount=0.0, export_stems=False)


def _hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(directory).rglob("*") if p.is_file()}


def _session(root):
    sess = srv.Session(str(root))
    for name in ("kick.wav", "room.wav"):
        with wave.open(str(Path(sess.stems_dir) / name), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(48000)
            out.writeframes(b"\x00\x01" * 480)
    for name in ("reference.wav", "mix.wav", "raw.wav", "monitor_mix.wav", "monitor_raw.wav"):
        (Path(sess.dir) / name).write_bytes((Path(sess.stems_dir) / "kick.wav").read_bytes())
    sess.reference_path = str(Path(sess.dir) / "reference.wav")
    sess.mix_path = str(Path(sess.dir) / "mix.wav")
    sess.raw_path = str(Path(sess.dir) / "raw.wav")
    sess.monitor_mix_path = str(Path(sess.dir) / "monitor_mix.wav")
    sess.monitor_raw_path = str(Path(sess.dir) / "monitor_raw.wav")
    sess.result = {"style": {"id": "rock"}, "monitoring": {"status": "matched"}}
    sess.roles = {"kick.wav": "kick", "room.wav": "room"}
    sess.role_revision = 3
    sess.trims = {"kick.wav": 1.25}
    sess.pans = {"room.wav": 0.0}
    sess.muted = {"room.wav"}
    sess.capture_groups = [{"id": "kit", "members": ["kick.wav", "room.wav"], "pan": 0.0}]
    sess.controls = dict(CONTROLS)
    sess.reverb_amount = 0.5
    (Path(sess.dir) / "result.json").write_text(json.dumps(sess.result))
    (Path(sess.dir) / "mixplan.json").write_text(json.dumps({
        "source": {"stems_dir": sess.stems_dir, "sample_rate": 48000},
        "tracks": [{"id": "kick.wav", "gain_db": 1.234, "pan": 0.0}],
        "style": {"id": "rock"},
    }))
    sess.save()
    return sess


def test_snapshot_roundtrip_after_source_prune_and_new_server_memory(tmp_path):
    sessions = tmp_path / "sessions"
    projects = tmp_path / "projects"
    sess = _session(sessions)
    original = _hashes(sess.dir)
    saved = srv._save_project(sess, str(projects), "曲 / <A>", DRAFT)
    saved_dir = projects / saved["id"]
    saved_hashes = _hashes(saved_dir)
    assert _hashes(sess.dir) == original
    assert saved_dir.name != saved["name"]
    assert srv._list_projects(str(projects)) == [
        {k: saved[k] for k in ("id", "name", "saved_at")}
    ]
    old = time.time() - 13 * 3600
    os.utime(sess.dir, (old, old))
    assert srv._prune(str(sessions)) == 1
    assert not Path(sess.dir).exists()
    with srv._lock:
        srv._sessions.pop(sess.id, None)
    reopened = srv._open_project(str(projects), saved["id"], str(sessions))
    assert reopened.id not in {sess.id, saved["id"]}
    assert reopened.project_name == saved["name"]
    assert reopened.controls == CONTROLS
    assert reopened.draft_controls == DRAFT
    assert reopened.roles == sess.roles
    assert reopened.role_revision == 3
    assert reopened.trims == sess.trims
    assert reopened.pans == sess.pans
    assert reopened.muted == sess.muted
    assert reopened.capture_groups == sess.capture_groups
    assert reopened.decode_policy == sess.decode_policy
    assert reopened.monitoring_state()["status"] == "matched"
    restored_hashes = _hashes(reopened.dir)
    for name, digest in original.items():
        if name not in {"mixplan.json", "session.json"}:
            assert restored_hashes[name] == digest
    original_plan = json.loads((saved_dir / "mixplan.json").read_text())
    reopened_plan = json.loads((Path(reopened.dir) / "mixplan.json").read_text())
    assert original_plan["source"].pop("stems_dir") == str(saved_dir / "stems")
    assert reopened_plan["source"].pop("stems_dir") == reopened.stems_dir
    assert original_plan == reopened_plan
    # The editable copy never writes through into the saved snapshot.
    reopened.trims["kick.wav"] = -2.0
    reopened.save()
    assert _hashes(saved_dir) == saved_hashes
    assert srv._prune(str(sessions)) == 0


def test_project_storage_cannot_overlap_disposable_session_root(tmp_path):
    sessions = tmp_path / "sessions"
    sess = _session(sessions)
    original = _hashes(sess.dir)
    for root in (sessions, sessions / "saved"):
        with pytest.raises(ValueError, match="outside disposable sessions"):
            srv._save_project(sess, str(root), "unsafe root", DRAFT)
    assert not (sessions / "saved").exists()
    assert _hashes(sess.dir) == original


def test_invalidated_mix_does_not_revive_from_copied_files(tmp_path):
    sess = _session(tmp_path / "sessions")
    sess.mix_path = None
    sess.invalidate_result()
    saved = srv._save_project(sess, str(tmp_path / "projects"), "invalid", DRAFT)
    opened = srv._open_project(str(tmp_path / "projects"), saved["id"],
                               str(tmp_path / "sessions"))
    assert (Path(opened.dir) / "mix.wav").exists()  # recovery bytes kept
    assert not opened.has_mix()
    assert opened.result is None
    assert opened.monitoring_pair() == (None, None)
    assert opened.draft_controls == DRAFT


@pytest.mark.parametrize("boundary", ["copy", "manifest", "publish"])
def test_failed_save_keeps_old_project_source_and_no_partial(tmp_path, monkeypatch, boundary):
    sess = _session(tmp_path / "sessions")
    projects = tmp_path / "projects"
    srv._save_project(sess, str(projects), "first", DRAFT)
    original = _hashes(sess.dir)
    saved = _hashes(projects)
    def fail(*args, **kwargs):
        raise OSError("simulated disk full")
    if boundary == "copy":
        monkeypatch.setattr(srv, "_copy_project_tree", fail)
    elif boundary == "manifest":
        monkeypatch.setattr(sess, "write_manifest", fail)
    else:
        monkeypatch.setattr(srv.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        srv._save_project(sess, str(projects), "second", DRAFT)
    assert _hashes(sess.dir) == original
    assert _hashes(projects) == saved
    assert not any(p.name.startswith(".project-stage-") for p in projects.iterdir())
    assert len(list(projects.iterdir())) == 1


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_symlink_not_copied_or_dereferenced(tmp_path, kind):
    sess = _session(tmp_path / "sessions")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.wav").write_bytes(b"outside")
    target = outside / "private.wav" if kind == "file" else outside
    (Path(sess.dir) / "link").symlink_to(target, target_is_directory=kind == "directory")
    projects = tmp_path / "projects"
    with pytest.raises(ValueError, match="links"):
        srv._save_project(sess, str(projects), "bad", DRAFT)
    assert list(projects.iterdir()) == []
    assert (outside / "private.wav").read_bytes() == b"outside"


def test_invalid_id_controls_and_reference_are_rejected(tmp_path):
    sess = _session(tmp_path / "sessions")
    projects = tmp_path / "projects"
    for project_id in ("..", "../outside", "a" * 13, "A" * 12, "%2e%2e"):
        with pytest.raises(ValueError, match="invalid project id"):
            srv._open_project(str(projects), project_id, str(tmp_path / "sessions"))
    with pytest.raises(ValueError, match="controls"):
        srv._save_project(sess, str(projects), "bad", dict(DRAFT, intensity=True))
    assert not projects.exists()
    external = tmp_path / "outside.wav"
    external.write_bytes(b"outside")
    sess.reference_path = str(external)
    with pytest.raises(ValueError, match="reference"):
        srv._save_project(sess, str(projects), "bad", DRAFT)
    assert list(projects.iterdir()) == []


@pytest.mark.parametrize("fail_publication", [False, True])
def test_new_success_controls_replace_opened_draft_only_on_success(tmp_path, monkeypatch, fail_publication):
    sess = _session(tmp_path / "sessions")
    sess.draft_controls = dict(DRAFT)
    sess.save()
    stage = Path(srv._new_staging_dir(sess.dir))
    for name in ("mix.wav", "raw.wav", "mixplan.json", "report.md"):
        (stage / name).write_bytes(b"new " + name.encode())
    updated = dict(CONTROLS, intensity=0.625, target_lufs=-18.0)
    if fail_publication:
        original = srv._Publication.publish
        def fail(self, name):
            if name == "session.json":
                raise OSError("publish failed")
            return original(self, name)
        monkeypatch.setattr(srv._Publication, "publish", fail)
        with pytest.raises(srv.PublishError):
            srv._publish_render(sess, str(stage), {"monitoring": {}}, updated, None, 0.5)
    else:
        srv._publish_render(sess, str(stage), {"monitoring": {}}, updated, None, 0.5)
    loaded = srv.Session.load(str(tmp_path / "sessions"), sess.id)
    assert loaded.controls == (CONTROLS if fail_publication else updated)
    assert loaded.draft_controls == (DRAFT if fail_publication else None)
    assert sess.draft_controls == loaded.draft_controls


@pytest.fixture
def http(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    projects = tmp_path / "projects"
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
    httpd.session_root = str(sessions)
    httpd.project_root = str(projects)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", sessions, projects
    finally:
        httpd.shutdown()
        httpd.server_close()
        with srv._lock:
            for sid, session in list(srv._sessions.items()):
                if Path(session.dir).parent == sessions:
                    del srv._sessions[sid]


def _request(base, path, body=None):
    req = urllib.request.Request(base + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def test_http_list_lazy_save_open_state_and_busy_rejection(http):
    base, sessions, projects = http
    code, result = _request(base, "/api/projects")
    assert code == 200 and result["projects"] == []
    assert not projects.exists()
    sess = _session(sessions)
    with srv._lock:
        srv._sessions[sess.id] = sess
    path = f"/api/session/{sess.id}/project"
    sess.busy = True
    assert _request(base, path, {"name": "busy", "controls": DRAFT})[0] == 409
    assert not projects.exists()
    sess.busy = False
    code, saved = _request(base, path, {"name": "saved", "controls": DRAFT})
    assert code == 200
    code, opened = _request(base, f"/api/projects/{saved['id']}/open", {})
    assert code == 200
    code, state = _request(base, f"/api/session/{opened['id']}/state")
    assert code == 200 and state["project_name"] == "saved"
    assert state["controls"] == CONTROLS and state["draft_controls"] == DRAFT
    assert state["has_mix"] is True
