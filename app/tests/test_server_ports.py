"""How the local UI picks the port it starts on.

The App has to come up even when the twenty ports it prefers are taken, and
the URL it prints (and opens) has to be the address it is really listening
on -- never ``:0`` and never the last port it merely attempted.

Nothing here binds the real 8765-8784 range: a fake server class stands in
for every case except one, where a genuine OS-assigned socket is bound on
port 0 and closed again.  The session directory is redirected into the test's
own tmp dir, so the real global one is never touched.
"""

from __future__ import annotations

import errno
import re
import types

import pytest

from automix import server as srvmod

HOST = "127.0.0.1"
URL_RE = re.compile(r"automix UI\s+->\s+(\S+)")


# --------------------------------------------------------------------- doubles

class _FakeServer:
    """A ThreadingHTTPServer stand-in that records what it was asked to bind."""

    def __init__(self, address, handler):
        self.server_address = address
        self.handler = handler
        self.session_root = None
        self.served = False
        self.closed = False

    def serve_forever(self, poll_interval=0.5):
        self.served = True

    def server_close(self):
        self.closed = True


class _FakeFactory:
    """Scriptable stand-in for the server class the app binds with.

    ``occupied`` answers "already in use" for the given ports, ``error`` fails
    every bind the way a permission error would, and ``ephemeral_error``
    fails only the ``(host, 0)`` bind.
    """

    def __init__(self, occupied=(), ephemeral_port=54321, error=None, ephemeral_error=None):
        self.occupied = set(occupied)
        self.ephemeral_port = ephemeral_port
        self.error = error
        self.ephemeral_error = ephemeral_error
        self.calls = []
        self.instances = []

    def __call__(self, address, handler):
        host, port = address
        self.calls.append((host, port))
        if port == 0 and self.ephemeral_error is not None:
            raise self.ephemeral_error
        if self.error is not None:
            raise self.error
        if port in self.occupied:
            raise OSError(errno.EADDRINUSE, "Address already in use")
        actual = self.ephemeral_port if port == 0 else port
        instance = _FakeServer((host, actual), handler)
        self.instances.append(instance)
        return instance


class _FakeTimer:
    def __init__(self, delay, function):
        self.delay = delay
        self.function = function
        self.started = False

    def start(self):
        self.started = True
        return self


class _FakeThreading:
    """Only the Timer the app uses to open the browser."""

    def __init__(self):
        self.timers = []

    def Timer(self, delay, function):
        timer = _FakeTimer(delay, function)
        self.timers.append(timer)
        return timer


class _FakeBrowser:
    def __init__(self):
        self.opened = []

    def open(self, url):
        self.opened.append(url)


# -------------------------------------------------------------------- fixtures

@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """Keep the test out of the real global session directory."""
    monkeypatch.setattr(
        srvmod, "tempfile", types.SimpleNamespace(gettempdir=lambda: str(tmp_path))
    )
    monkeypatch.setattr(srvmod, "_prune", lambda root: 0)
    monkeypatch.setattr(srvmod, "_load_sessions", lambda root: 0)
    return tmp_path


@pytest.fixture
def fake_world(monkeypatch):
    """Fake the browser and the timer that opens it."""
    browser = _FakeBrowser()
    threading = _FakeThreading()
    monkeypatch.setattr(srvmod, "webbrowser", browser)
    monkeypatch.setattr(srvmod, "threading", threading)
    return threading, browser


def _published_url(capsys):
    out = capsys.readouterr().out
    match = URL_RE.search(out)
    assert match, f"no published URL in {out!r}"
    return match.group(1)


# ----------------------------------------------------------------------- tests

def test_the_first_free_preferred_port_is_used(isolated, fake_world, monkeypatch, capsys):
    factory = _FakeFactory(occupied={8765, 8766})
    monkeypatch.setattr(srvmod, "ThreadingHTTPServer", factory)

    srvmod.serve(HOST, 8765, open_browser=False)

    assert factory.calls == [(HOST, 8765), (HOST, 8766), (HOST, 8767)]
    assert _published_url(capsys) == "http://127.0.0.1:8767/"
    # the server that was bound is the one that serves, and it is closed again
    assert len(factory.instances) == 1
    assert factory.instances[0].served
    assert factory.instances[0].closed


def test_all_twenty_preferred_ports_busy_falls_back_to_an_ephemeral_port(
    isolated, fake_world, monkeypatch, capsys
):
    factory = _FakeFactory(occupied=set(range(8765, 8785)), ephemeral_port=45678)
    monkeypatch.setattr(srvmod, "ThreadingHTTPServer", factory)

    srvmod.serve(HOST, 8765, open_browser=False)

    assert factory.calls[:20] == [(HOST, port) for port in range(8765, 8785)]
    assert factory.calls[20] == (HOST, 0)
    assert len(factory.calls) == 21
    # the published URL is the ephemeral port, not the last one it tried
    assert _published_url(capsys) == "http://127.0.0.1:45678/"
    assert factory.instances[0].served
    assert factory.instances[0].closed


def test_an_explicit_port_zero_publishes_the_port_the_os_gave(
    isolated, fake_world, monkeypatch, capsys
):
    factory = _FakeFactory(ephemeral_port=45679)
    monkeypatch.setattr(srvmod, "ThreadingHTTPServer", factory)

    srvmod.serve(HOST, 0, open_browser=False)

    assert factory.calls == [(HOST, 0)]
    url = _published_url(capsys)
    assert url == "http://127.0.0.1:45679/"
    assert ":0/" not in url


def test_ports_near_the_top_are_never_scanned_past_65535(
    isolated, fake_world, monkeypatch, capsys
):
    factory = _FakeFactory(occupied=set(range(65530, 65536)), ephemeral_port=45680)
    monkeypatch.setattr(srvmod, "ThreadingHTTPServer", factory)

    srvmod.serve(HOST, 65530, open_browser=False)

    assert factory.calls[:6] == [(HOST, port) for port in range(65530, 65536)]
    assert all(0 <= port <= 65535 for _, port in factory.calls)
    assert _published_url(capsys) == "http://127.0.0.1:45680/"


def test_the_last_valid_port_is_tried_on_its_own(isolated, fake_world, monkeypatch, capsys):
    factory = _FakeFactory(occupied={65535}, ephemeral_port=45681)
    monkeypatch.setattr(srvmod, "ThreadingHTTPServer", factory)

    srvmod.serve(HOST, 65535, open_browser=False)

    assert factory.calls[:2] == [(HOST, 65535), (HOST, 0)]
    assert all(0 <= port <= 65535 for _, port in factory.calls)
    assert _published_url(capsys) == "http://127.0.0.1:45681/"


@pytest.mark.parametrize("port", [-1, 65536, 100000])
def test_an_invalid_port_is_refused(isolated, fake_world, monkeypatch, port):
    factory = _FakeFactory()
    monkeypatch.setattr(srvmod, "ThreadingHTTPServer", factory)

    with pytest.raises(SystemExit):
        srvmod.serve(HOST, port)

    assert factory.calls == []


def test_a_permission_error_is_not_a_busy_port(isolated, fake_world, monkeypatch, capsys):
    factory = _FakeFactory(error=OSError(errno.EACCES, "Permission denied"))
    monkeypatch.setattr(srvmod, "ThreadingHTTPServer", factory)
    threading, browser = fake_world

    with pytest.raises(SystemExit):
        srvmod.serve(HOST, 8765, open_browser=True)

    # one attempt, not twenty: this is a failure, not a busy range
    assert factory.calls == [(HOST, 8765)]
    assert threading.timers == []
    assert browser.opened == []
    assert "automix UI" not in capsys.readouterr().out


def test_a_failed_ephemeral_bind_is_reported(isolated, fake_world, monkeypatch, capsys):
    factory = _FakeFactory(
        occupied=set(range(8765, 8785)),
        ephemeral_error=OSError(errno.EADDRINUSE, "Address already in use"),
    )
    monkeypatch.setattr(srvmod, "ThreadingHTTPServer", factory)
    threading, browser = fake_world

    with pytest.raises(SystemExit):
        srvmod.serve(HOST, 8765, open_browser=True)

    assert factory.calls[-1] == (HOST, 0)
    assert threading.timers == []
    assert browser.opened == []
    assert "automix UI" not in capsys.readouterr().out


def test_the_browser_opens_the_published_url(isolated, fake_world, monkeypatch, capsys):
    factory = _FakeFactory(occupied=set(range(8765, 8785)), ephemeral_port=45682)
    monkeypatch.setattr(srvmod, "ThreadingHTTPServer", factory)
    threading, browser = fake_world

    srvmod.serve(HOST, 8765, open_browser=True)

    url = _published_url(capsys)
    assert url == "http://127.0.0.1:45682/"
    assert len(threading.timers) == 1
    threading.timers[0].function()
    assert browser.opened == [url]


def test_a_real_ephemeral_socket_is_bound_published_and_closed(isolated, monkeypatch, capsys):
    """One genuine bind: the OS picks the port and the URL has to name it."""
    instances = []

    class _RealServer(srvmod.ThreadingHTTPServer):
        def serve_forever(self, poll_interval=0.5):
            instances.append(self)

    monkeypatch.setattr(srvmod, "ThreadingHTTPServer", _RealServer)

    srvmod.serve(HOST, 0, open_browser=False)

    assert len(instances) == 1
    port = instances[0].server_address[1]
    assert port != 0
    assert _published_url(capsys) == f"http://{HOST}:{port}/"
    # serve() closed the socket again on the way out
    assert instances[0].socket.fileno() == -1


def test_binding_never_waits_on_a_reverse_dns_lookup(monkeypatch):
    """The standard server looks up a name for 127.0.0.1 while it binds; on a
    Mac whose mDNS stalls that held the App on "Starting" for ~30 seconds."""
    import socket

    def no_lookup(*_args, **_kwargs):
        raise AssertionError("server_bind must not resolve the bound address")

    monkeypatch.setattr(socket, "getfqdn", no_lookup)
    monkeypatch.setattr(socket, "gethostbyaddr", no_lookup)
    httpd = srvmod.ThreadingHTTPServer((HOST, 0), srvmod.Handler)
    try:
        assert httpd.server_address[1] != 0
        assert httpd.server_name == HOST
    finally:
        httpd.server_close()
