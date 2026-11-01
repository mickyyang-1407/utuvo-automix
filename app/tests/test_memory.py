"""Allocator policy is optional and cannot change numerical resampling."""
import ctypes
import weakref

import numpy as np
import pytest

from automix import _memory, dsp


@pytest.fixture
def binding(monkeypatch):
    _memory._pressure_relief.cache_clear()
    monkeypatch.setattr(_memory, "_platform", "darwin")
    yield
    _memory._pressure_relief.cache_clear()


class FakeFunction:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def __call__(self, *args):
        self.calls.append(args)
        if self.error:
            raise self.error
        return 12345  # deliberately ignored: not an RSS/freed-byte estimate


def install_library(monkeypatch, function):
    calls = []

    def load(path):
        calls.append(path)
        library = type("Library", (), {})()
        library.malloc_zone_pressure_relief = function
        return library

    monkeypatch.setattr(ctypes, "CDLL", load)
    return calls


@pytest.mark.parametrize("platform", ["linux", "win32", "freebsd"])
def test_unsupported_platform_never_loads_native_library(binding, monkeypatch, platform):
    monkeypatch.setattr(_memory, "_platform", platform)

    def forbidden(*args):
        raise AssertionError("unsupported platform must not bind native API")

    monkeypatch.setattr(ctypes, "CDLL", forbidden)
    assert _memory.reclaim_unused() is None
    assert _memory._pressure_relief.cache_info().misses == 0


def test_native_binding_is_lazy_cached_and_has_exact_abi(binding, monkeypatch):
    native = FakeFunction()
    loads = install_library(monkeypatch, native)
    assert loads == []
    assert _memory.reclaim_unused() is None
    assert _memory.reclaim_unused() is None
    assert loads == ["/usr/lib/libSystem.B.dylib"]
    assert native.argtypes == [ctypes.c_void_p, ctypes.c_size_t]
    assert native.restype is ctypes.c_size_t
    assert native.calls == [(None, 0), (None, 0)]


def test_missing_library_is_cached_noop(binding, monkeypatch):
    attempts = []

    def unavailable(path):
        attempts.append(path)
        raise OSError("library unavailable")

    monkeypatch.setattr(ctypes, "CDLL", unavailable)
    assert _memory.reclaim_unused() is None
    assert _memory.reclaim_unused() is None
    assert attempts == ["/usr/lib/libSystem.B.dylib"]


def test_missing_symbol_is_cached_noop(binding, monkeypatch):
    attempts = []

    def no_symbol(path):
        attempts.append(path)
        return object()

    monkeypatch.setattr(ctypes, "CDLL", no_symbol)
    assert _memory.reclaim_unused() is None
    assert _memory.reclaim_unused() is None
    assert len(attempts) == 1


def test_binding_configuration_failure_is_noop(binding, monkeypatch):
    class Unbindable:
        @property
        def argtypes(self):
            return None

        @argtypes.setter
        def argtypes(self, value):
            raise RuntimeError("ABI binding failed")

        def __call__(self, *args):
            raise AssertionError("unconfigured function must never be called")

    loads = install_library(monkeypatch, Unbindable())
    assert _memory.reclaim_unused() is None
    assert _memory.reclaim_unused() is None
    assert len(loads) == 1


def test_native_call_failure_does_not_escape(binding, monkeypatch):
    native = FakeFunction(error=RuntimeError("native call failed"))
    loads = install_library(monkeypatch, native)
    assert _memory.reclaim_unused() is None
    assert _memory.reclaim_unused() is None
    assert len(loads) == 1 and native.calls == [(None, 0), (None, 0)]


@pytest.mark.parametrize("frames,expected", [(999999, ["inner"]),
                                           (1000000, ["release", "inner", "release"])])
def test_resample_boundary_preserves_arguments_result_and_input(monkeypatch, frames, expected):
    x = np.arange(frames, dtype=np.float32)[:, None]
    before = x.copy()
    result = np.ones((3, 2), dtype=np.float64)
    events = []

    def inner(signal, sr, target):
        assert signal is x and (sr, target) == (44100, 48000)
        events.append("inner")
        return result

    monkeypatch.setattr(dsp, "_resample_fft", inner)
    monkeypatch.setattr(_memory, "reclaim_unused", lambda: events.append("release"))
    assert dsp.resample_fft(x, 44100, 48000) is result
    np.testing.assert_array_equal(x, before)
    assert events == expected


def test_same_rate_long_signal_is_identity_without_reclamation(monkeypatch):
    def forbidden():
        raise AssertionError("same-rate identity must not load allocator API")

    monkeypatch.setattr(_memory, "reclaim_unused", forbidden)
    x = np.zeros((1000000, 2), dtype=np.float32)
    assert dsp.resample_fft(x, 44100, 44100) is x


def test_empty_unequal_rate_preserves_error_without_reclamation(monkeypatch):
    def forbidden():
        raise AssertionError("empty input must not load allocator API")

    monkeypatch.setattr(_memory, "reclaim_unused", forbidden)
    with pytest.raises(ValueError):
        dsp.resample_fft(np.zeros((0, 2)), 44100, 48000)


def test_after_reclamation_occurs_outside_completed_inner_frame(monkeypatch):
    events = []
    scratch = None
    result = np.ones(3)

    def inner(*args):
        nonlocal scratch
        temporary = np.ones(10)
        scratch = weakref.ref(temporary)
        events.append("inner")
        return result

    def release():
        if events:
            assert scratch() is None
        events.append("release")

    monkeypatch.setattr(dsp, "_resample_fft", inner)
    monkeypatch.setattr(_memory, "reclaim_unused", release)
    assert dsp.resample_fft(np.zeros(1000000), 44100, 48000) is result
    assert events == ["release", "inner", "release"]


def test_inner_error_is_preserved_and_after_cleanup_still_runs(monkeypatch):
    events = []
    error = LookupError("original resampling failure")

    def inner(*args):
        events.append("inner")
        raise error

    monkeypatch.setattr(dsp, "_resample_fft", inner)
    monkeypatch.setattr(_memory, "reclaim_unused", lambda: events.append("release"))
    with pytest.raises(LookupError) as caught:
        dsp.resample_fft(np.zeros(1000000), 44100, 48000)
    assert caught.value is error
    assert events == ["release", "inner", "release"]


@pytest.mark.parametrize('frames,dtype,channels,same_rate,expected',[
    (999999,np.float64,2,False,[(2,0)]),
    (1000000,np.float64,2,False,[(1,-1),(1,-1)]),
    (1000000,np.float32,2,False,[(2,0)]),
    (1000000,np.float64,1,False,[(2,0)]),
    (1000000,np.float64,None,False,[(1,-1)]),
    (1000000,np.int16,2,False,[(2,0)]),
    (1000000,np.float64,2,True,[]),
])
def test_workspace_gate_and_zero_signal_contract(monkeypatch,frames,dtype,channels,same_rate,expected):
    shape=(frames,) if channels is None else (frames,channels)
    x=np.zeros(shape,dtype=dtype)
    original=np.fft.irfft;calls=[]
    def observed(a,*args,**kwargs):
        calls.append((a.ndim,kwargs.get('axis',-1)))
        return original(a,*args,**kwargs)
    monkeypatch.setattr(np.fft,'irfft',observed)
    target=44100 if same_rate else 48000
    actual=dsp.resample_fft(x,44100,target)
    assert calls==expected
    assert actual.shape[0]==round(frames*target/44100)
    assert actual.shape[1:]==x.shape[1:] and not np.any(actual)
    assert not np.any(x)
    if same_rate:assert actual is x


def test_workspace_column_error_keeps_outer_cleanup_and_source(monkeypatch):
    x=np.ones((1000000,2),dtype=np.float64);before=x.copy()
    original=np.fft.irfft;calls=0;reclaims=[];error=RuntimeError('inverse failure')
    def failing(a,*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2:raise error
        return original(a,*args,**kwargs)
    monkeypatch.setattr(np.fft,'irfft',failing)
    monkeypatch.setattr(_memory,'reclaim_unused',lambda:reclaims.append(True))
    with pytest.raises(RuntimeError) as caught:dsp.resample_fft(x,44100,48000)
    assert caught.value is error and calls==2 and len(reclaims)==2
    np.testing.assert_array_equal(x,before)
