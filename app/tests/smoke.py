"""Per-function DSP smoke test. Prints as it goes so a crash shows the culprit."""
import sys, time
sys.path.insert(0, '.')
import numpy as np
from automix import dsp

sr = 48000
t = np.arange(sr) / sr
x = np.stack([0.3 * np.sin(2 * np.pi * 440 * t)] * 2, axis=1)

def step(name, fn):
    t0 = time.time()
    out = fn()
    shape = getattr(out, 'shape', out)
    print(f'OK  {name:14s} {time.time()-t0:6.2f}s  {shape}', flush=True)
    return out

step('lufs', lambda: round(dsp.loudness_lufs(x, sr), 2))
step('true_peak', lambda: round(dsp.true_peak_db(x, sr), 2))
step('apply_eq', lambda: dsp.apply_eq(x, sr, [('highpass', 40.0, 0.0, 0.7),
                                              ('peaking', 1000.0, 6.0, 1.0)]))
step('tonal_tilt', lambda: dsp.tonal_tilt(x, sr, {'low': 1.5, 'presence': 1.2}))
step('mono_below', lambda: dsp.mono_below(x, sr, 120.0))
step('compress', lambda: dsp.compress(x, sr, 0.6))
step('saturate', lambda: dsp.saturate(x, 0.3))
step('soft_limit', lambda: dsp.soft_limit(x, -1.0))
step('make_ir', lambda: dsp.make_ir(sr, 1.5, 20.0, 8000.0))
step('reverb_send', lambda: dsp.reverb_send(x, sr, 0.2, 1.5, 20.0, 8000.0))
step('crest_db', lambda: round(dsp.crest_db(x), 2))
step('centroid', lambda: round(dsp.spectral_centroid(x, sr), 1))
step('band_energy', lambda: np.round(dsp.band_energy(x, sr), 3))
step('stereo_corr', lambda: round(dsp.stereo_correlation(x), 3))
step('width_ratio', lambda: round(dsp.width_ratio(x), 3))
print('ALL OK', flush=True)