
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

BAND_DEFINITIONS = [
    ("sub", 20.0, 200.0),
    ("low", 200.0, 600.0),
    ("mid", 600.0, 1500.0),
    ("high", 1500.0, 4000.0),
    ("ultra", 4000.0, 8000.0),
]


class NoiseFloorTracker:
    """
    Tracks the running background noise floor of audio blocks using an asymmetric
    exponential moving average (EMA) filter. This helps ignore short spikes (taps)
    while adjusting to persistent ambient noise changes.
    """
    def __init__(self, alpha_up: float = 0.02, alpha_down: float = 0.10, initial_noise: float = 0.002):
        self.noise_level = initial_noise
        self.alpha_up = alpha_up
        self.alpha_down = alpha_down

    def update(self, block_rms: float) -> float:
        # Asymmetric update: rise slowly, fall faster.
        # Short transients (taps) have high RMS but don't significantly raise the estimated floor,
        # while persistent changes in background noise will be adapted to quickly.
        if block_rms > self.noise_level:
            self.noise_level += self.alpha_up * (block_rms - self.noise_level)
        else:
            self.noise_level += self.alpha_down * (block_rms - self.noise_level)
        return self.noise_level


def compute_spectrum(samples: Sequence[float], sample_rate: int, n_fft: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(samples, dtype=np.float32)
    if signal.size < n_fft:
        signal = np.pad(signal, (0, n_fft - signal.size))
    else:
        n_fft = signal.size

    window = np.hanning(n_fft)
    spectrum = np.fft.rfft(signal * window)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    mags = np.abs(spectrum)
    return freqs, mags


def band_energy(freqs: np.ndarray, mags: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    return float(np.sum(mags[mask] ** 2))


def find_dominant_band(
    samples: Sequence[float],
    sample_rate: int = 16000,
    energy_threshold: float = 0.05,
) -> Tuple[Optional[str], Dict[str, float]]:
    freqs, mags = compute_spectrum(samples, sample_rate)
    total_energy = float(np.sum(mags ** 2)) + 1e-12

    band_energies: Dict[str, float] = {}
    for name, low, high in BAND_DEFINITIONS:
        band_energies[name] = band_energy(freqs, mags, low, high)

    best_band = max(band_energies, key=band_energies.get)
    if band_energies[best_band] < energy_threshold * total_energy:
        return None, band_energies

    return best_band, band_energies


def biquad_bandpass(samples: Sequence[float], lowcut: float, highcut: float, sample_rate: int) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32)
    if x.size == 0:
        return x
    f0 = (lowcut + highcut) / 2.0
    bw = max(highcut - lowcut, 1e-6)
    q = max(f0 / bw, 0.5)
    omega = 2.0 * math.pi * f0 / sample_rate
    alpha = math.sin(omega) / (2.0 * q)

    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * math.cos(omega)
    a2 = 1.0 - alpha

    y = np.zeros_like(x)
    x1 = x2 = 0.0
    y1 = y2 = 0.0
    for i, xn in enumerate(x):
        yn = (b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0
        y[i] = yn
        x2 = x1
        x1 = xn
        y2 = y1
        y1 = yn
    return y


def smooth_envelope(signal: Sequence[float], sample_rate: int, window_ms: float = 6.0) -> np.ndarray:
    x = np.abs(np.asarray(signal, dtype=np.float32))
    window = max(1, int(sample_rate * window_ms / 1000.0))
    if window == 1:
        return x
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x, kernel, mode="same")


def extract_peaks(
    signal: Sequence[float],
    sample_rate: int,
    threshold: float,
    merge_ms: float = 35.0,
    min_separation_ms: float = 20.0,
    min_absolute_threshold: float = 0.0,
) -> List[Tuple[float, float]]:
    envelope = smooth_envelope(signal, sample_rate)
    if envelope.size == 0:
        return []

    max_envelope = float(np.max(envelope))
    if max_envelope < min_absolute_threshold:
        return []

    if 0.0 < threshold <= 1.0:
        threshold_value = max_envelope * threshold
    else:
        threshold_value = threshold
    threshold_value = min(threshold_value, max_envelope * 0.9)
    candidates: List[int] = []
    for i in range(1, envelope.size - 1):
        if (envelope[i] > threshold_value and 
            envelope[i] >= min_absolute_threshold and 
            envelope[i] >= envelope[i - 1] and 
            envelope[i] >= envelope[i - 1 + 2]):  # envelope[i + 1]
            candidates.append(i)

    if not candidates:
        return []

    merge_samples = max(1, int(sample_rate * merge_ms / 1000.0))
    groups: List[List[int]] = [[candidates[0]]]
    for idx in candidates[1:]:
        if idx - groups[-1][-1] <= merge_samples:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    peaks: List[Tuple[float, float]] = []
    for group in groups:
        best_idx = max(group, key=lambda idx: envelope[idx])
        peaks.append((float(best_idx) / float(sample_rate), float(envelope[best_idx])))

    separation = max(1, int(sample_rate * min_separation_ms / 1000.0))
    compressed: List[Tuple[float, float]] = []
    for time_sec, value in peaks:
        if not compressed or int(time_sec * sample_rate) - int(compressed[-1][0] * sample_rate) >= separation:
            compressed.append((time_sec, value))
    return compressed


def analyze_audio_block(
    samples: Sequence[float],
    sample_rate: int = 16000,
    energy_threshold: float = 0.005,
    sensitivity: float = 0.12,
    noise_level: Optional[float] = None,
) -> Dict[str, object]:
    dominant_band, band_energies = find_dominant_band(samples, sample_rate, energy_threshold)
    result: Dict[str, object] = {
        "dominant_band": dominant_band,
        "band_energies": band_energies,
        "total_energy": float(np.sum(np.abs(samples) ** 2)),
        "filtered": np.zeros(len(samples), dtype=np.float32),
        "peaks": [],
    }

    min_abs_thresh = noise_level * 3.5 if noise_level is not None else 0.0

    if dominant_band is not None:
        lowcut = next(low for name, low, high in BAND_DEFINITIONS if name == dominant_band)
        highcut = next(high for name, low, high in BAND_DEFINITIONS if name == dominant_band)
        filtered = biquad_bandpass(samples, lowcut, highcut, sample_rate)
        result["filtered"] = filtered
        threshold = min(sensitivity, np.max(np.abs(filtered)) * 0.18)
        result["peaks"] = extract_peaks(
            filtered,
            sample_rate,
            threshold=threshold,
            min_absolute_threshold=min_abs_thresh,
        )
        return result

    # Fallback to raw tap-like energy detection when no dominant band is strong enough.
    filtered = np.asarray(samples, dtype=np.float32)
    result["filtered"] = filtered
    max_val = float(np.max(np.abs(filtered)))
    threshold = min(0.12, max(max_val * 0.05, 0.02))
    actual_min_abs = max(threshold, min_abs_thresh)
    result["dominant_band"] = "raw"
    result["peaks"] = extract_peaks(
        filtered,
        sample_rate,
        threshold=threshold,
        min_absolute_threshold=actual_min_abs,
    )
    return result
