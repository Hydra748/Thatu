
import argparse
import os
import platform
import subprocess
import time
import webbrowser
from pynput import mouse
from typing import List

import numpy as np
import sounddevice as sd

from tap_detector import analyze_audio_block, NoiseFloorTracker


class TapEventTracker:
    def __init__(self, tap_interval: float = 0.5, debounce: float = 0.08, min_tap_gap: float = 0.15):
        self.tap_interval = tap_interval
        self.debounce = debounce
        self.min_tap_gap = min_tap_gap
        self.pending_taps = 0
        self.last_peak_time = 0.0
        self.last_event_time = 0.0
        self.current_band = "none"
        self.best_peak_value = 0.0

    def add_peaks(self, peaks: List[tuple], band: str, block_time: float):
        events = []
        for offset, peak_value in peaks:
            absolute_time = block_time + offset
            if self.pending_taps == 0:
                self.pending_taps = 1
                self.last_peak_time = absolute_time
                self.current_band = band
                self.best_peak_value = peak_value
            elif absolute_time - self.last_peak_time <= self.tap_interval:
                if absolute_time - self.last_peak_time < self.min_tap_gap:
                    # Same physical tap event (decay/jitter), do not count as a new tap
                    self.last_peak_time = absolute_time
                    self.best_peak_value = max(self.best_peak_value, peak_value)
                else:
                    # Distinct second/subsequent tap
                    self.pending_taps += 1
                    self.last_peak_time = absolute_time
                    self.best_peak_value = max(self.best_peak_value, peak_value)
            else:
                events.extend(self._emit_event())
                self.pending_taps = 1
                self.last_peak_time = absolute_time
                self.current_band = band
                self.best_peak_value = peak_value
        return events

    def flush(self, now: float):
        if self.pending_taps > 0 and now - self.last_peak_time > self.tap_interval:
            return self._emit_event()
        return []

    def _emit_event(self):
        if self.pending_taps == 0:
            return []
        if self.last_event_time and self.last_peak_time - self.last_event_time < self.debounce:
            self.pending_taps = 0
            self.best_peak_value = 0.0
            self.last_peak_time = 0.0
            return []

        event = [{
            "count": self.pending_taps,
            "band": self.current_band,
            "time": self.last_peak_time,
            "peak_value": self.best_peak_value,
        }]
        self.last_event_time = self.last_peak_time
        self.pending_taps = 0
        self.best_peak_value = 0.0
        self.last_peak_time = 0.0
        return event


class MouseDoubleClickListener:
    """
    Listens to global mouse events in a background thread and triggers a callback
    upon detecting a double-click of the middle mouse button.
    """
    def __init__(self, callback, interval: float = 0.5):
        self.callback = callback
        self.interval = interval
        self.last_click_time = 0.0
        self.click_count = 0
        self.listener = None

    def _on_click(self, x, y, button, pressed):
        if button == mouse.Button.middle and pressed:
            now = time.time()
            if now - self.last_click_time <= self.interval:
                self.click_count += 1
            else:
                self.click_count = 1
            self.last_click_time = now

            if self.click_count >= 2:
                print("Double click of middle mouse button detected. Opening Google...")
                self.callback()
                self.click_count = 0  # Reset count after action

    def start(self):
        self.listener = mouse.Listener(on_click=self._on_click)
        self.listener.start()

    def stop(self):
        if self.listener:
            self.listener.stop()


class TapListener:
    def __init__(
        self,
        sample_rate: int = 16000,
        block_duration: float = 0.04,
        sensitivity: float = 0.12,
        tap_window: float = 0.5,
        debounce: float = 0.08,
    ):
        self.sample_rate = sample_rate
        self.block_duration = block_duration
        self.sensitivity = sensitivity
        self.tracker = TapEventTracker(tap_interval=tap_window, debounce=debounce)
        self.noise_tracker = NoiseFloorTracker()
        self.consecutive_active_blocks = 0
        self.buffer: List[float] = []
        self.app_path = ""

    def set_app(self, app_path: str) -> None:
        self.app_path = app_path

    def _open_app(self) -> None:
        if not self.app_path:
            print("No app configured to open.")
            return
        try:
            if platform.system() == "Windows":
                os.startfile(self.app_path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen([self.app_path])
        except Exception as exc:
            print(f"Failed to open app '{self.app_path}': {exc}")
            if platform.system() == "Windows":
                try:
                    subprocess.Popen(["cmd", "/c", "start", "", self.app_path])
                except Exception as fallback_exc:
                    print(f"Fallback open failed: {fallback_exc}")

    def _handle_event(self, event: dict) -> None:
        if event["count"] >= 2:
            if event["band"] in ("sub", "low"):
                print(f"Double tap on the table detected (band: {event['band']}). Opening Google...")
                try:
                    webbrowser.open("https://www.google.com")
                except Exception as exc:
                    print(f"Failed to open Google: {exc}")
            else:
                print(f"Double tap detected (band: {event['band']}).")
                self._open_app()
        else:
            print(f"Single tap detected (band: {event['band']}).")

    def _process_block(self, block: np.ndarray, block_time: float) -> None:
        block_rms = float(np.sqrt(np.mean(block ** 2)))
        noise_level = self.noise_tracker.update(block_rms)

        # Noise-gate SNR check: block must stand out from background noise level.
        snr_threshold = 2.5
        min_rms_threshold = 0.003

        if block_rms <= max(noise_level * snr_threshold, min_rms_threshold):
            self.consecutive_active_blocks = 0
            # Still flush the event tracker in case a previous tap was pending.
            events = self.tracker.flush(block_time + len(block) / self.sample_rate)
            for event in events:
                self._handle_event(event)
            return

        self.consecutive_active_blocks += 1

        # If sound is sustained (too many consecutive loud blocks), it's not a tap.
        # 3 blocks of 40ms = 120ms, which exceeds a transient tap duration.
        if self.consecutive_active_blocks > 3:
            # Clear pending tap state to ignore the sustained sound.
            self.tracker.pending_taps = 0
            self.tracker.best_peak_value = 0.0
            self.tracker.last_peak_time = 0.0
            return

        analysis = analyze_audio_block(
            block,
            self.sample_rate,
            energy_threshold=0.002,
            sensitivity=self.sensitivity,
            noise_level=noise_level,
        )
        peaks = analysis["peaks"] if analysis["dominant_band"] else []
        events = self.tracker.add_peaks(peaks, analysis["dominant_band"] or "none", block_time)
        events.extend(self.tracker.flush(block_time + len(block) / self.sample_rate))

        for event in events:
            self._handle_event(event)

    def start(self) -> None:
        block_size = int(self.sample_rate * self.block_duration)

        def open_google():
            try:
                webbrowser.open("https://www.google.com")
            except Exception as exc:
                print(f"Failed to open Google: {exc}")

        mouse_listener = MouseDoubleClickListener(callback=open_google)
        mouse_listener.start()

        def callback(indata, frames, time_info, status):
            if status:
                print(status)
            self.buffer.extend(indata[:, 0].tolist())
            if len(self.buffer) >= block_size:
                chunk = self.buffer[:block_size]
                del self.buffer[:block_size]
                block_time = time.time() - float(block_size) / float(self.sample_rate)
                self._process_block(np.array(chunk, dtype=np.float32), block_time)

        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32", callback=callback):
                print("Listening for taps / middle mouse clicks. Press Ctrl+C to stop.")
                while True:
                    time.sleep(0.1)
        finally:
            mouse_listener.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Listen for microphone taps using frequency-domain filtering")
    parser.add_argument("--app", default=r"C:\Windows\System32\notepad.exe", help="Path to the app to open")
    parser.add_argument("--sensitivity", type=float, default=0.12, help="Amplitude threshold for tap detection")
    parser.add_argument("--duration", type=float, default=0.04, help="Audio block duration in seconds")
    parser.add_argument("--tap-window", type=float, default=0.5, help="Maximum gap in seconds between taps")
    parser.add_argument("--debounce", type=float, default=0.08, help="Minimum time in seconds between events")
    args = parser.parse_args()

    listener = TapListener(
        sample_rate=16000,
        block_duration=args.duration,
        sensitivity=args.sensitivity,
        tap_window=args.tap_window,
        debounce=args.debounce,
    )
    listener.set_app(args.app)
    listener.start()
