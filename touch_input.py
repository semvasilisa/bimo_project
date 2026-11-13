import threading
import time

try:
    from evdev import InputDevice, ecodes, list_devices
except ImportError:  # the main program catches the RuntimeError below
    InputDevice = None

NAME_HINTS = ("ads7846", "xpt2046", "touch")   # part of the input device's name
MIN_TAP_GAP = 1.0                              # seconds; ignore taps closer together


class TouchInput:
    def __init__(self):
        if InputDevice is None:
            raise RuntimeError("Python package 'evdev' is missing (pip install evdev).")

        self.device = self._find_device()
        print(f"[TOUCH] Using {self.device.path} ({self.device.name})")

        self._tapped = threading.Event()
        self._stop_event = threading.Event()
        self._thread = None
        self._last_tap = 0.0

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------
    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def was_tapped(self):
        """True if the screen was tapped since the last check or clear()."""
        if self._tapped.is_set():
            self._tapped.clear()
            return True
        return False

    def clear(self):
        """Forget any tap that happened earlier."""
        self._tapped.clear()

    def stop(self):
        self._stop_event.set()
        try:
            self.device.close()   # makes the blocked read in the thread end
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _find_device(self):
        names = []
        for path in list_devices():
            try:
                device = InputDevice(path)
            except PermissionError:
                raise RuntimeError(
                    f"No permission to read {path}. Run: sudo usermod -aG input $USER "
                    "and log in again."
                )
            names.append(device.name)
            if any(hint in device.name.lower() for hint in NAME_HINTS):
                return device
        raise RuntimeError(f"No touch device found. Input devices seen: {names}")

    def _run(self):
        try:
            for event in self.device.read_loop():
                if self._stop_event.is_set():
                    return
                # BTN_TOUCH value 1 = finger down (one event per tap)
                if event.type == ecodes.EV_KEY and event.code == ecodes.BTN_TOUCH \
                        and event.value == 1:
                    now = time.monotonic()
                    if now - self._last_tap >= MIN_TAP_GAP:
                        self._last_tap = now
                        self._tapped.set()
        except OSError as e:
            if not self._stop_event.is_set():
                print(f"[TOUCH] Stopped reading: {e}")


if __name__ == "__main__":
    touch = TouchInput()
    touch.start()
    print("[TOUCH] Tap the screen (Ctrl+C to quit)...")
    try:
        while True:
            if touch.was_tapped():
                print("[TOUCH] Tap!")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        touch.stop()