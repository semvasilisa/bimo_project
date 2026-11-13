import os
import random
import threading
import time

import numpy as np
from PIL import Image

FB_DEVICE = "/dev/fb0"
FACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces")

# state -> (seconds between frames, how the next frame is chosen)
#   "cycle"  = show the pictures one after another
#   "random" = pick a random picture (looks like a talking mouth)
STATE_SETTINGS = {
    "warmup": (0.5, "cycle"),
    "idle": (0.5, "cycle"),
    "listening": (0.5, "cycle"),
    "thinking": (0.4, "cycle"),
    "speaking": (0.2, "random"),
    "error": (0.5, "cycle"),
}
START_STATE = "warmup"


class FaceDisplay:
    def __init__(self, fb_device=FB_DEVICE, faces_dir=FACES_DIR):
        self.fb_device = fb_device
        self.width, self.height, self.bits_per_pixel, self.stride = (
            self._read_framebuffer_info()
        )
        print(f"[FACE] Screen {fb_device}: {self.width}x{self.height}, "
              f"{self.bits_per_pixel} bits per pixel")

        # state -> list of ready-to-write frames (bytes in the screen's format)
        self.frames = self._load_all_frames(faces_dir)

        self._state = START_STATE            # the state the main program asked for
        self._wake = threading.Event()       # wakes the thread early when the state changes
        self._stop_event = threading.Event()
        self._thread = None
        self._write_error_reported = False

    # ------------------------------------------------------------------
    # Public methods (the only ones the main program needs)
    # ------------------------------------------------------------------
    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_state(self, name):
        """Change the face. Only changes a variable, so it is fast and safe to call."""
        if name not in STATE_SETTINGS:
            print(f"[FACE] Unknown state '{name}' ignored.")
            return
        self._state = name   # assigning one value is atomic in Python
        self._wake.set()     # do not wait for the current delay to finish

    def stop(self):
        self._stop_event.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self._write(self._to_fb_bytes(
            np.zeros((self.height, self.width, 3), dtype=np.uint8)
        ))

    # ------------------------------------------------------------------
    # The background thread
    # ------------------------------------------------------------------
    def _run(self):
        state = None       # the state we are showing now
        frame_index = 0    # which picture of that state is on the screen
        shown = None       # (state, frame_index) that was last written

        while not self._stop_event.is_set():
            requested = self._state
            frames = self._frames_for(requested)
            delay, mode = STATE_SETTINGS[requested]

            if requested != state:
                state = requested
                frame_index = 0                      # new state: start with its first picture
            else:
                frame_index = self._next_index(frame_index, len(frames), mode)

            # Write only when the picture changed: single-picture states cost nothing
            if (state, frame_index) != shown:
                if self._write(frames[frame_index]):
                    shown = (state, frame_index)

            self._wake.wait(delay)   # sleeps, but wakes at once on set_state()
            self._wake.clear()

    @staticmethod
    def _next_index(current, count, mode):
        if count < 2:
            return 0
        if mode == "random":
            # never the same picture twice in a row
            return random.choice([i for i in range(count) if i != current])
        return (current + 1) % count

    def _frames_for(self, state):
        # a state without pictures shows the idle face
        return self.frames.get(state) or self.frames["idle"]

    # ------------------------------------------------------------------
    # Screen access
    # ------------------------------------------------------------------
    def _read_framebuffer_info(self):
        sys_dir = f"/sys/class/graphics/{os.path.basename(self.fb_device)}"
        try:
            with open(f"{sys_dir}/virtual_size") as f:
                width, height = (int(v) for v in f.read().strip().split(","))
            with open(f"{sys_dir}/bits_per_pixel") as f:
                bits_per_pixel = int(f.read().strip())
            with open(f"{sys_dir}/stride") as f:
                stride = int(f.read().strip())  # bytes per screen line
        except OSError as e:
            raise RuntimeError(
                f"Cannot read the screen info of {self.fb_device} ({e}). "
                "Is the display overlay in /boot/firmware/config.txt?"
            )
        return width, height, bits_per_pixel, stride

    def _to_fb_bytes(self, rgb):
        """Convert an (height, width, 3) uint8 RGB image to the framebuffer format."""
        height, width, _ = rgb.shape

        if self.bits_per_pixel == 16:
            # RGB565: 5 bits red, 6 bits green, 5 bits blue, stored little-endian
            r = rgb[:, :, 0].astype(np.uint16)
            g = rgb[:, :, 1].astype(np.uint16)
            b = rgb[:, :, 2].astype(np.uint16)
            pixels = (((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)).astype("<u2")
            raw = pixels.view(np.uint8).reshape(height, width * 2)
        elif self.bits_per_pixel == 32:
            # Assumed byte order B, G, R, unused
            raw = np.zeros((height, width, 4), dtype=np.uint8)
            raw[:, :, 0] = rgb[:, :, 2]
            raw[:, :, 1] = rgb[:, :, 1]
            raw[:, :, 2] = rgb[:, :, 0]
            raw = raw.reshape(height, width * 4)
        else:
            raise RuntimeError(f"Unsupported bits per pixel: {self.bits_per_pixel}")

        # Some drivers pad every line to `stride` bytes
        if self.stride > raw.shape[1]:
            padded = np.zeros((height, self.stride), dtype=np.uint8)
            padded[:, :raw.shape[1]] = raw
            raw = padded

        return raw.tobytes()

    def _write(self, frame_bytes):
        """Write one full picture into the framebuffer. Returns True on success."""
        try:
            with open(self.fb_device, "wb", buffering=0) as fb:
                fb.write(frame_bytes)
            return True
        except OSError as e:
            # The face must never crash the assistant; report once and carry on
            if not self._write_error_reported:
                print(f"[FACE] Cannot write to {self.fb_device}: {e}")
                self._write_error_reported = True
            return False

    # ------------------------------------------------------------------
    # Loading the pictures
    # ------------------------------------------------------------------
    def _load_all_frames(self, faces_dir):
        frames = {}

        for state in STATE_SETTINGS:
            folder = os.path.join(faces_dir, state)
            state_frames = []

            if os.path.isdir(folder):
                for name in sorted(os.listdir(folder)):
                    if name.lower().endswith(".png"):
                        state_frames.append(self._load_picture(os.path.join(folder, name)))

            print(f"[FACE] {state}: {len(state_frames)} picture(s)")
            if state_frames:
                frames[state] = state_frames

        if "idle" not in frames:
            print("[FACE] WARNING: no idle pictures, using a black screen.")
            frames["idle"] = [self._to_fb_bytes(
                np.zeros((self.height, self.width, 3), dtype=np.uint8)
            )]

        return frames

    def _load_picture(self, path):
        # convert("RGB") drops transparency (transparent areas usually become black)
        image = Image.open(path).convert("RGB")

        if image.size != (self.width, self.height):
            print(f"[FACE] {os.path.basename(path)} is {image.size[0]}x{image.size[1]}, "
                  f"resizing to {self.width}x{self.height}")
            image = image.resize((self.width, self.height))

        return self._to_fb_bytes(np.asarray(image))


if __name__ == "__main__":
    # Demo: show every state for 4 seconds
    face = FaceDisplay()
    face.start()
    try:
        for demo_state in STATE_SETTINGS:
            print(f"[FACE] Showing: {demo_state}")
            face.set_state(demo_state)
            time.sleep(4)
    except KeyboardInterrupt:
        pass
    finally:
        face.stop()