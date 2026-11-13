"""Framebuffer test for the SPI screen.

It prints the framebuffers the Pi can see, shows solid colors and a corner
pattern (to check the orientation), and measures how fast a full frame can be
written.

Usage:  python fb_test.py [/dev/fb1]
"""
import glob
import os
import sys
import time

import numpy as np

FB_DEVICE = sys.argv[1] if len(sys.argv) > 1 else "/dev/fb1"
SYS_DIR = f"/sys/class/graphics/{os.path.basename(FB_DEVICE)}"


def read_sysfs(file_name):
    with open(f"{SYS_DIR}/{file_name}") as f:
        return f.read().strip()


def framebuffer_info():
    """Read the screen geometry from sysfs (no root needed)."""
    width, height = (int(v) for v in read_sysfs("virtual_size").split(","))
    bits_per_pixel = int(read_sysfs("bits_per_pixel"))
    stride = int(read_sysfs("stride"))  # bytes per screen line
    return width, height, bits_per_pixel, stride


def to_fb_bytes(rgb, bits_per_pixel, stride):
    """Convert an (height, width, 3) uint8 RGB image to the framebuffer format."""
    height, width, _ = rgb.shape

    if bits_per_pixel == 16:
        # RGB565: 5 bits red, 6 bits green, 5 bits blue, stored little-endian
        r = rgb[:, :, 0].astype(np.uint16)
        g = rgb[:, :, 1].astype(np.uint16)
        b = rgb[:, :, 2].astype(np.uint16)
        pixels = (((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)).astype("<u2")
        raw = pixels.view(np.uint8).reshape(height, width * 2)
    elif bits_per_pixel == 32:
        # Assumed byte order B, G, R, unused (common for 32-bit framebuffers)
        raw = np.zeros((height, width, 4), dtype=np.uint8)
        raw[:, :, 0] = rgb[:, :, 2]
        raw[:, :, 1] = rgb[:, :, 1]
        raw[:, :, 2] = rgb[:, :, 0]
        raw = raw.reshape(height, width * 4)
    else:
        raise SystemExit(f"Unsupported bits per pixel: {bits_per_pixel}")

    # Some drivers pad every line, so each line must be `stride` bytes long
    if stride > raw.shape[1]:
        padded = np.zeros((height, stride), dtype=np.uint8)
        padded[:, :raw.shape[1]] = raw
        raw = padded

    return raw.tobytes()


def show(frame_bytes):
    """Write one full frame into the framebuffer device."""
    with open(FB_DEVICE, "wb", buffering=0) as fb:
        fb.write(frame_bytes)


def solid(width, height, color):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = color
    return image


def corner_pattern(width, height, size=60):
    """Black screen with a colored square in each corner."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:size, :size] = (255, 0, 0)                # top-left: red
    image[:size, -size:] = (0, 255, 0)               # top-right: green
    image[-size:, :size] = (0, 0, 255)               # bottom-left: blue
    image[-size:, -size:] = (255, 255, 255)          # bottom-right: white
    return image


def main():
    print("Framebuffers found:")
    for path in sorted(glob.glob("/sys/class/graphics/fb*")):
        with open(f"{path}/name") as f:
            name = f.read().strip()
        with open(f"{path}/virtual_size") as f:
            size = f.read().strip()
        print(f"  {os.path.basename(path)}: {name}, {size}")

    width, height, bits_per_pixel, stride = framebuffer_info()
    print(f"\nUsing {FB_DEVICE}: {width}x{height}, "
          f"{bits_per_pixel} bits per pixel, stride {stride} bytes")

    try:
        for label, color in [
            ("RED", (255, 0, 0)),
            ("GREEN", (0, 255, 0)),
            ("BLUE", (0, 0, 255)),
            ("WHITE", (255, 255, 255)),
            ("BLACK", (0, 0, 0)),
        ]:
            print(f"Showing {label}")
            show(to_fb_bytes(solid(width, height, color), bits_per_pixel, stride))
            time.sleep(1.5)

        print("Corner test: red = top-left, green = top-right, "
              "blue = bottom-left, white = bottom-right")
        show(to_fb_bytes(corner_pattern(width, height), bits_per_pixel, stride))
        time.sleep(6)

        # Speed test: how long does one full frame take?
        frames = [
            to_fb_bytes(solid(width, height, (255, 0, 0)), bits_per_pixel, stride),
            to_fb_bytes(solid(width, height, (0, 0, 255)), bits_per_pixel, stride),
        ]
        count = 20
        start = time.perf_counter()
        for i in range(count):
            show(frames[i % 2])
        per_frame = (time.perf_counter() - start) / count
        print(f"Speed: {per_frame * 1000:.0f} ms per frame "
              f"(about {1 / per_frame:.1f} frames per second)")

        show(to_fb_bytes(solid(width, height, (0, 0, 0)), bits_per_pixel, stride))

    except PermissionError:
        print(f"\nNo permission to write to {FB_DEVICE}. "
              "Check that your user is in the 'video' group (groups command).")


if __name__ == "__main__":
    main()
