"""Engineered vision helpers: BMP decoding + color-blob enemy detection.

Part of the vision-feature plan (see /memories/session/plan.md): rather than
training a CNN end-to-end via ES (which scales poorly with parameter count --
see repo memory's shot-index-one-hot probe), this module implements a cheap,
non-learned "vision" feature: detect the nearest enemy sprite's on-screen
position via simple color thresholding, so only a couple of extra scalar obs
dims need to be learned by the (still tiny, still ES-trained) policy.

``decode_bmp`` matches the wire format BizHawk's ``comm.socketServerScreenShot()``
sends (raw, uncompressed BMP bytes via .NET's
``ImageConverter.ConvertTo(bitmap, typeof(byte[]))`` -- verified against
BizHawk's SocketServer.cs / MainForm.cs source). ``detect_enemy`` works on a
plain HxWx3 uint8 RGB array regardless of source (decoded BMP from a real
capture, or a synthetic frame from a sim probe), so the same function is
reused by tests/test_simulation.py's sim-only probes and later by the real
env_timecrisis.py integration.

Numpy-only by design (no Pillow/OpenCV dependency), matching this repo's
existing "numpy only" convention (requirements.txt has only numpy+matplotlib).

``detect_enemy_multi``/``detect_motion_mask`` are a follow-up to the single-
color ``detect_enemy``: real enemies each have a distinct sprite color AND
are visibly moving, so detection can be made more selective by requiring a
palette-color match AND (once a previous frame is available) genuine motion
between captures -- this avoids being fooled by static same-colored scenery
that a plain color threshold cannot distinguish from a real target.
"""

import struct

import numpy as np


def decode_bmp(data: bytes) -> np.ndarray:
    """Decode an uncompressed 24 or 32 bpp BMP byte string into an HxWx3
    uint8 RGB array (row 0 = top of the image).

    Only supports BI_RGB (uncompressed) 24/32bpp BMPs, which is what
    .NET's ``ImageConverter.ConvertTo(bitmap, typeof(byte[]))`` produces for
    an in-memory framebuffer bitmap -- the exact path BizHawk's
    ``comm.socketServerScreenShot()`` uses. Raises ValueError for anything
    else (compressed, indexed/palette, etc.).
    """
    if len(data) < 54 or data[0:2] != b"BM":
        raise ValueError("Not a BMP file (missing 'BM' magic)")

    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    width = struct.unpack_from("<i", data, 18)[0]
    height = struct.unpack_from("<i", data, 22)[0]
    bpp = struct.unpack_from("<H", data, 28)[0]
    compression = struct.unpack_from("<I", data, 30)[0]

    if compression != 0:
        raise ValueError(f"Unsupported BMP compression: {compression}")
    if bpp not in (24, 32):
        raise ValueError(f"Unsupported BMP bit depth: {bpp}")

    bottom_up = height > 0
    h = abs(height)
    w = width
    channels = bpp // 8
    # Rows are padded to a multiple of 4 bytes.
    row_bytes = ((w * bpp + 31) // 32) * 4

    needed = pixel_offset + row_bytes * h
    if len(data) < needed:
        raise ValueError(
            f"Truncated BMP: need {needed} bytes, got {len(data)}"
        )

    raw = np.frombuffer(data, dtype=np.uint8, count=row_bytes * h, offset=pixel_offset)
    rows = raw.reshape(h, row_bytes)[:, : w * channels].reshape(h, w, channels)
    # BMP stores pixels as BGR(A); reverse the first 3 channels to get RGB.
    rgb = np.ascontiguousarray(rows[:, :, 2::-1][:, :, :3])
    if bottom_up:
        rgb = rgb[::-1]
    return rgb


def detect_enemy(
    frame: np.ndarray,
    target_color: tuple[int, int, int] = (220, 30, 30),
    tolerance: int = 40,
) -> tuple[float, float] | None:
    """Detect the centroid of the nearest-enemy color blob in an HxWx3 uint8
    RGB frame.

    Returns normalized (x, y) in [0, 1] (matching this project's aim_x/aim_y
    and cursor-normalization convention -- 0 = left/top), or None if no
    pixel in the frame is within ``tolerance`` of ``target_color`` on every
    channel.

    This is intentionally simple (single-color threshold + centroid, no
    real detector/classifier) -- see the vision plan's "engineered feature
    extraction" decision: the goal is a cheap, non-learned signal, not a
    trained perception model.
    """
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(f"Expected HxWx3(+) frame, got shape {frame.shape}")

    diff = np.abs(frame[:, :, :3].astype(np.int16) - np.array(target_color, dtype=np.int16))
    mask = np.all(diff <= tolerance, axis=-1)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None

    h, w = frame.shape[:2]
    cx = float((xs.mean() + 0.5) / w)
    cy = float((ys.mean() + 0.5) / h)
    return cx, cy


def detect_motion_mask(
    prev_frame: np.ndarray,
    frame: np.ndarray,
    threshold: int = 25,
) -> np.ndarray:
    """Boolean HxW mask of pixels that changed between two same-shape HxWx3(+)
    uint8 RGB frames.

    A pixel counts as "moved" when the summed absolute per-channel
    difference (R+G+B) exceeds ``threshold``. Used to filter color-blob
    detection down to genuinely animating/moving sprites, rather than any
    static same-colored scenery (see ``detect_enemy_multi``).
    """
    if prev_frame.shape != frame.shape:
        raise ValueError(
            f"Frame shape mismatch: {prev_frame.shape} vs {frame.shape}"
        )
    diff = np.abs(
        frame[:, :, :3].astype(np.int16) - prev_frame[:, :, :3].astype(np.int16)
    )
    return diff.sum(axis=-1) > threshold


def detect_enemy_multi(
    frame: np.ndarray,
    palette,
    tolerance: int = 40,
    prev_frame: np.ndarray | None = None,
    motion_threshold: int = 25,
    ref_pos: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    """Detect the on-screen enemy nearest ``ref_pos`` using a per-enemy-type
    color PALETTE and, when ``prev_frame`` is given, a MOTION filter.

    Real Time Crisis enemies each have a roughly fixed, distinct sprite
    color and are visibly moving/animating on screen -- a single static
    color threshold (``detect_enemy``) can be fooled by any static
    same-colored scenery. ``palette`` is an iterable of (r, g, b) colors,
    one per known enemy type/costume; for each color, pixels within
    ``tolerance`` on every channel form a candidate blob and its centroid is
    computed independently (no cross-color mixing).

    When ``prev_frame`` is provided, each color's mask is additionally
    ANDed with ``detect_motion_mask(prev_frame, frame, motion_threshold)``
    -- pixels that didn't change since the previous capture are excluded,
    so a static prop sharing an enemy's color is not mistaken for a real,
    moving target. This means a capture with no motion at all (relative to
    ``prev_frame``) correctly reports no detection, even if a same-colored
    static object is present. With no ``prev_frame`` (e.g. the very first
    capture, before any motion reference exists), detection falls back to
    color-only across the palette.

    Among the resulting per-color candidate blobs, the one nearest to
    ``ref_pos`` (typically the current cursor position) is returned as its
    normalized [0, 1] (x, y) centroid; if ``ref_pos`` is None, the largest
    (most pixels) candidate is returned instead. Returns None if no palette
    color has any (post-motion-filter) matching pixels.
    """
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(f"Expected HxWx3(+) frame, got shape {frame.shape}")

    motion_mask = None
    if prev_frame is not None:
        motion_mask = detect_motion_mask(prev_frame, frame, motion_threshold)

    h, w = frame.shape[:2]
    frame_i16 = frame[:, :, :3].astype(np.int16)
    candidates = []  # (cx, cy, pixel_count)
    for color in palette:
        diff = np.abs(frame_i16 - np.array(color, dtype=np.int16))
        mask = np.all(diff <= tolerance, axis=-1)
        if motion_mask is not None:
            mask = mask & motion_mask
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            continue
        cx = float((xs.mean() + 0.5) / w)
        cy = float((ys.mean() + 0.5) / h)
        candidates.append((cx, cy, int(xs.size)))

    if not candidates:
        return None
    if ref_pos is not None:
        rx, ry = ref_pos
        best = min(candidates, key=lambda c: (c[0] - rx) ** 2 + (c[1] - ry) ** 2)
    else:
        best = max(candidates, key=lambda c: c[2])
    return best[0], best[1]
