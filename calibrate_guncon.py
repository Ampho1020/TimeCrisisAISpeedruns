"""Automated GunCon X/Y calibration probe.

Measures the TRUE mapping from the aim value we WRITE to the actual on-screen
reticle position the game renders (read back from RAM ``cursor_x``/``cursor_y``),
then solves for the ``GUNCON_CALIB`` scale/offset that makes a written aim land
exactly where intended (identity end-to-end).

Why this exists
---------------
Live runs show a right-side enemy consistently missed to the LEFT. The aim
pipeline is: detector box-center -> policy blend -> ``apply_guncon_calibration``
-> Lua ``X Axis`` 0..2640 -> game reticle. The detector X is unbiased (box
center; the shield lateral offset is disabled), and the Lua map is linear, so
the only horizontal distortion is ``GUNCON_CALIB``. Its current
``scale_x = 0.94`` compresses every off-center aim TOWARD screen center, which
pulls right-side shots left -- the exact symptom. Rather than guess a new
scale, this probe drives the axis to known positions and reads the real reticle
location from RAM to compute the correct transform.

How it works
------------
1. Temporarily forces ``GUNCON_CALIB`` to identity so ``set_input`` writes the
   raw aim (we want to characterize the *device* mapping, uncorrected).
2. Launches one BizHawk via ``WorkerPool`` (reuses the tested launch/handshake),
   loads ``STATE_SLOT``, and holds peek so the reticle is live.
3. Sweeps aim across the screen (forward and reverse to expose any lag), reading
   back the normalized reticle position at each step.
4. Least-squares fits ``measured = m * written + c`` and solves for the
   correcting transform ``T(v) = 0.5 + (v-0.5)*scale_x + offset_x`` such that
   ``g(T(v)) = v``:  ``scale_x = 1/m``,  ``offset_x = (0.5 - c)/m - 0.5``.
5. Prints a table and the recommended ``GUNCON_CALIB`` values. It does NOT edit
   config.py -- you review the numbers first.

Usage
-----
    DISPLAY=:0 PYTHONPATH=. .venv/bin/python calibrate_guncon.py
    DISPLAY=:0 PYTHONPATH=. .venv/bin/python calibrate_guncon.py --validate

``--validate`` leaves the current ``GUNCON_CALIB`` in place (does NOT force
identity) and reports the end-to-end error, so after updating config you can
confirm the reticle now lands on target (measured ~= written, error ~= 0).

Requires ``STATE_SLOT`` to hold a gameplay point where the reticle is visible.
"""

import argparse

import numpy as np

import config
from config import (
    RAM,
    CURSOR_X_MIN, CURSOR_X_MAX,
    CURSOR_Y_MIN, CURSOR_Y_MAX,
)


def _norm(v: float, lo: float, hi: float) -> float:
    return (float(v) - lo) / max(1e-9, float(hi - lo))


def _sweep(client, axis: str, warm_peek_frames: int, settle_frames: int):
    """Drive one axis across the screen; return list of (written, measured_norm,
    raw_int) tuples for a forward + reverse pass.

    ``axis`` is "x" or "y"; the other axis is held at center (0.5).
    """
    # Warm up: exit cover so the reticle is controllable, aim at center first.
    client.set_input(shoot=False, peek=True, aim_x=0.5, aim_y=0.5)
    client.step_frames(warm_peek_frames)

    written_points = [round(v, 3) for v in np.linspace(0.1, 0.9, 9)]
    # Forward then reverse to reveal hysteresis / one-frame lag.
    passes = [("fwd", written_points), ("rev", list(reversed(written_points)))]

    records: list[tuple[str, float, float, int]] = []
    for tag, seq in passes:
        for v in seq:
            ax = v if axis == "x" else 0.5
            ay = v if axis == "y" else 0.5
            client.set_input(shoot=False, peek=True, aim_x=ax, aim_y=ay)
            client.step_frames(settle_frames)
            lo = CURSOR_X_MIN if axis == "x" else CURSOR_Y_MIN
            hi = CURSOR_X_MAX if axis == "x" else CURSOR_Y_MAX
            # The reticle RAM word is occasionally read on a transient/flash
            # frame and returns garbage (e.g. 0xFFFF=65535). Retry a few times,
            # stepping a couple frames, until we get a plausible in-range value.
            raw = None
            for _ in range(4):
                cx, cy = client.read_u16_multi([RAM.cursor_x, RAM.cursor_y])
                candidate = int(cx if axis == "x" else cy)
                if lo - 10 <= candidate <= hi + 10:
                    raw = candidate
                    break
                client.step_frames(2)
            if raw is None:
                # Give up on this point; record it as invalid (raw = -1).
                records.append((tag, v, float("nan"), -1))
                continue
            meas = _norm(raw, lo, hi)
            records.append((tag, v, meas, raw))
    return records


def _fit_and_solve(records, axis_label: str):
    """Least-squares fit measured = m*written + c and solve for the correcting
    scale/offset. Returns (m, c, r2, scale, offset)."""
    valid = [r for r in records if r[3] >= 0 and not np.isnan(r[2])]
    written = np.array([r[1] for r in valid], dtype=np.float64)
    measured = np.array([r[2] for r in valid], dtype=np.float64)

    # Guard: a frozen/unresponsive reticle => no usable signal.
    if len(valid) < 3 or float(np.ptp(measured)) < 0.02:
        print(
            f"\n[{axis_label}] WARNING: measured reticle barely moved "
            f"(range {float(np.ptp(measured)):.4f}). The reticle may not be "
            f"controllable in this state -- check STATE_SLOT / peek. Skipping fit."
        )
        return None

    m, c = np.polyfit(written, measured, 1)
    pred = m * written + c
    ss_res = float(np.sum((measured - pred) ** 2))
    ss_tot = float(np.sum((measured - measured.mean()) ** 2)) or 1e-9
    r2 = 1.0 - ss_res / ss_tot

    scale = 1.0 / m if abs(m) > 1e-9 else float("nan")
    offset = (0.5 - c) / m - 0.5 if abs(m) > 1e-9 else float("nan")

    print(f"\n===== {axis_label} axis =====")
    print(f"{'pass':<4} {'written':>8} {'measured':>9} {'raw':>6} {'error':>8}")
    for tag, w, meas, raw in records:
        if raw < 0 or np.isnan(meas):
            print(f"{tag:<4} {w:>8.3f} {'INVALID':>9} {raw:>6d} {'--':>8}")
        else:
            print(f"{tag:<4} {w:>8.3f} {meas:>9.4f} {raw:>6d} {meas - w:>+8.4f}")
    print(
        f"\nfit: measured = {m:.4f} * written + {c:+.4f}   (R^2 = {r2:.4f})"
    )
    print(
        f"recommended GUNCON_CALIB for this axis: "
        f"scale = {scale:.4f}, offset = {offset:+.4f}"
    )
    return m, c, r2, scale, offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate",
        action="store_true",
        help=(
            "Keep the current GUNCON_CALIB (do not force identity) and report "
            "end-to-end error, to confirm a freshly-tuned calibration."
        ),
    )
    args = parser.parse_args()

    if not args.validate:
        # 1. Bypass the existing calibration so set_input writes the RAW aim --
        #    we are characterizing the uncorrected device mapping.
        config.GUNCON_CALIB.update({
            "center_x": 0.5, "center_y": 0.5,
            "scale_x": 1.0, "scale_y": 1.0,
            "offset_x": 0.0, "offset_y": 0.0,
            "min_x": 0.0, "max_x": 1.0, "min_y": 0.0, "max_y": 1.0,
        })
    else:
        print(
            "\n[validate] using the LIVE GUNCON_CALIB from config.py; "
            "measured should track written (error ~= 0) if calibration is good."
        )

    # Import AFTER mutating the calibration dict (bridge_client reads it live per
    # call, so order is not strictly required, but keep intent clear).
    from worker_pool import WorkerPool

    pool = WorkerPool(num_workers=1)
    pool.start()
    env = pool.envs[0]
    client = env.client
    try:
        client.load_state(env.state_slot)
        client.step_frames(4)

        print(
            "\nCalibration probe: driving the GunCon axes to known positions "
            "and reading the on-screen reticle from RAM.\n"
            "(GUNCON_CALIB forced to identity for the measurement.)"
        )

        x_records = _sweep(client, "x", warm_peek_frames=24, settle_frames=6)
        y_records = _sweep(client, "y", warm_peek_frames=12, settle_frames=6)

        x_fit = _fit_and_solve(x_records, "X")
        y_fit = _fit_and_solve(y_records, "Y")

        if args.validate:
            x_err = max((abs(m - w) for _, w, m, r in x_records
                         if r >= 0 and not np.isnan(m)), default=float("nan"))
            y_err = max((abs(m - w) for _, w, m, r in y_records
                         if r >= 0 and not np.isnan(m)), default=float("nan"))
            print("\n===== VALIDATE (live GUNCON_CALIB) =====")
            print(f"  max |error| X = {x_err:.4f}  ({x_err * 100:.2f}% screen)")
            print(f"  max |error| Y = {y_err:.4f}  ({y_err * 100:.2f}% screen)")
            print("  Good calibration => both well under ~0.02 (2% screen).")
        else:
            print("\n===== SUMMARY (recommended GUNCON_CALIB) =====")
            if x_fit is not None:
                _, _, _, sx, ox = x_fit
                print(f'  "scale_x": {sx:.4f}, "offset_x": {ox:+.4f}')
            if y_fit is not None:
                _, _, _, sy, oy = y_fit
                print(f'  "scale_y": {sy:.4f}, "offset_y": {oy:+.4f}')
            print(
                "\nReview these, then update GUNCON_CALIB in config.py if they "
                "look sane (scale near 1.0, high R^2, small residual errors), "
                "then re-run with --validate to confirm end-to-end."
            )
    finally:
        pool.close()


if __name__ == "__main__":
    main()
