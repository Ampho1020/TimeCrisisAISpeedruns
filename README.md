# TimeCrisisAISpeedruns

Evolution Strategies (ES) agent that learns to clear Time Crisis (PS1) quickly and without taking damage, driven through BizHawk 2.11.1 via a small Lua<->Python bridge.

## Design summary

- **Algorithm:** Evolution Strategies with mirrored sampling and rank transformation. No gradients, no replay buffer.
- **Decision rate:** every 5 emulator frames (60 Hz game, so ~83 ms per decision).
- **Signal source:** RAM counters (found via BizHawk RAM Search) for reward/state, plus a YOLO enemy detector over captured frames that conditions aim and shooting (see *Policy modes* and *Vision + detector*).
- **Fitness:** clear time, with a deliberately harsh penalty on any damage taken (a hit costs ~2.5s of frozen input, which is never worth staying out of cover), plus a bonus that scales with the number of screens cleared.

## Vision class taxonomy

- Detector/vision-schedule class IDs are now a 3-class schema:
   - `0 = ENEMY`
   - `1 = GRENADE`
   - `2 = PROJECTILE`
- This is a breaking change for `vision_schedule` checkpoints because the
   class-priority tail in theta changed dimensionality.
- Existing `.npy` checkpoints from the old 4-class schema are not compatible
   with the current config and should be regenerated.

## RAM map (BizHawk `MainRAM` domain, all u16)

| Value | Offset | Notes |
|---|---|---|
| shots_fired | `0x0B1F94` | |
| shots_hit | `0x0B1E90` | |
| timer | `0x0B1D64` | counts down; area-clear bumps it up |
| life | `0x0B20C0` | `0` = death terminal |
| cursor_x | `0x0B1C74` | on-screen reticle X, observed range `1..259` |
| cursor_y | `0x0B1C78` | on-screen reticle Y, observed range `1..232` |
| ammo | `0x0B1DDC` | rounds left in the current clip (read live, not simulated) |

The cursor and ammo addresses feed the observation/aim and the
cover-enforcement logic directly. Ammo is read straight from RAM every frame
(the old software ammo tracker was removed), so reload/refill timing always
matches the real game.

## Files

| File | Purpose |
|---|---|
| `config.py` | All constants, RAM map, hyperparameters, GunCon calibration |
| `bridge_client.py` | Python listener/server for the BizHawk `comm.*` bridge |
| `policy.py` | Flat-vector policies: MLP, open-loop schedule, vision-conditioned schedule |
| `detector.py` | Enemy detector backends (Torch/GPU, ONNX, classical CV) + `build_detector()` |
| `vision.py` | numpy-only image decode + classical blob detection helpers |
| `phase_inference.py` | Derived ACTION/WAIT/CUTSCENE/TERMINAL classifier |
| `env_timecrisis.py` | Environment wrapper: reset, step, fitness |
| `worker_pool.py` | Parallel evaluation across N BizHawk instances |
| `logger.py` | CSV logger, flushes each generation |
| `es_train.py` | ES training loop |
| `run_eval.py` | Load a checkpoint and watch one episode |
| `inspect_vision.py` | Visualize detector output on live frames or dumped PNGs |
| `calibrate_guncon.py` | Automated GunCon X/Y calibration probe (measures + solves the transform) |
| `plot_progress.py` | Plot training curves from the CSV |
| `bizhawk_bridge.lua` | Lua side: RAM reads, input, savestates, screenshots, HUD |

## Setup (Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## BizHawk / bridge setup

This repository targets **BizHawk 2.11.1** with the **Nymashock** PSX core.
The Lua bridge no longer uses LuaSocket; it uses BizHawk's native
`comm.socketServer*` API, which means:

- the Python process listens on `HOST` / `PORT` from `config.py`
- BizHawk connects out to that listener from `bizhawk_bridge.lua`
- `bizhawk_bridge.lua` calls `comm.socketServerSetIp(HOST)` and
  `comm.socketServerSetPort(PORT)` directly, so separate BizHawk
  `--socket_ip` / `--socket_port` launch flags are not required
- the logical command set stays line-based: `read_u16`, `read_u16_multi`
  (batched multi-address read in one round trip), `set_input`, `input_state`,
  `step`, `load`, `save`, `frame`, `screenshot` (returns an encoded frame for
  the vision pipeline), `hud`, `hud_clear`

Before running:

1. Open `bizhawk_bridge.lua` from the repository root.
2. Replace these clearly-marked constants with the exact Guncon key names your
   BizHawk/Nymashock build reports:
   - `GUNCON_TRIGGER_KEY`
   - `GUNCON_AIM_X_KEY`
   - `GUNCON_AIM_Y_KEY`
   - optionally `GUNCON_COVER_BUTTON_KEY` if your setup uses a dedicated cover button
3. If you do not know the exact names, use the commented diagnostic block at the
   top of `apply_input()`: uncomment it temporarily, run one frame, and copy the
   printed `joypad.get()` / `input.get()` key names back into the constants.
4. If your build needs different axis bounds, adjust `GUNCON_AXIS_MIN` /
   `GUNCON_AXIS_MAX` after checking the values your Guncon fields expect.

## Run order

With `AUTO_LAUNCH_BIZHAWK = True` (the default) `es_train.py` spawns and manages
the BizHawk instance(s) itself — one per worker up to `NUM_WORKERS` — loading the
ROM and `bizhawk_bridge.lua` automatically.

1. Open BizHawk once, load Time Crisis, reach the Stage 1 Area A start, and
   **save to slot 1**. Every episode resets to this savestate.
2. Start training: `python es_train.py`. It launches BizHawk, which connects
   back to the Python listener and prints
   `[bridge] configured comm target 127.0.0.1:8765 (BizHawk connects out to Python)`.
3. In a second terminal, any time: `python plot_progress.py`.

Set `AUTO_LAUNCH_BIZHAWK = False` to drive an already-open BizHawk manually
instead: start `python es_train.py`, then in BizHawk open
Tools -> Lua Console -> `bizhawk_bridge.lua`.

Evaluate a checkpoint:

```bash
python run_eval.py theta_gen_050.npy
```

## Shakedown run first

Before committing to a long run, set `POP_SIZE = 6` and `GENERATIONS = 3` in `config.py`. This confirms:

- the bridge survives thousands of round trips
- savestate reload resets cleanly every episode
- fitness `std` is non-zero (hyperparameters in the right range)
- clear detection fires (or doesn't)

If `std` is 0 or every episode reports an identical `elapsed`, stop — that's a reset or clear-detection bug, and it's far easier to diagnose on 18 episodes than 8000.

## Reading the logs

`std` (fitness standard deviation across the population) is the key diagnostic:

| Symptom | Meaning | Fix |
|---|---|---|
| `std` ~ 0 | Perturbations don't change behavior | Raise `SIGMA` (0.05 -> 0.1) |
| `std` huge, `mean` bouncing | Update steps overshooting | Lower `ALPHA` (0.02 -> 0.005) |
| `std` fine, `mean` flat 30+ gens | Genuinely stuck | Raise `SIGMA` slightly |
| `mean` rising slowly and steadily | Working — leave it alone | — |

Expect the first ~20 generations to look like nothing is happening. That's normal for ES.

## Policy modes

`POLICY_MODE` in `config.py` selects the policy shape (all are flat numpy
vectors evolved by ES):

- `mlp` — small feed-forward net over the RAM-derived observation (original v1).
- `schedule` — open-loop per-tick action schedule tied to the slot-1 savestate.
- `vision_schedule` — **current default.** The open-loop schedule, but aim and
  shoot are blended with live enemy detections so the agent tracks targets in
  2-D instead of firing on a fixed timeline. An *aim-on-target* gate
  (`AIM_ON_TARGET_RADIUS`) suppresses shots whose reticle is too far from the
  chosen enemy so ammo is not wasted mid-sweep.

## Vision + detector

`detector.py`/`build_detector()` picks the best available backend at runtime and
falls through automatically:

1. `TorchYoloDetector` — Ultralytics YOLO on `best.pt` (CUDA/GPU when available); the production backend.
2. `ONNXDetector` — `best.onnx` via onnxruntime.
3. `ClassicalDetector` — pure classical CV (`vision.py`), always available as a fallback.

The detector emits the 3-class schema (`ENEMY`/`GRENADE`/`PROJECTILE`, see
*Vision class taxonomy*). Use `inspect_vision.py` to overlay detections on live
frames or dumped PNGs when debugging.

The Torch/ONNX backends need extra packages beyond `requirements.txt`
(`ultralytics` and a matching `torch`); install them if you want the GPU
detector, otherwise the classical fallback runs CPU-only.

## GunCon calibration

The reticle the emulator draws does not line up 1:1 with the normalized aim the
policy emits, so `GUNCON_CALIB` in `config.py` applies a per-axis affine
correction:

```
x_corr = center + (x - center) * scale_x + offset_x
y_corr = center + (y - center) * scale_y + offset_y
```

The shipped values are empirically measured (not guessed). Re-measure them for a
different build with `calibrate_guncon.py`, which fires a grid of shots, reads
back `cursor_x`/`cursor_y`, and solves the transform; pass `--validate` to
verify an existing calibration instead of overwriting it.

## Known limitations

- **Open-loop backbone.** Even in `vision_schedule` the timing skeleton is a
  fixed per-tick schedule tied to the slot-1 savestate. It generalizes to the
  trained area but is not a fully closed-loop controller.
- **Clear detection is a heuristic** (`timer` jumping upward past
  `SCREEN_CLEAR_TIMER_BUMP`, tick-granular). It may misfire. Replace it if a
  real area-clear RAM flag turns up.
- **Detector quality bounds behavior.** Aim/shoot are only as good as `best.pt`;
  noisy detections (especially GRENADE vs PROJECTILE) show up as wasted shots or
  missed threats.
