# TimeCrisisAISpeedruns

Evolution Strategies (ES) agent that learns to clear Time Crisis (PS1) quickly and without taking damage, driven through BizHawk 2.11.1 via a small Lua<->Python bridge.

## Design summary

- **Algorithm:** Evolution Strategies with mirrored sampling and rank transformation. No gradients, no replay buffer.
- **Decision rate:** every 5 emulator frames (60 Hz game, so ~83 ms per decision).
- **Signal source:** RAM counters found via BizHawk RAM Search, not screen scraping.
- **Fitness:** clear time, with a deliberately harsh penalty on any damage taken (a hit costs ~2.5s of frozen input, which is never worth staying out of cover).

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

| Value | Offset |
|---|---|
| shots_fired | `0x0B1F94` |
| shots_hit | `0x0B1E90` |
| timer | `0x0B1D64` |
| life | `0x0B20C0` |

## Files

| File | Purpose |
|---|---|
| `config.py` | All constants, RAM map, hyperparameters |
| `bridge_client.py` | Python listener/server for the BizHawk `comm.*` bridge |
| `policy.py` | Flat-vector MLP policy (numpy only) |
| `phase_inference.py` | Derived ACTION/WAIT/CUTSCENE/TERMINAL classifier |
| `env_timecrisis.py` | Environment wrapper: reset, step, fitness |
| `logger.py` | CSV logger, flushes每 generation |
| `es_train.py` | ES training loop |
| `run_eval.py` | Load a checkpoint and watch one episode |
| `plot_progress.py` | Plot training curves from the CSV |
| `bizhawk_bridge.lua` | Lua side: RAM reads, input, savestates, HUD |

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
- the logical command set stays line-based (`read_u16`, `set_input`, `step`,
  `load`, `save`, `frame`, `hud`, `hud_clear`)

Before running:

1. Open `/home/runner/work/TimeCrisisAISpeedruns/TimeCrisisAISpeedruns/bizhawk_bridge.lua`.
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

1. Open BizHawk, load Time Crisis, reach Stage 1 Area A start, and **save to slot 1**.
2. Start the Python listener: `python es_train.py`
3. In BizHawk, Tools -> Lua Console -> open `bizhawk_bridge.lua`. It should print
   `[bridge] configured comm target 127.0.0.1:8765 (BizHawk connects out to Python)`.
4. In a second terminal, any time: `python plot_progress.py`

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

## Known limitations (v1)

- **Aim is AI-driven but 1-D.** The bridge now writes the Guncon `P1 X Axis` /
  `P1 Y Axis` via `joypad.setanalog` (overriding the host mouse), and
  `env_timecrisis.py` maps the policy's `aim_bias` output to horizontal aim.
  Vertical stays centered and there is no enemy tracking yet — reactive 2-D
  aiming waits for the vision step.
- **Clear detection is a heuristic** (`timer` jumping upward by >100). It may misfire. Replace it if a real area-clear RAM flag turns up.
