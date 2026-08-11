# TimeCrisisAISpeedruns

Evolution Strategies (ES) agent that learns to clear Time Crisis (PS1) quickly
and without taking damage, driven through BizHawk 2.11.1 via a Lua↔Python
socket bridge.

## Design summary

- **Algorithm:** Evolution Strategies with mirrored sampling and rank
  transformation. No gradients, no replay buffer.
- **Decision rate:** every 5 emulator frames (60 Hz game, ≈83 ms per decision).
- **Signal source:** RAM counters found via BizHawk RAM Search — no screen
  scraping.
- **Fitness:** clear time, with a harsh penalty on any damage taken and a bonus
  per confirmed hit. A hit is never worth staying exposed.
- **Parallelism:** `NUM_WORKERS` BizHawk instances are evaluated concurrently
  (one per socket port), reducing wall-clock time per generation roughly
  linearly. The emulators can be auto-launched or managed manually.

## RAM map (BizHawk `MainRAM` domain, all u16)

| Value | Offset |
|---|---|
| shots_fired | `0x0B1F94` |
| shots_hit | `0x0B1E90` |
| timer | `0x0B1D64` |
| life | `0x0B20C0` |
| cursor_x | `0x0B1C74` |
| cursor_y | `0x0B1C78` |

`cursor_x` / `cursor_y` are the on-screen Guncon reticle coordinates (X: 1–259,
Y: 1–232). They are normalized to [0, 1] and fed into the policy so it can
correlate where rewarded hits happened with where the reticle actually was.

## Policy observation / action space

```
obs (13-D):
  timer_norm, life_norm, fired_norm, hit_norm, accuracy,
  last_hit_flag, last_miss_flag, peek_phase,
  ammo_norm, prev_aim_x_bias, prev_aim_y_bias,
  cursor_x_norm, cursor_y_norm

act (4-D):
  shoot_logit, cover_logit, aim_x_bias, aim_y_bias
```

`peek_phase` encodes both direction (`±1`) and hold progress (magnitude grows
toward 1 over `PEEK_TRAVERSE_TICKS` ticks). `ammo_norm` tracks a software clip
counter mirroring Time Crisis' 6-round Guncon clip; the policy is hard-forced to
duck when `ammo_left == 0` so it can reload. `prev_aim_x_bias` /
`prev_aim_y_bias` feed the previous tick's aim back in so the memoryless MLP can
learn to shift aim across ticks.

## Files

| File | Purpose |
|---|---|
| `config.py` | All constants — RAM map, ES hyperparameters, fitness weights, Guncon calibration |
| `bridge_client.py` | Python TCP listener; speaks the BizHawk `comm.*` length-framed protocol |
| `policy.py` | Two-layer MLP policy (NumPy only) |
| `phase_inference.py` | Majority-vote ACTION/WAIT/CUTSCENE/TERMINAL classifier |
| `env_timecrisis.py` | Environment wrapper: reset, step, fitness shaping |
| `worker_pool.py` | `WorkerPool` — parallel evaluation across N emulator workers |
| `logger.py` | CSV logger, flushes every generation |
| `es_train.py` | ES training loop with stagnation kick |
| `run_eval.py` | Load a checkpoint and watch one episode |
| `plot_progress.py` | Plot training curves from the CSV |
| `bizhawk_bridge.lua` | Lua side: RAM reads, input injection, savestates, HUD |
| `tests/` | Unit tests (`test_bridge_client.py`, `test_simulation.py`) |

## Setup (Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the unit tests to verify the setup:

```bash
python -m unittest discover -s tests -v
```

## BizHawk / bridge setup

This project targets **BizHawk 2.11.1** with the **Nymashock** PSX core.
The bridge uses BizHawk's native `comm.socketServer*` API — no LuaSocket, no
`--socket_ip` / `--socket_port` launch flags required for single-worker mode.

**One-time calibration steps:**

1. Open `bizhawk_bridge.lua` and fill in the clearly-marked constants with the
   exact Guncon key names your BizHawk/Nymashock build reports:
   - `GUNCON_TRIGGER_KEY`
   - `GUNCON_AIM_X_KEY`
   - `GUNCON_AIM_Y_KEY`
   - optionally `GUNCON_COVER_BUTTON_KEY`

   If you don't know the exact names, uncomment the diagnostic block at the top
   of `apply_input()`, run one frame, and copy the printed `joypad.get()` /
   `input.get()` key names back.

2. Check `GUNCON_AXIS_MIN` / `GUNCON_AXIS_MAX` against the axis range your
   Nymashock build expects.

3. Aim correction is applied in Python via `GUNCON_CALIB` in `config.py`
   (default: X axis scaled to 94 %, Y unchanged — matches DuckStation-verified
   behaviour; adjust if your Nymashock build drifts differently).

**Parallel workers** (`NUM_WORKERS > 1`):

Set `AUTO_LAUNCH_BIZHAWK = True` in `config.py` and fill in `BIZHAWK_LAUNCH`,
`BIZHAWK_ROM`, and `BIZHAWK_LUA` with the absolute paths for your machine.
`WorkerPool` will spawn `NUM_WORKERS` emulators automatically, one per port
(`BASE_PORT` + `i`), and wait for them to connect.

If `AUTO_LAUNCH_BIZHAWK = False`, launch the emulators yourself — one window per
worker — each pointed at a unique `BASE_PORT + i` port.

## Run order

1. Open BizHawk, load Time Crisis, reach Stage 1 Area A start, and
   **save to slot 1** (`STATE_SLOT = 1` in `config.py`).
2. Start training: `python es_train.py`  
   (with `AUTO_LAUNCH_BIZHAWK = True` this spawns all emulators for you).
3. For manual/single-worker mode: in BizHawk go to Tools → Lua Console → open
   `bizhawk_bridge.lua`. It should print  
   `[bridge] configured comm target 127.0.0.1:8765 (BizHawk connects out to Python)`.
4. In a second terminal, at any time: `python plot_progress.py`

Evaluate a saved checkpoint:

```bash
python run_eval.py theta_gen_050.npy
```

## Shakedown run first

Before committing to a long run, set `POP_SIZE = 6` and `GENERATIONS = 3` in
`config.py`. This confirms:

- the bridge survives thousands of round trips
- savestate reload resets cleanly every episode
- fitness `std` is non-zero (hyperparameters in the right ballpark)
- clear detection fires (or doesn't)

If `std` is 0 or every episode reports an identical `elapsed`, stop — that is a
reset or clear-detection bug, far easier to diagnose on 18 episodes than 8 000.

## Reading the logs

`std` (fitness standard deviation across the population) is the key diagnostic:

| Symptom | Meaning | Fix |
|---|---|---|
| `std` ~ 0 | Perturbations don't change behavior | `SIGMA` too small; raise it (0.05 → 0.1) |
| `std` huge, `mean` bouncing | Update steps overshooting | Lower `ALPHA` (0.02 → 0.005) |
| `std` fine, `mean` flat 30+ gens | Stuck in a local optimum | Stagnation kick should fire; if not, lower `STAGNATION_PATIENCE` |
| `mean` rising slowly and steadily | Working — leave it alone | — |

**Stagnation kick:** if `std` stays below `STD_STAGNATION_THRESHOLD` for
`STAGNATION_PATIENCE` consecutive generations, `es_train.py` temporarily
multiplies `SIGMA` by `STAGNATION_SIGMA_MULT` for that generation to reintroduce
variance. It reverts as soon as `std` recovers.

Expect the first ~20 generations to look like nothing is happening. That is
normal for ES.

## Continue-screen watchdog

Time Crisis counts the area timer **down**. If the timer reaches zero (or the
agent's life reaches zero), the game drops to a "continue?" screen and freezes
all useful RAM. Two safeguards prevent the episode from idling forever:

1. **Timer threshold** — any tick where `timer ≤ TIMEOUT_TIMER_THRESHOLD` (60
   by default, i.e. one second of game time) is treated as a timeout terminal.
2. **Stale-counter watchdog** — if `shots_fired`, `shots_hit`, `timer`, and
   `life` are all identical for `CONTINUE_SCREEN_STALE_TICKS` consecutive
   decision ticks, the episode is force-terminated.

## Known limitations

- **No enemy-position RAM.** The cursor RAM (`cursor_x` / `cursor_y`) tells the
  policy where the reticle is, but enemy positions still come from no dedicated
  address. 2-D reactive aiming awaits the next RAM-search pass.
- **Clear detection is a heuristic.** The area-clear check is a `timer` upward
  jump of >100 ticks. It may misfire. Replace with a dedicated RAM flag if one
  is found.
