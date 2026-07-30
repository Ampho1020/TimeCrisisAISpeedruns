# TimeCrisisAISpeedruns

Evolution Strategies (ES) agent that learns to clear Time Crisis (PS1) quickly and without taking damage, driven through BizHawk via a small Lua<->Python bridge.

## Design summary

- **Algorithm:** Evolution Strategies with mirrored sampling and rank transformation. No gradients, no replay buffer.
- **Decision rate:** every 5 emulator frames (60 Hz game, so ~83 ms per decision).
- **Signal source:** RAM counters found via BizHawk RAM Search, not screen scraping.
- **Fitness:** clear time, with a deliberately harsh penalty on any damage taken (a hit costs ~2.5s of frozen input, which is never worth staying out of cover).

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
| `bridge_client.py` | TCP line-protocol client for the Lua bridge |
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

## Run order

1. Open BizHawk, load Time Crisis, reach Stage 1 Area A start, **save to slot 1**.
2. Tools -> Lua Console -> open `bizhawk_bridge.lua`. It should print `[bridge] listening on 127.0.0.1:8765`.
3. **Edit `apply_input()` in the Lua file** so the button names match your real shoot/cover bindings. This is the one thing guaranteed to need adjusting.
4. `python es_train.py`
5. In a second terminal, any time: `python plot_progress.py`

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

- **`aim_bias` is inert.** The Lua side accepts it and drops it, so the agent currently chooses only shoot / cover / wait with aim fixed. Expect an early fitness plateau; this is not a hyperparameter problem.
- **Clear detection is a heuristic** (`timer` jumping upward by >100). It may misfire. Replace it if a real area-clear RAM flag turns up.
- **No vision yet.** Next step is enemy detection feeding target-slot scores in place of the single `aim_bias` output.
