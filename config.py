"""Central configuration. Edit values here, not in the other files."""

from dataclasses import dataclass

# -----------------------------
# Bridge connection (Python listener / BizHawk comm.* target)
# -----------------------------
HOST = "127.0.0.1"
PORT = 8765

# -----------------------------
# RAM map (BizHawk MainRAM domain offsets, all u16)
# -----------------------------
@dataclass(frozen=True)
class RamMap:
    shots_fired: int = 0x0B1F94
    shots_hit:   int = 0x0B1E90
    timer:       int = 0x0B1D64
    life:        int = 0x0B20C0

RAM = RamMap()

# -----------------------------
# Episode / stepping
# -----------------------------
FRAME_SKIP = 5        # emulator frames per decision (60Hz -> ~83ms)
MAX_TICKS  = 900      # 900 * 5 = 4500 frames hard cap (~75s)
STATE_SLOT = 1        # savestate slot holding Stage 1 Area A start

# Time Crisis counts the area timer DOWN; when it hits zero the game drops to a
# "continue?" screen. That is a SEPARATE terminal from life == 0 (being shot),
# and if we don't detect it the episode idles on the continue screen until
# MAX_TICKS, spamming inputs into a frozen state (the "repetitive" behavior).
# We catch the moment the timer reaches this threshold so the episode ends as a
# timeout failure. If you locate a dedicated continue-screen RAM flag, wire it in
# via RamMap and prefer it -- this timer zero-cross is the address-free fallback.
TIMEOUT_TIMER_THRESHOLD = 0

# -----------------------------
# ES hyperparameters
# -----------------------------
POP_SIZE    = 30      # MUST be even (mirrored sampling)
SIGMA       = 0.05    # perturbation scale
ALPHA       = 0.02    # learning rate
GENERATIONS = 100
SEED        = 42
CHECKPOINT_EVERY = 10

# -----------------------------
# Parallel training
# -----------------------------
# Run several BizHawk instances at once, one per worker, each on its own port
# (BASE_PORT + worker_index). The population is split across the workers and
# evaluated concurrently, so wall-clock per generation drops by ~NUM_WORKERS.
# Set NUM_WORKERS = 1 for the original single-instance flow.
#
# One emulator per worker is heavy (CPU + RAM + a window each). Start with a few
# and raise toward POP_SIZE only if your machine keeps up.
NUM_WORKERS = 4
BASE_PORT   = PORT            # worker i listens on BASE_PORT + i

# Auto-launch: if True, training spawns the BizHawk instances for you (no manual
# per-window commands). Fill in the paths for your machine. If False, launch
# NUM_WORKERS instances yourself, each with --socket_port = BASE_PORT + i, and
# Python just connects to them.
AUTO_LAUNCH_BIZHAWK = True
BIZHAWK_LAUNCH      = "/home/ampho/Downloads/BizHawk/BizHawk-2.11.1-linux-x64/./EmuHawkMono.sh"  # EmuHawk launcher (Linux: EmuHawkMono.sh)
BIZHAWK_ROM         = "/home/ampho/Downloads/TimeCrisis_NTSC/Time Crisis.cue"                  # absolute path to the Time Crisis disc image
BIZHAWK_LUA         = "/home/ampho/TimeCrisisAISpeedruns/bizhawk_bridge.lua"  # absolute path recommended
BIZHAWK_EXTRA_ARGS  = []                  # any extra EmuHawk CLI flags

# -----------------------------
# Fitness shaping
# -----------------------------
CLEAR_BONUS        = 1000.0
DAMAGE_PENALTY     = 300.0   # deliberately harsh: a hit is never worth it
PARTIAL_HIT_REWARD = 8.0     # only applies to FAILED episodes
FAIL_PENALTY       = 200.0

# Cover is a HOLD, not a tap: the ~0.2s (~12-frame) in/out traverse is only
# useful if the button is held through it. We reward holding DENSELY -- each
# extra consecutive held tick, up to COVER_TRAVERSE_TICKS, adds COVER_HOLD_REWARD
# -- so evolution gets a smooth gradient to climb from spamming toward holding.
# A length-1 tap earns nothing and holding past the traverse is capped (no
# camping). It never penalizes short/partial holds, so it won't punish the
# future half-out "peek" the agent uses to read incoming bullets.
COVER_TRAVERSE_TICKS = 3     # ticks (x FRAME_SKIP frames) to clear the traverse
COVER_HOLD_REWARD    = 12.0  # per extra held tick, up to the traverse; capped
COVER_FLIP_PENALTY   = 10.0  # subtracted from fitness per cover state toggle
COVER_TIME_PENALTY   = 0.3   # per tick spent in cover; discourages camping
SHOT_FIRED_REWARD    = 3.0   # per shot attempted; incentivises exposing to shoot

# -----------------------------
# Policy dims
# obs = [timer_norm, life_norm, fired_norm, hit_norm, acc, last_hit, last_miss, cover_phase]
# cover_phase in [-1, +1]: sign = current cover state, magnitude = ticks_held / COVER_TRAVERSE_TICKS
# act = [shoot_logit, cover_logit, aim_bias]
# The bridge now accepts normalized aim_x/aim_y, but policy.py still emits a
# single aim_bias scalar until the vision-based aiming step lands.
# -----------------------------
OBS_DIM = 8
HIDDEN  = 64
ACT_DIM = 3

# -----------------------------
# Logging / feedback
# -----------------------------
VERBOSE_EPISODES = True
LOG_CSV = "training_log.csv"
HUD_ENABLED = True    # draw status text on the emulator window

# -----------------------------
# Guncon lightgun calibration
#
# Nymashock has no built-in offset/scale UI (unlike DuckStation), so we
# correct aim in software before writing X/Y to the Guncon port.
#
# From DuckStation testing: scaling the X axis to 94% of screen width fixes
# the edge drift flawlessly. Y axis needs no correction.
#
# Correction is applied about screen center:
#   x_corr = center + (x - center) * scale_x + offset_x
#   y_corr = center + (y - center) * scale_y + offset_y
#
# Axes are normalized to [0.0, 1.0] (0 = left/top, 1 = right/bottom),
# so center = 0.5.
# -----------------------------
GUNCON_CALIB = {
    "center_x": 0.5,
    "center_y": 0.5,
    "scale_x": 0.94,   # DuckStation-verified: X axis to 94%
    "scale_y": 1.0,    # Y already perfect
    "offset_x": 0.0,
    "offset_y": 0.0,
}
