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
POP_SIZE    = 6      # MUST be even (mirrored sampling)
SIGMA       = 0.05    # perturbation scale
ALPHA       = 0.02    # learning rate
GENERATIONS = 3
SEED        = 42
CHECKPOINT_EVERY = 10

# -----------------------------
# Fitness shaping
# -----------------------------
CLEAR_BONUS        = 1000.0
DAMAGE_PENALTY     = 300.0   # deliberately harsh: a hit is never worth it
PARTIAL_HIT_REWARD = 8.0     # only applies to FAILED episodes
FAIL_PENALTY       = 200.0

# Cover is a HOLD, not a tap: the ~0.2s (~12-frame) in/out traverse is only
# useful if the button is held through it. Reward each committed hold that lasts
# at least COVER_TRAVERSE_TICKS decision ticks so the policy learns to commit to
# cover instead of spamming it every tick. Spam never reaches the threshold, so
# it earns nothing -- we reward holding, not toggling.
COVER_TRAVERSE_TICKS = 3     # ticks (x FRAME_SKIP frames) to clear the traverse
COVER_HOLD_REWARD    = 4.0   # per committed hold; small shaping term

# -----------------------------
# Policy dims
# obs = [timer_norm, life_norm, fired_norm, hit_norm, acc, last_hit, last_miss]
# act = [shoot_logit, cover_logit, aim_bias]
# The bridge now accepts normalized aim_x/aim_y, but policy.py still emits a
# single aim_bias scalar until the vision-based aiming step lands.
# -----------------------------
OBS_DIM = 7
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
