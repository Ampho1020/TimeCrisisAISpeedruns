"""Central configuration. Edit values here, not in the other files."""

from dataclasses import dataclass

# -----------------------------
# Bridge connection (BizHawk Lua)
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

# -----------------------------
# ES hyperparameters
# -----------------------------
POP_SIZE    = 40      # MUST be even (mirrored sampling)
SIGMA       = 0.05    # perturbation scale
ALPHA       = 0.02    # learning rate
GENERATIONS = 200
SEED        = 42
CHECKPOINT_EVERY = 10

# -----------------------------
# Fitness shaping
# -----------------------------
CLEAR_BONUS        = 1000.0
DAMAGE_PENALTY     = 300.0   # deliberately harsh: a hit is never worth it
PARTIAL_HIT_REWARD = 8.0     # only applies to FAILED episodes
FAIL_PENALTY       = 200.0

# -----------------------------
# Policy dims
# obs = [timer_norm, life_norm, fired_norm, hit_norm, acc, last_hit, last_miss]
# act = [shoot_logit, cover_logit, aim_bias]
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
