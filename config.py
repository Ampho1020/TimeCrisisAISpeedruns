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
# Set with margin above the true zero: at FRAME_SKIP-sized steps a fast tick can
# jump clean past 0 without ever sampling it, missing the exact zero-cross. No
# clear is ever legitimately optimal with under a second left, so treating
# anything <= 60 (one second at 60Hz) as "timed out" catches the drop reliably.
TIMEOUT_TIMER_THRESHOLD = 60

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
HIT_REWARD         = 5.0     # per confirmed hit, all episodes; teaches aim
FAIL_PENALTY       = 200.0

# Peeking out is a HOLD, not a tap: the ~0.2s (~12-frame) in/out traverse only
# completes if the button is held through it. PEEK_TRAVERSE_TICKS is a game-
# mechanics constant (minimum hold lock so a transition can't be reversed
# mid-animation; also gates when shots are allowed to register) -- it is NOT
# a reward shaping knob. We used to also reward holding densely
# (COVER_HOLD_REWARD), penalize flip-flopping (COVER_FLIP_PENALTY) and camping
# (COVER_TIME_PENALTY), and give extra hit credit only on failed episodes
# (PARTIAL_HIT_REWARD). All four were removed: they just layered noisy shaping
# on top of the raw ES signal instead of letting evolution optimize the actual
# outcome (clear/fail, elapsed time, damage, hits). peek_flips/peek_hold/
# cover_time are still tracked and logged for diagnostics, just no longer fed
# into fitness.
PEEK_TRAVERSE_TICKS = 3     # ticks (x FRAME_SKIP frames) to clear the traverse

# NOTE: we deliberately do NOT reward raw shots fired (nor per-shot reload/
# active-fire bonuses). Any reward that scales with shot COUNT is a magdump
# hack -- the policy learns to spam the trigger for free reward regardless of
# whether it hits anything. Only HIT_REWARD (accuracy) and RELOAD_BONUS (a
# flat, count-independent event reward, see below) touch shooting behaviour.

# Rounds per clip. Time Crisis' Guncon always starts a screen with a full 6-
# round clip; we mirror that in software (ammo_left is not read from RAM --
# there's no known counter for it) so the policy can observe when it's about
# to run dry and learn to duck instead of dry-firing.
AMMO_MAX_ROUNDS = 6

# Penalty per tick the agent is fully exposed with an EMPTY clip (ammo_left
# == 0 at the start of the tick) instead of ducking back into cover to
# reload. This no longer fires just because the agent chose not to shoot --
# only true "should have ducked, gun is empty" ticks count.
DRY_FIRE_PENALTY = 2.0

# Flat bonus (NOT scaled by shots fired) awarded exactly once, on the tick
# the agent ducks back into cover with an empty clip (ammo_left == 0).
# Reinforces the "empty clip -> duck to reload" loop without rewarding shot
# count itself, so it can't be farmed by magdumping.
RELOAD_BONUS = 15.0

# -----------------------------
# Policy dims
# obs = [timer_norm, life_norm, fired_norm, hit_norm, acc, last_hit, last_miss,
#        cover_phase, ammo_norm, prev_aim_x_bias, prev_aim_y_bias]
# peek_phase in [-1, +1]: sign = current peek state, magnitude = ticks_held / PEEK_TRAVERSE_TICKS
# ammo_norm = ammo_left / AMMO_MAX_ROUNDS, in [0, 1]
# prev_aim_x_bias / prev_aim_y_bias in [-1, 1]: the aim_x_bias/aim_y_bias the
# policy itself output on the PREVIOUS tick, fed back in as the next tick's
# input. The net is otherwise purely feedforward/memoryless, so without this
# it has no way to know what it last chose -- this closes that loop, letting
# weights learn to shift aim across ticks instead of latching onto one spot.
# act = [shoot_logit, cover_logit, aim_x_bias, aim_y_bias]
# Both aim axes are policy-controlled: bias in [-1, 1] -> screen position in [0, 1].
# -----------------------------
OBS_DIM = 11
HIDDEN  = 64
ACT_DIM = 4

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
