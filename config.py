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
    cursor_x:    int = 0x0B1C74
    cursor_y:    int = 0x0B1C78

RAM = RamMap()

# -----------------------------
# Episode / stepping
# -----------------------------
FRAME_SKIP = 5        # emulator frames per decision (60Hz -> ~83ms)
MAX_TICKS  = 900      # 900 * 5 = 4500 frames hard cap (~75s)
STATE_SLOT = 1        # savestate slot holding Stage 1 Area A start

# On-screen gun cursor coordinates (confirmed RAM-backed screen-space position,
# not enemy positions). Values observed in BizHawk on this setup:
#   X: 1..259
#   Y: 1..232
# We normalize both to [0, 1] before feeding them into the policy.
CURSOR_X_MIN = 1
CURSOR_X_MAX = 259
CURSOR_Y_MIN = 1
CURSOR_Y_MAX = 232

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

# Fallback continue/menu detector: if all sampled core counters
# (shots_fired/shots_hit/timer/life) are identical for this many consecutive
# decision ticks, treat it as a frozen non-playable terminal and force episode
# termination/reset. This catches missed life==0 / timer-zero transitions under
# frame-skip or RAM sampling gaps.
CONTINUE_SCREEN_STALE_TICKS = 3

# -----------------------------
# ES hyperparameters
# -----------------------------

# Which action-selection paradigm theta represents:
#   "mlp"      -- closed-loop: theta is feedforward-net weights, action
#                 computed from the live observation every tick (policy.act).
#   "schedule" -- open-loop: theta is a fixed per-tick action TABLE indexed
#                 directly by the current tick, no observation consumed for
#                 action selection at all (policy.act_schedule). This suits
#                 this project's actual goal (one fixed savestate/level, not
#                 a generalist policy) -- see repo memory "Open-loop schedule
#                 search (2026-08-09): POSITIVE result" for the sim probe
#                 that validated it: at GENERATIONS=80 schedule search won
#                 5/5 seeds on fitness vs the closed-loop policy above
#                 (clear_rate 0.667 vs 0.162, final_best 3170 vs -62.9).
#                 Only validated at probe scale (pop=12) so far, not yet at
#                 live POP_SIZE=30 -- watch early live generations closely.
POLICY_MODE = "schedule"

POP_SIZE    = 30      # MUST be even (mirrored sampling)
# SIGMA raised 0.05 -> 0.1 (2026-08-04): in-sim trend testing
# (tests/test_simulation.py ExtendedMiniESTrendSuite) showed the population
# can collapse into a "never expose" local optimum -- staying in cover is a
# free, zero-risk -FAIL_PENALTY baseline, while a botched exposure risks the
# much harsher DAMAGE_PENALTY. Once every mirrored candidate's peek-logit
# lands on the same side of 0, fitness std hits exactly 0.00 and the ES
# gradient carries no signal, so theta can no longer move on its own. A
# larger SIGMA makes it more likely at least some candidates flip sign again.
SIGMA       = 0.1     # perturbation scale
ALPHA       = 0.02    # learning rate

# Episodes evaluated per candidate per generation, averaged before ranking
# (mirrors run_timed_spot_probe's episodes_per_candidate in
# tests/test_simulation.py). Added 2026-08-05: single-episode fitness is
# noisy enough that rank-transform ES (which only uses ORDER) can flip a
# genuinely-better candidate below a worse one just from one unlucky
# episode's stochastic hit/damage rolls. A sim A/B probe (5 seeds x 40
# gens, see repo memory "Multi-episode fitness averaging probe") showed
# averaging 3 episodes/candidate fixed two seeds that otherwise collapsed
# to ~8% clear rate, and improved every aggregate metric (clear_rate,
# mean/best accuracy, mean hits) with no seed getting more than a small dip
# worse. Set to 1 to fully disable (exact prior single-episode behavior).
#
# Reverted 3 -> 1 for schedule mode (2026-08-10): ran a full live 80-gen
# EPISODES_PER_CANDIDATE=3 pass and then directly re-evaluated the resulting
# theta_final.npy for 16 real BizHawk episodes -- fitness/clear/damage/acc
# were BIT-IDENTICAL across all 16 (std=0.0 exactly). Schedule mode's
# open-loop action table is indexed only by tick count (never reacts to
# observations), and the emulator is deterministic from a fixed savestate,
# so a fixed schedule-mode theta has ZERO episode-to-episode variance.
# EPISODES_PER_CANDIDATE's whole purpose is averaging out per-episode
# stochastic noise for a FIXED candidate -- with zero variance to average
# out, it was pure wasted compute (3x wall-clock) for schedule mode. That
# rationale (repo memory's "Multi-episode fitness averaging probe") was
# validated for "mlp" (closed-loop, reactive) mode, which may still have
# real timing/observation jitter -- raise this again if POLICY_MODE is
# switched back to "mlp".
EPISODES_PER_CANDIDATE = 1

# GENERATIONS raised back up for live convergence passes (2026-08-08).
GENERATIONS = 80
SEED        = 10
CHECKPOINT_EVERY = 5

# Stagnation kick (see SIGMA note above): if fitness std stays below
# STD_STAGNATION_THRESHOLD for STAGNATION_PATIENCE consecutive generations,
# es_train.py temporarily samples that generation's population with
# SIGMA * STAGNATION_SIGMA_MULT instead of SIGMA, to try to reintroduce
# variance and escape the flat/no-gradient trap. Reverts to normal SIGMA as
# soon as std recovers above the threshold for one generation.
#
# Values tuned against the harder 12-enemy/3-target sim model
# (tests/test_simulation.py ExtendedMiniESTrendSuite, 2026-08-04): the
# tougher task made the ES population stall into flat multi-generation
# "never expose" collapses more often than the original 6-enemy sim. A/B
# comparison across 5 seeds x 80 generations (patience=3/mult=3.0 baseline vs
# patience=2/mult=5.0) showed reacting sooner (patience 3 -> 2) with a
# stronger kick (mult 3.0 -> 5.0) meaningfully improves outcomes: more
# generations reaching a clear (10.4 vs 8.2 avg), fewer flat/collapsed
# generations (13.4 vs 15.6 avg), and a less negative final-generation mean
# fitness (-299.3 vs -633.5 avg). Not a full grid search -- if real training
# still stalls often, re-run that A/B experiment (see repo memory) with a
# wider sweep before hand-tuning further.
STD_STAGNATION_THRESHOLD = 1e-3
STAGNATION_PATIENCE      = 2     # consecutive flat generations before kicking
STAGNATION_SIGMA_MULT    = 5.0   # SIGMA multiplier while kicking

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

# All EmuHawk instances on a machine share a single config.ini in the BizHawk
# install directory (confirmed: only one config.ini exists, no per-instance
# copy). Launching NUM_WORKERS instances back-to-back with zero delay lets them
# race on reading/writing that shared file during startup, which crashed a real
# 4-worker run within the first tick (bridge_client got ConnectionResetError
# right after all 4 handshakes succeeded -- one instance died moments later).
# A short stagger between launches avoids the race. Raise this if instances
# still crash shortly after startup; 0 restores the old (unstaggered) behavior.
BIZHAWK_LAUNCH_STAGGER_SECONDS = 3.0

# Warm-start scale for the two shot-phase input rows in es_train.py's w1.
# 0.0 keeps the strict zero-init behavior (gen0 exactly matches pre-port).
# >0 seeds the new shot-phase channels with small non-zero weights so arc
# variation can show up earlier instead of waiting for many generations to
# discover/use those dims from pure perturbation noise.
SHOT_PHASE_WARMSTART_ROW_STD = 0.08

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
#
# NOTE (2026-08-04): env_timecrisis.py's step() now HARD-ENFORCES the duck
# the instant ammo_left hits 0 (overrides the policy's own peek output --
# see the comment there), because relying on this penalty alone to teach
# that behavior kept failing in real training (agents mag-dumped and stayed
# exposed anyway). With the override in place, dry_fire_ticks should always
# be 0 in practice -- this penalty is now a harmless backstop/diagnostic, not
# the primary mechanism. Left in place in case the override ever has a gap.
DRY_FIRE_PENALTY = 2.0

# Miss-correction shaping: reward sequences that recover from a miss by
# shifting aim instead of repeating the same spot, and penalize repeated
# misses / edge-center camping patterns that tend to waste shots.
MISS_CORRECTION_BONUS = 180.0
REPEATED_MISS_PENALTY = 60.0
MOVE_EPS = 0.03
SAME_EPS = 0.015
EDGE_BAND = 0.08
CENTER_BAND = 0.04
EDGE_SCATTER_PENALTY = 30.0
CENTER_CAMP_PENALTY = 30.0

# hit_delta shaping: per-frame counter of how long we've gone without a hit.
# The counter resets to 0 on any confirmed hit and increments by 1 on every
# frame that does not register a hit. It is exposed to the policy as
# hit_delta_norm and penalized in fitness (below) so the agent is nudged away
# from long dry streaks.
HIT_DELTA_NORM_FRAMES = 300.0
HIT_DELTA_PENALTY = 80.0

# Accuracy-shaped fitness bonus (rewards hit RATE, not just hit COUNT).
# Sim-validated (repo memory "Miss-correction objective probe", 2026-08-06):
# combining this with the miss-correction terms above (the "miss-
# correction+accuracy" arm) was the best-performing arm of that probe --
# preserved clear rate, improved mean_acc (0.089 -> 0.111), raised the local
# self-correction rate, and increased aim coordinate diversity, all without
# collapsing to edges/center. Also note: fitness shaping only changes what
# future ES generations are selected for -- it does not alter the behavior
# of an already-trained checkpoint (theta_*.npy) without re-running training.
ACCURACY_BONUS_WEIGHT = 1000.0

# Direct reward for clip_shift (aim variation between consecutive magazines/
# reloads) -- targets the "same aim arc every reload" symptom specifically.
# Sim-validated (repo memory "Clip-shift reward probe", 2026-08-06): a 5-seed
# x 30-gen A/B on top of the miss-correction+accuracy formula above found
# weight=60 actually made clip_shift WORSE (0.057 -> 0.049) while weight=150
# improved clear rate (60% -> 80%), mean_acc (0.069 -> 0.092), AND clip_shift
# (0.057 -> 0.076) together, with no metric regressing -- a clean win, not a
# trade-off. Earlier naive diversity-reward attempts (a reload-parity obs bit,
# a raw clip-novelty bonus) both hurt training in prior probes, so don't
# assume this generalizes to other diversity-reward designs without testing.
CLIP_SHIFT_BONUS = 150.0

# Direct reward for per-shot-slot diversity across clips (same shot index,
# different coordinates). This specifically targets the "same 6-shot arch
# every reload" behavior even when a policy still earns decent clip_shift by
# making only occasional whole-clip moves.
#
# The metric is normalized in env_timecrisis.py as:
#   mean_slot_std = mean_j sqrt(var(x_j across clips) + var(y_j across clips))
#   shot_slot_diversity = clip(mean_slot_std / SHOT_SLOT_DIVERSITY_SCALE, 0, 1)
# where j is shot index in the 6-shot clip.
SHOT_SLOT_DIVERSITY_BONUS = 120.0
SHOT_SLOT_DIVERSITY_SCALE = 0.08

# Flat bonus (NOT scaled by shots fired) awarded exactly once, on the tick
# the agent ducks back into cover with an empty clip (ammo_left == 0).
# Reinforces the "empty clip -> duck to reload" loop without rewarding shot
# count itself, so it can't be farmed by magdumping. 2026-08-05: set to 0
# during sweep-pattern debugging -- duck-on-empty is now hard-enforced in
# env_timecrisis.py, so this reward is no longer needed for that behavior and
# can bias policies toward repetitive magdump/reload loops.
RELOAD_BONUS = 0.0

# -----------------------------
# Policy dims
# obs = [timer_norm, life_norm, fired_norm, hit_norm, acc, last_hit, last_miss,
#        hit_delta_norm, peek_phase, ammo_norm, prev_aim_x_bias, prev_aim_y_bias,
#        cursor_x_norm, cursor_y_norm,
#        shot_phase_sin, shot_phase_cos]
# peek_phase in [-1, +1]: sign = current peek state, magnitude = ticks_held / PEEK_TRAVERSE_TICKS
# ammo_norm = ammo_left / AMMO_MAX_ROUNDS, in [0, 1]
# prev_aim_x_bias / prev_aim_y_bias in [-1, 1]: the aim_x_bias/aim_y_bias the
# policy itself output on the PREVIOUS tick, fed back in as the next tick's
# input. The net is otherwise purely feedforward/memoryless, so without this
# it has no way to know what it last chose -- this closes that loop, letting
# weights learn to shift aim across ticks instead of latching onto one spot.
# hit_delta_norm in [0, 1] (clipped): normalized per-frame streak length since
# the last confirmed hit. 0 means "just hit"; larger values mean the policy has
# gone longer without landing anything.
# cursor_x_norm / cursor_y_norm in [0, 1]: the actual current on-screen gun
# cursor read back from RAM. This is the first live screen-space signal in the
# policy input: even without enemy RAM yet, the policy can now correlate where
# rewarded hits happened with where the reticle actually was.
# shot_phase_sin / shot_phase_cos: (sin, cos) of the shot-in-clip phase angle,
# angle = 2*pi * (AMMO_MAX_ROUNDS - ammo_left) / AMMO_MAX_ROUNDS. Non-smooth
# per-shot signal (adjacent shots land at distinct 2-D positions on the unit
# circle, not on a monotonic ramp like ammo_norm) intended to break the "same
# 6-shot arc every clip" symptom -- the memoryless MLP fed only smooth
# monotonic inputs (ammo_norm ramp, prev_aim drift) naturally emits a smooth
# deterministic arc; adding a non-smooth per-shot cue lets ES route each shot
# through distinct hidden-layer paths without disentangling it from the ramp.
# Sim-validated (repo memory "Shot-phase sin/cos + zero-init port", 2026-08-06):
# a 5-seed x 30-gen A/B on TimedSpotBaselineAccuracyEnv showed this loosens the
# per-shot arc std on BOTH x (+66%) and y (+62%) axes vs. baseline OBS_DIM=13,
# and roughly triples pooled aim-y range -- the direct symptom fix the user
# asked for. Caveat: it also cost ~22 percentage points of clear rate in the
# sim (52% -> 30% at 30 gens) because the extra 2 dims add 2*HIDDEN=128 more
# parameters for ES to search over. The es_train.py warm-start zero-initializes
# the input-column weights on these 2 dims so at gen 0 behavior matches the
# pre-port baseline exactly and ES has to actively learn to USE these dims.
# act = [shoot_logit, cover_logit, aim_x_bias, aim_y_bias]
# Both aim axes are policy-controlled: bias in [-1, 1] -> screen position in [0, 1].
# -----------------------------
OBS_DIM = 16
HIDDEN  = 64
ACT_DIM = 4

# -----------------------------
# Logging / feedback
# -----------------------------
VERBOSE_EPISODES = False
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
