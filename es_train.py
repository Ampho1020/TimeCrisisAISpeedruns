"""Evolution Strategies training loop."""

import time

import numpy as np

from config import (
    ACT_DIM, ALPHA, CHECKPOINT_EVERY, EPISODES_PER_CANDIDATE, GENERATIONS,
    HIDDEN, HUD_ENABLED, LOG_CSV, MAX_TICKS, OBS_DIM, POLICY_MODE, POP_SIZE,
    SEED, SIGMA, SHOT_PHASE_WARMSTART_ROW_STD,
    STAGNATION_PATIENCE, STAGNATION_SIGMA_MULT, STD_STAGNATION_THRESHOLD,
    VERBOSE_EPISODES,
)
from logger import TrainingLogger
from policy import PARAM_COUNT, SCHEDULE_PARAM_COUNT, VISION_SCHEDULE_PARAM_COUNT, VISION_SCHEDULE_ROW_DIM
from worker_pool import WorkerPool


def rank_transform(fitnesses: np.ndarray) -> np.ndarray:
    """
    Map raw fitness to evenly spaced ranks in [-0.5, 0.5].
    Makes the update depend only on ORDERING, so one lucky outlier
    run can't dominate the gradient estimate. Do not skip this.
    """
    ranks = np.empty_like(fitnesses, dtype=np.float64)
    ranks[np.argsort(fitnesses)] = np.arange(len(fitnesses))
    return ranks / (len(fitnesses) - 1) - 0.5


def _average_episode_infos(infos: list) -> dict:
    """Average a list of same-keyed per-episode info dicts into one dict
    (used to collapse EPISODES_PER_CANDIDATE raw episodes down to one
    averaged info per candidate before ranking). Booleans (e.g. 'cleared',
    'timed_out', 'dead') become fractions in [0, 1] rather than a single
    True/False -- with EPISODES_PER_CANDIDATE=1 this reduces to the exact
    original 0.0/1.0 value.
    """
    keys = infos[0].keys()
    return {k: float(np.mean([float(info[k]) for info in infos])) for k in keys}


def train():
    if POP_SIZE % 2 != 0:
        raise ValueError("POP_SIZE must be even for mirrored sampling.")

    rng = np.random.default_rng(SEED)
    if POLICY_MODE == "schedule":
        param_count = SCHEDULE_PARAM_COUNT
    elif POLICY_MODE == "vision_schedule":
        param_count = VISION_SCHEDULE_PARAM_COUNT
    else:
        param_count = PARAM_COUNT
    # Small init -- large weights saturate tanh and kill the signal.
    theta = rng.normal(0.0, 0.1, size=(param_count,)).astype(np.float64)

    if POLICY_MODE == "schedule":
        # Open-loop: theta is a flat (MAX_TICKS, 4) per-tick action table
        # (shoot_logit, peek_logit, aim_x_bias, aim_y_bias). Mild peek-
        # forward bias on EVERY tick's row -- same anti-"never expose"
        # collapse rationale as the closed-loop peek-logit warm-start below,
        # but applied per-row since there's no shared bias term here. See
        # tests/test_simulation.py's `_theta_schedule_warm_start` / repo
        # memory "Open-loop schedule search" for the sim-validated version
        # of this warm-start.
        theta = theta.reshape(MAX_TICKS, 4)
        theta[:, 1] += 1.0
        theta = theta.reshape(-1)
    elif POLICY_MODE == "vision_schedule":
        # Vision-conditioned schedule: per-tick (shoot, peek, base_ax,
        # base_ay, vision_gain) rows followed by a global class-priority
        # vector. To keep the Phase 3 rollout risk-controlled we make gen-0
        # behaviour BYTE-IDENTICAL to plain schedule mode:
        #   * ``vision_gain`` column (col 4) zero-init -> tanh(0)=0 collapses
        #     the vision blend to the base aim regardless of what the
        #     detector returns.
        #   * class-priority tail zero-init -> softmax is uniform, no bias
        #     toward any particular EnemyClass.
        # ES must then actively LEARN a non-zero gain / class priority to
        # improve over the schedule baseline -- if it can't, we lose
        # nothing (see /memories/session/plan.md Phase 4 gate).
        per_tick = MAX_TICKS * VISION_SCHEDULE_ROW_DIM
        grid = theta[:per_tick].reshape(MAX_TICKS, VISION_SCHEDULE_ROW_DIM)
        grid[:, 1] += 1.0   # peek-forward bias (same as schedule mode)
        grid[:, 4] = 0.0    # vision_gain -> disabled at gen 0
        theta[:per_tick] = grid.reshape(-1)
        theta[per_tick:] = 0.0  # class-priority tail -> uniform softmax
    else:
        # Warm-start the shoot logit to +2 and the peek logit to +1 (asymmetric,
        # 2026-08-04). Without some positive bias, ~50% of random seeds produce a
        # theta whose peek output is negative for the fixed initial observation,
        # so every perturbed candidate stays in cover and ES is stuck at std = 0
        # on generation 0.
        #
        # An earlier version of this fix raised BOTH logits to +2 to guard against
        # that "always in cover" collapse. Live training logs then showed the
        # opposite failure mode instead: mean_peek_flips pinned at exactly 0.0 and
        # mean_cover_time at exactly 0.0 across the *entire* population for many
        # generations straight -- every candidate stayed exposed 100% of the time
        # and never ducked to reload (the "mag dump" behaviour). The cause is
        # that +2 sits far outside SIGMA's (0.1) reach: as long as the perturbed
        # bias stays positive (which it does for all but the most extreme
        # samples), peek is always True and behaviour never changes with it, so
        # there is *zero* local fitness gradient telling ES to move the bias down
        # -- it's a flat plateau, not just a slow climb. Learning to duck is then
        # entirely dependent on the (initially near-zero) hidden-layer weights on
        # the ammo_norm input overcoming that +2 bias, which is a much harder
        # thing for random perturbations to stumble into than simply crossing 0
        # from a smaller starting bias. Dropping the peek bias back to +1 keeps
        # the original anti-"stuck in cover" guarantee while leaving duck
        # behaviour reachable on a normal training timescale; the stagnation-kick
        # in the loop below (not a large static bias) is what should catch a
        # renewed collapse toward "always in cover" if SIGMA=0.1 isn't already
        # enough on its own.
        _b2_start = OBS_DIM * HIDDEN + HIDDEN + HIDDEN * ACT_DIM
        theta[_b2_start + 0] += 2.0  # shoot logit
        theta[_b2_start + 1] += 1.0  # peek  logit

        # Shot-phase port (OBS_DIM 13 -> 15): initialize ONLY the newly-added
        # input rows (shot_phase_sin/cos) in w1 with a small configurable std.
        # - std=0.0 reproduces strict zero-init (gen0 equals pre-port behavior).
        # - std>0.0 injects a small early signal so the policy can start using
        #   shot-phase features sooner, which helps break repeated clip arcs.
        shot_phase_extra_dims = 2
        base_obs_dim = OBS_DIM - shot_phase_extra_dims
        if base_obs_dim > 0:
            _w1_end = OBS_DIM * HIDDEN
            w1 = theta[:_w1_end].reshape(OBS_DIM, HIDDEN)
            if SHOT_PHASE_WARMSTART_ROW_STD > 0.0:
                w1[base_obs_dim:OBS_DIM, :] = rng.normal(
                    0.0,
                    SHOT_PHASE_WARMSTART_ROW_STD,
                    size=(shot_phase_extra_dims, HIDDEN),
                )
            else:
                w1[base_obs_dim:OBS_DIM, :] = 0.0

    pool = WorkerPool()
    pool.start()
    logger = TrainingLogger(LOG_CSV)
    # Consecutive generations with fitness std below STD_STAGNATION_THRESHOLD.
    # When this reaches STAGNATION_PATIENCE, every mirrored candidate is
    # landing on the same behavior (e.g. the "never expose" collapse -- see
    # SIGMA note in config.py) and the ES gradient carries no real signal.
    # Temporarily sampling with a larger SIGMA gives perturbations a better
    # chance of flipping a candidate's behavior again.
    stagnant_gens = 0
    try:
        for gen in range(GENERATIONS):
            kicking = stagnant_gens >= STAGNATION_PATIENCE
            sigma_this_gen = SIGMA * STAGNATION_SIGMA_MULT if kicking else SIGMA
            if kicking:
                print(
                    f"  !! stagnation kick: std < {STD_STAGNATION_THRESHOLD} for "
                    f"{stagnant_gens} gens -- sampling with SIGMA x"
                    f"{STAGNATION_SIGMA_MULT} this generation",
                    flush=True,
                )

            # --- mirrored sampling: test both +eps and -eps ---
            # Halves estimator variance for free.
            half = POP_SIZE // 2
            eps_half = rng.normal(0.0, 1.0, size=(half, param_count))
            eps = np.concatenate([eps_half, -eps_half], axis=0)
            candidates = theta[None, :] + sigma_this_gen * eps

            # Evaluate the whole population across all workers in parallel.
            # Each candidate gets EPISODES_PER_CANDIDATE episodes (averaged
            # before ranking) to reduce single-episode fitness-ranking noise
            # -- see EPISODES_PER_CANDIDATE's comment in config.py and repo
            # memory's "Multi-episode fitness averaging probe" for why.
            expanded_candidates = np.repeat(candidates, EPISODES_PER_CANDIDATE, axis=0)
            total_evals = len(expanded_candidates)
            report_every = max(1, total_evals // 10)
            gen_eval_start = time.monotonic()

            def _on_eval_progress(done: int, total: int):
                if done % report_every == 0 or done == total:
                    elapsed = time.monotonic() - gen_eval_start
                    print(
                        f"  [gen {gen:03d}] eval {done:3d}/{total:3d} "
                        f"({100.0 * done / total:5.1f}%) in {elapsed:6.1f}s",
                        flush=True,
                    )

            raw_results = pool.evaluate(expanded_candidates, progress_cb=_on_eval_progress)

            fitnesses = np.empty(POP_SIZE, dtype=np.float64)
            infos = []
            for i in range(POP_SIZE):
                chunk = raw_results[i * EPISODES_PER_CANDIDATE:(i + 1) * EPISODES_PER_CANDIDATE]
                fitnesses[i] = float(np.mean([r[0] for r in chunk]))
                infos.append(_average_episode_infos([r[1] for r in chunk]))

            if VERBOSE_EPISODES:
                for i, (fit, info) in enumerate(zip(fitnesses, infos)):
                    print(
                        f"  gen {gen:03d} ep {i + 1:02d}/{POP_SIZE} | "
                        f"fit {fit:8.2f} | t {info['elapsed']:6.1f} | "
                        f"dmg {info['damage']:4.0f} | acc {info['accuracy']:5.1%} | "
                        f"{'CLEAR' if info['cleared'] >= 1.0 else '.....'}",
                        flush=True,
                    )

            # --- the ES update ---
            shaped = rank_transform(fitnesses)
            gradient = (eps.T @ shaped) / (POP_SIZE * sigma_this_gen)
            theta = theta + ALPHA * gradient

            # --- evaluate the actual (unperturbed) center theta itself ---
            # The population stats below (best/mean/clear_rate/etc.) describe
            # the 30 PERTURBED candidates (theta +/- sigma*eps) used only to
            # estimate the gradient -- they are neighborhood diagnostics, not
            # a measurement of the policy this generation actually produces.
            # For schedule mode this gap can look alarming (e.g. a 3600-dim
            # raw action table is very perturbation-sensitive, so population
            # clear_rate can sit at ~30-40% while the center theta itself
            # clears every time -- confirmed empirically 2026-08-10: 16/16
            # real BizHawk replays of one center theta were bit-identical,
            # std=0.0, since schedule mode is a fixed open-loop action table
            # replayed deterministically from a fixed savestate). One extra
            # episode/gen is negligible next to the population's evals.
            theta_fitness, theta_info = pool.evaluate(theta[None, :])[0]

            # --- diagnostics ---
            best_i = int(np.argmax(fitnesses))
            best = infos[best_i]
            std = float(fitnesses.std())
            spread = float(fitnesses.max() - fitnesses.min())
            # x["cleared"]/x["timed_out"] are already fractions in [0, 1]
            # per candidate (mean over its EPISODES_PER_CANDIDATE episodes),
            # exactly 0.0/1.0 when EPISODES_PER_CANDIDATE == 1.
            clear_rate = float(np.mean([x["cleared"] for x in infos]))
            timeout_rate = float(np.mean([x["timed_out"] for x in infos]))
            mean_acc = float(np.mean([x["accuracy"] for x in infos]))
            mean_flips      = float(np.mean([x["peek_flips"]      for x in infos]))
            mean_hold       = float(np.mean([x["peek_hold_score"] for x in infos]))
            mean_cover_time = float(np.mean([x["cover_time"]       for x in infos]))
            mean_aim_x_std = float(np.mean([x["aim_x_std"] for x in infos]))
            mean_aim_y_std = float(np.mean([x["aim_y_std"] for x in infos]))
            mean_aim_span_x = float(np.mean([x["aim_span_x"] for x in infos]))
            mean_aim_span_y = float(np.mean([x["aim_span_y"] for x in infos]))
            mean_aim_dx = float(np.mean([x["mean_abs_aim_dx"] for x in infos]))
            mean_shot_left_frac = float(np.mean([x["shot_left_frac"] for x in infos]))
            mean_shot_mid_frac = float(np.mean([x["shot_mid_frac"] for x in infos]))
            mean_shot_right_frac = float(np.mean([x["shot_right_frac"] for x in infos]))
            mean_hit_delta = float(np.mean([x["mean_hit_delta"] for x in infos]))

            print(
                f"\n=== gen {gen:03d} | best {fitnesses[best_i]:8.2f} "
                f"| mean {fitnesses.mean():8.2f} | std {std:7.2f} | spread {spread:8.2f} "
                f"| clear {clear_rate:5.1%} | timeout {timeout_rate:5.1%} | t {best['elapsed']:6.1f} "
                f"| dmg {best['damage']:4.0f} | acc {best['accuracy']:5.1%} "
                f"| ctime {mean_cover_time:5.1f} | flips {mean_flips:5.1f} | hold {mean_hold:5.1f} "
                f"| aimstd ({mean_aim_x_std:.3f},{mean_aim_y_std:.3f}) "
                f"| aimspan ({mean_aim_span_x:.3f},{mean_aim_span_y:.3f}) "
                f"| aimdx {mean_aim_dx:.3f} "
                f"| hitd {mean_hit_delta:.1f} "
                f"| lanes L/M/R {mean_shot_left_frac:.0%}/{mean_shot_mid_frac:.0%}/{mean_shot_right_frac:.0%} ===\n"
                f"    [center theta] fit {theta_fitness:8.2f} | clear {'YES' if theta_info['cleared'] >= 1.0 else 'no '} "
                f"| t {theta_info['elapsed']:6.1f} | dmg {theta_info['damage']:4.0f} | acc {theta_info['accuracy']:5.1%}\n",
                flush=True,
            )

            if std < STD_STAGNATION_THRESHOLD:
                stagnant_gens += 1
                print("  !! WARNING: fitness spread ~0 -- perturbations do nothing.", flush=True)
            else:
                stagnant_gens = 0
            if gen > 5 and clear_rate == 0.0:
                print("  !! note: no clears yet -- running on partial credit only.", flush=True)

            if HUD_ENABLED:
                for env in pool.envs:
                    try:
                        env.client.hud([f"gen {gen}", f"best {fitnesses.max():.0f}"])
                    except Exception:
                        pass

            logger.log({
                "gen": gen,
                "best": float(fitnesses[best_i]),
                "mean": float(fitnesses.mean()),
                "std": std,
                "spread": spread,
                "clear_rate": clear_rate,
                "best_time": best["elapsed"],
                "best_damage": best["damage"],
                "best_acc": best["accuracy"],
                "mean_acc": mean_acc,
                "mean_peek_flips": mean_flips,
                "mean_peek_hold": mean_hold,
                "mean_cover_time": mean_cover_time,
                "mean_aim_x_std": mean_aim_x_std,
                "mean_aim_y_std": mean_aim_y_std,
                "mean_aim_span_x": mean_aim_span_x,
                "mean_aim_span_y": mean_aim_span_y,
                "mean_aim_dx": mean_aim_dx,
                "mean_hit_delta": mean_hit_delta,
                "mean_shot_left_frac": mean_shot_left_frac,
                "mean_shot_mid_frac": mean_shot_mid_frac,
                "mean_shot_right_frac": mean_shot_right_frac,
                "sigma_used": sigma_this_gen,
                "theta_fitness": theta_fitness,
                "theta_clear": theta_info["cleared"],
                "theta_time": theta_info["elapsed"],
                "theta_damage": theta_info["damage"],
                "theta_acc": theta_info["accuracy"],
            })

            if gen % CHECKPOINT_EVERY == 0:
                np.save(f"theta_gen_{gen:03d}.npy", theta)

        np.save("theta_final.npy", theta)
        print("Training complete -> theta_final.npy")

    except KeyboardInterrupt:
        print("\nInterrupted -- saving theta_interrupt.npy")
        np.save("theta_interrupt.npy", theta)

    finally:
        logger.close()
        if HUD_ENABLED:
            for env in pool.envs:
                try:
                    env.client.hud_clear()
                except Exception:
                    pass
        pool.close()


if __name__ == "__main__":
    train()
