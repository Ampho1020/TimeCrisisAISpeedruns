"""Evolution Strategies training loop."""

import numpy as np

from config import (
    ACT_DIM, ALPHA, CHECKPOINT_EVERY, GENERATIONS, HIDDEN, HUD_ENABLED,
    LOG_CSV, OBS_DIM, POP_SIZE, SEED, SIGMA, STAGNATION_PATIENCE,
    STAGNATION_SIGMA_MULT, STD_STAGNATION_THRESHOLD, VERBOSE_EPISODES,
)
from logger import TrainingLogger
from policy import PARAM_COUNT
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


def train():
    if POP_SIZE % 2 != 0:
        raise ValueError("POP_SIZE must be even for mirrored sampling.")

    rng = np.random.default_rng(SEED)
    # Small init -- large weights saturate tanh and kill the signal.
    theta = rng.normal(0.0, 0.1, size=(PARAM_COUNT,)).astype(np.float64)
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
            eps_half = rng.normal(0.0, 1.0, size=(half, PARAM_COUNT))
            eps = np.concatenate([eps_half, -eps_half], axis=0)
            candidates = theta[None, :] + sigma_this_gen * eps

            # Evaluate the whole population across all workers in parallel.
            results = pool.evaluate(candidates)
            fitnesses = [r[0] for r in results]
            infos = [r[1] for r in results]

            if VERBOSE_EPISODES:
                for i, (fit, info) in enumerate(zip(fitnesses, infos)):
                    print(
                        f"  gen {gen:03d} ep {i + 1:02d}/{POP_SIZE} | "
                        f"fit {fit:8.2f} | t {info['elapsed']:6.1f} | "
                        f"dmg {info['damage']:4.0f} | acc {info['accuracy']:5.1%} | "
                        f"{'CLEAR' if info['cleared'] else '.....'}",
                        flush=True,
                    )

            fitnesses = np.asarray(fitnesses, dtype=np.float64)

            # --- the ES update ---
            shaped = rank_transform(fitnesses)
            gradient = (eps.T @ shaped) / (POP_SIZE * sigma_this_gen)
            theta = theta + ALPHA * gradient

            # --- diagnostics ---
            best_i = int(np.argmax(fitnesses))
            best = infos[best_i]
            std = float(fitnesses.std())
            spread = float(fitnesses.max() - fitnesses.min())
            clear_rate = float(np.mean([1.0 if x["cleared"] else 0.0 for x in infos]))
            timeout_rate = float(np.mean([1.0 if x["timed_out"] else 0.0 for x in infos]))
            mean_acc = float(np.mean([x["accuracy"] for x in infos]))
            mean_flips      = float(np.mean([x["peek_flips"]      for x in infos]))
            mean_hold       = float(np.mean([x["peek_hold_score"] for x in infos]))
            mean_cover_time = float(np.mean([x["cover_time"]       for x in infos]))

            print(
                f"\n=== gen {gen:03d} | best {fitnesses[best_i]:8.2f} "
                f"| mean {fitnesses.mean():8.2f} | std {std:7.2f} | spread {spread:8.2f} "
                f"| clear {clear_rate:5.1%} | timeout {timeout_rate:5.1%} | t {best['elapsed']:6.1f} "
                f"| dmg {best['damage']:4.0f} | acc {best['accuracy']:5.1%} "
                f"| ctime {mean_cover_time:5.1f} | flips {mean_flips:5.1f} | hold {mean_hold:5.1f} ===\n",
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
                "sigma_used": sigma_this_gen,
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
