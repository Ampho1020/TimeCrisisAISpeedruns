"""Evolution Strategies training loop."""

import numpy as np

from config import (
    ALPHA, CHECKPOINT_EVERY, GENERATIONS, HUD_ENABLED, LOG_CSV,
    POP_SIZE, SEED, SIGMA, VERBOSE_EPISODES,
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

    pool = WorkerPool()
    pool.start()
    logger = TrainingLogger(LOG_CSV)
    try:
        for gen in range(GENERATIONS):
            # --- mirrored sampling: test both +eps and -eps ---
            # Halves estimator variance for free.
            half = POP_SIZE // 2
            eps_half = rng.normal(0.0, 1.0, size=(half, PARAM_COUNT))
            eps = np.concatenate([eps_half, -eps_half], axis=0)
            candidates = theta[None, :] + SIGMA * eps

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
            gradient = (eps.T @ shaped) / (POP_SIZE * SIGMA)
            theta = theta + ALPHA * gradient

            # --- diagnostics ---
            best_i = int(np.argmax(fitnesses))
            best = infos[best_i]
            std = float(fitnesses.std())
            spread = float(fitnesses.max() - fitnesses.min())
            clear_rate = float(np.mean([1.0 if x["cleared"] else 0.0 for x in infos]))
            timeout_rate = float(np.mean([1.0 if x["timed_out"] else 0.0 for x in infos]))
            mean_acc = float(np.mean([x["accuracy"] for x in infos]))
            mean_flips      = float(np.mean([x["cover_flips"]      for x in infos]))
            mean_hold       = float(np.mean([x["cover_hold_score"] for x in infos]))
            mean_cover_time = float(np.mean([x["cover_time"]       for x in infos]))

            print(
                f"\n=== gen {gen:03d} | best {fitnesses[best_i]:8.2f} "
                f"| mean {fitnesses.mean():8.2f} | std {std:7.2f} | spread {spread:8.2f} "
                f"| clear {clear_rate:5.1%} | timeout {timeout_rate:5.1%} | t {best['elapsed']:6.1f} "
                f"| dmg {best['damage']:4.0f} | acc {best['accuracy']:5.1%} "
                f"| ctime {mean_cover_time:5.1f} | flips {mean_flips:5.1f} | hold {mean_hold:5.1f} ===\n",
                flush=True,
            )

            if std < 1e-3:
                print("  !! WARNING: fitness spread ~0 -- perturbations do nothing. RAISE SIGMA.", flush=True)
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
                "mean_cover_flips": mean_flips,
                "mean_cover_hold": mean_hold,
                "mean_cover_time": mean_cover_time,
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
