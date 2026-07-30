"""Load a checkpoint and watch one episode."""

import sys

import numpy as np

from env_timecrisis import TimeCrisisEnv


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "theta_final.npy"
    theta = np.load(ckpt)

    env = TimeCrisisEnv()
    env.connect()
    try:
        fitness, info = env.episode_fitness(theta)
        print(f"checkpoint : {ckpt}")
        print(f"fitness    : {fitness:.2f}")
        print(f"cleared    : {info['cleared']}")
        print(f"elapsed    : {info['elapsed']:.1f}")
        print(f"damage     : {info['damage']:.0f}")
        print(f"accuracy   : {info['accuracy']:.1%} "
              f"({info['shots_hit']}/{info['shots_fired']})")
    finally:
        env.close()


if __name__ == "__main__":
    main()
