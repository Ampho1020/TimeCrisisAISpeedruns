"""Load a checkpoint and watch one episode.

Usage:
    python run_eval.py [checkpoint.npy] [--dump-frames <dir>]

Flags:
    --dump-frames <dir>   Save one PNG per decision tick during the episode
                          under <dir>. Used to build a labelling corpus for
                          the offline YOLO fine-tune workflow documented in
                          detector.py's footer. No effect on fitness.
"""

import argparse
import sys

import numpy as np

from config import POLICY_MODE
from env_timecrisis import TimeCrisisEnv
from policy import PARAM_COUNT, SCHEDULE_PARAM_COUNT, VISION_SCHEDULE_PARAM_COUNT


def _expected_theta_size() -> int:
    if POLICY_MODE == "schedule":
        return SCHEDULE_PARAM_COUNT
    if POLICY_MODE == "vision_schedule":
        return VISION_SCHEDULE_PARAM_COUNT
    return PARAM_COUNT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="theta_final.npy",
        help="Path to the theta .npy checkpoint to load (default: theta_final.npy)",
    )
    parser.add_argument(
        "--dump-frames",
        metavar="DIR",
        default=None,
        help="Save one PNG per decision tick under DIR (for offline labelling).",
    )
    args = parser.parse_args()
    theta = np.asarray(np.load(args.checkpoint), dtype=np.float64).reshape(-1)
    expected = _expected_theta_size()
    if theta.size != expected:
        raise ValueError(
            f"Checkpoint parameter count mismatch for POLICY_MODE='{POLICY_MODE}': "
            f"loaded {theta.size}, expected {expected}. "
            "This is expected after class-schema changes; regenerate checkpoints "
            "with the current config."
        )

    env = TimeCrisisEnv()
    if args.dump_frames:
        env.dump_frames_dir = args.dump_frames
    env.connect()
    try:
        fitness, info = env.episode_fitness(theta)
        print(f"checkpoint : {args.checkpoint}")
        print(f"fitness    : {fitness:.2f}")
        print(f"cleared    : {info['cleared']}")
        print(f"screens    : {info.get('screens_cleared', 0)}")
        print(f"elapsed    : {info['elapsed']:.1f}")
        print(f"damage     : {info['damage']:.0f}")
        print(f"accuracy   : {info['accuracy']:.1%} "
              f"({info['shots_hit']}/{info['shots_fired']})")
        if args.dump_frames:
            print(f"frames     : {env._dump_frame_counter} saved to {args.dump_frames}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
