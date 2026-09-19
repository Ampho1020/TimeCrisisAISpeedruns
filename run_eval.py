"""Load a checkpoint and watch one episode.

Usage:
    python run_eval.py [checkpoint.npy] [--dump-frames <dir>] [--tick-vision]
                       [--speed <pct>]

Flags:
    --dump-frames <dir>   Save one PNG per decision tick during the episode
                          under <dir>. Used to build a labelling corpus for
                          the offline YOLO fine-tune workflow documented in
                          detector.py's footer. No effect on fitness.
    --tick-vision         Opt out of per-frame vision (revert to the
                          training-matching cadence: vision refreshed once
                          every VISION_CAPTURE_EVERY_N_TICKS ticks instead of
                          every raw emulator frame). By default eval now
                          refreshes vision + aim on every single frame, like
                          a human player watching the screen continuously.
    --speed <pct>         Emulator throttle for the run, percent of real time.
                          Default 100 (human speed): eval plays back at normal
                          speed with every frame drawn, unlike training which
                          runs at 3200% (EMULATOR_SPEED_PERCENT in the Lua
                          bridge). NOTE: this is the emulator throttle, NOT the
                          Python-side decision FRAME_SKIP -- that cadence is
                          part of the trained schedule and stays at FRAME_SKIP.
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
    parser.add_argument(
        "--tick-vision",
        action="store_true",
        help=(
            "Revert to training-cadence vision (refresh every "
            "VISION_CAPTURE_EVERY_N_TICKS ticks) instead of every frame."
        ),
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=100,
        metavar="PCT",
        help=(
            "Emulator throttle for the eval run, as a percent of real time. "
            "Default 100 (human speed) so you watch it play like a person "
            "would; pass e.g. 400 for a faster-but-still-watchable run, or "
            "3200 to match training speed."
        ),
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

    env = TimeCrisisEnv(per_frame_vision=not args.tick_vision)
    if args.dump_frames:
        env.dump_frames_dir = args.dump_frames
    env.connect()
    try:
        # Override the launch-time training throttle (EMULATOR_SPEED_PERCENT
        # =3200 in bizhawk_bridge.lua) so eval plays back like a human would
        # see it, and disable BizHawk's display frameskip so every frame is
        # drawn (at high speed the emulator auto-drops rendered frames, which
        # looks like stuttering). This does NOT touch the Python-side decision
        # FRAME_SKIP -- that cadence is baked into the trained tick schedule.
        env.client.set_speed(args.speed)
        env.client.set_frameskip(0)
        print(f"[eval] emulator speed set to {args.speed}% with display "
              f"frameskip 0 (real-time playback)", flush=True)
        fitness, info = env.episode_fitness(theta)
        print(f"checkpoint : {args.checkpoint}")
        print(f"fitness    : {fitness:.2f}")
        print(f"cleared    : {info['cleared']}")
        print(f"screens    : {info.get('screens_cleared', 0)}")
        print(f"elapsed    : {info['elapsed']:.1f}")
        print(f"damage     : {info['damage']:.0f}")
        print(f"accuracy   : {info['accuracy']:.1%} "
              f"({info['shots_hit']}/{info['shots_fired']})")
        print(f"dry_fire   : {info['dry_fire_ticks']} ticks")
        print(f"reload ok  : {info['reload_correct_count']}")
        print(f"no_shot exposed : {info['no_shot_exposed_ticks']} ticks, "
              f"hesitated cover : {info['hesitated_cover_ticks']} ticks")
        if args.dump_frames:
            print(f"frames     : {env._dump_frame_counter} saved to {args.dump_frames}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
