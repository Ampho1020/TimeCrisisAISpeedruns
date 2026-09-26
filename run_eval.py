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
                          Default 100 (human speed): eval plays at normal speed
                          with every frame drawn, unlike training's 3200%
                          (EMULATOR_SPEED_PERCENT in the Lua bridge). This is the
                          emulator throttle, NOT the Python-side decision
                          FRAME_SKIP -- that cadence is part of the trained
                          schedule and stays put.
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


def _print_shot_diag(recs: list, on_target_radius: float = 0.08) -> None:
    """Break down every fired shot into no-detection / off-target / on-target
    buckets and report the hit rate of each, so the source of wasted bullets
    is unambiguous (blind schedule pulses vs aim/detector precision misses)."""
    fired = sum(r["fired"] for r in recs)
    hits = sum(r["hit"] for r in recs)
    if fired == 0:
        print("shot-diag  : no shots fired")
        return

    def bucket(pred):
        f = sum(r["fired"] for r in recs if pred(r))
        h = sum(r["hit"] for r in recs if pred(r))
        return f, h

    blind_f, blind_h = bucket(lambda r: not r["had_detection"])
    off_f, off_h = bucket(
        lambda r: r["had_detection"] and r["nearest_dist"] >= on_target_radius
    )
    on_f, on_h = bucket(
        lambda r: r["had_detection"]
        and 0.0 <= r["nearest_dist"] < on_target_radius
    )
    det_recs = [r for r in recs if r["had_detection"]]
    mean_dist = (
        sum(r["nearest_dist"] for r in det_recs) / len(det_recs)
        if det_recs else -1.0
    )

    def pct(n):
        return 100.0 * n / fired

    def rate(h, f):
        return f"{100.0 * h / f:.0f}%" if f else "--"

    print("---- shot diagnostics ----")
    print(f"fired={fired}  hits={hits}  accuracy={100.0*hits/fired:.1f}%")
    print(f"  blind (no detection)   : {blind_f:3d} ({pct(blind_f):4.1f}%)  "
          f"hit {rate(blind_h, blind_f)}")
    print(f"  off-target (aim>{on_target_radius:.2f})   : {off_f:3d} ({pct(off_f):4.1f}%)  "
          f"hit {rate(off_h, off_f)}")
    print(f"  on-target (aim<{on_target_radius:.2f})    : {on_f:3d} ({pct(on_f):4.1f}%)  "
          f"hit {rate(on_h, on_f)}")
    print(f"  mean aim->nearest box  : {mean_dist:.3f}  (detected-shot frames)")


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
        "--shot-diag",
        action="store_true",
        help=(
            "Record every fired shot and print a breakdown of WHERE bullets "
            "are wasted: no-detection (blind), off-target, on-target misses."
        ),
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=100,
        metavar="PCT",
        help=(
            "Emulator throttle for the eval run, percent of real time "
            "(default 100 = human speed). Pass 3200 to match training speed."
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
    if args.shot_diag:
        env.shot_diag = []
    env.connect()
    try:
        # Override the launch-time training throttle (EMULATOR_SPEED_PERCENT
        # =3200 in bizhawk_bridge.lua) and disable BizHawk's display frameskip
        # so eval plays back like a human sees it. Does NOT touch the
        # Python-side decision FRAME_SKIP (baked into the trained schedule).
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
        print(f"post-kill shots suppressed : {info.get('suppressed_shot_pulses', 0)}")
        print(f"no_shot exposed : {info['no_shot_exposed_ticks']} ticks, "
              f"hesitated cover : {info['hesitated_cover_ticks']} ticks")
        if args.dump_frames:
            print(f"frames     : {env._dump_frame_counter} saved to {args.dump_frames}")
        if args.shot_diag and env.shot_diag is not None:
            _print_shot_diag(env.shot_diag)
    finally:
        env.close()


if __name__ == "__main__":
    main()
