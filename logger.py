"""CSV logger -- flushes every generation so plots can read it live."""

import csv
import os


class TrainingLogger:
    FIELDS = [
        "run_id",
        "gen", "best", "mean", "std", "spread",
        "clear_rate", "best_time", "best_damage", "best_acc", "mean_acc",
        "mean_peek_flips", "mean_peek_hold", "mean_cover_time",
        "mean_aim_x_std", "mean_aim_y_std", "mean_aim_span_x", "mean_aim_span_y",
        "mean_aim_dx", "mean_hit_delta",
        "mean_shot_left_frac", "mean_shot_mid_frac", "mean_shot_right_frac",
        "sigma_used",
        "theta_fitness", "theta_clear", "theta_time", "theta_damage", "theta_acc",
        # Multi-screen tracking (added 2026-08-10 alongside MULTI_CLEAR_BONUS
        # in config.py). mean_screens_cleared / max_screens_cleared aggregate
        # across the population per generation; theta_screens_cleared is the
        # current mean-theta's single-episode count.
        "mean_screens_cleared", "max_screens_cleared", "theta_screens_cleared",
        # Vision-gain tracking (added 2026-08-15 alongside VISION_GAIN_WARMSTART
        # in config.py). mean_vision_gain is the population's average
        # tanh(vision_gain_logit) across all MAX_TICKS rows (0.0 for
        # non-vision_schedule POLICY_MODE); theta_vision_gain is the same
        # computed on the updated center theta. Both should stay non-trivial
        # (not collapse toward 0) if ES is keeping vision meaningfully in the
        # aim blend rather than learning to ignore it.
        "mean_vision_gain", "theta_vision_gain",
    ]

    def __init__(self, path: str):
        self.path = path
        is_new = not os.path.exists(path)
        if not is_new:
            # If FIELDS has grown since this file's header was written (e.g.
            # sigma_used added 2026-08-04), appending rows with the new
            # column count under the old header desyncs every row after it --
            # DictReader-based tools (plot_progress.py) silently drop the
            # extra trailing values instead of erroring. Detect that and
            # rotate the stale file out of the way rather than corrupt it
            # further; a fresh file gets the current header.
            with open(path, newline="") as fh:
                existing_header = fh.readline().strip().split(",")
            if existing_header != self.FIELDS:
                backup_path = path + ".pre_" + "_".join(self.FIELDS[-1:]) + ".bak"
                if not os.path.exists(backup_path):
                    os.rename(path, backup_path)
                    print(
                        f"[logger] {path} had an outdated header -- moved to "
                        f"{backup_path}, starting a fresh log.",
                    )
                    is_new = True
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=self.FIELDS)
        if is_new:
            self.w.writeheader()
            self.f.flush()

    def log(self, row: dict):
        self.w.writerow({k: row.get(k, "") for k in self.FIELDS})
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass
