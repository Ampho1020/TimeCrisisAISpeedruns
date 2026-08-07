"""CSV logger -- flushes every generation so plots can read it live."""

import csv
import os


class TrainingLogger:
    FIELDS = [
        "gen", "best", "mean", "std", "spread",
        "clear_rate", "best_time", "best_damage", "best_acc", "mean_acc",
        "mean_peek_flips", "mean_peek_hold", "mean_cover_time",
        "mean_aim_x_std", "mean_aim_y_std", "mean_aim_span_x", "mean_aim_span_y",
        "mean_aim_dx", "mean_hit_delta",
        "mean_shot_left_frac", "mean_shot_mid_frac", "mean_shot_right_frac",
        "sigma_used",
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
