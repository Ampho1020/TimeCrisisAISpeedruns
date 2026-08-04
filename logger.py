"""CSV logger -- flushes every generation so plots can read it live."""

import csv
import os


class TrainingLogger:
    FIELDS = [
        "gen", "best", "mean", "std", "spread",
        "clear_rate", "best_time", "best_damage", "best_acc", "mean_acc",
        "mean_peek_flips", "mean_peek_hold", "mean_cover_time", "sigma_used",
    ]

    def __init__(self, path: str):
        self.path = path
        is_new = not os.path.exists(path)
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
