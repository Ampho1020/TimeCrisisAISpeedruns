"""CSV logger -- flushes every generation so plots can read it live."""

import csv
import os


class TrainingLogger:
    # One row per generation. Every column es_train.py can compute is logged
    # here so plot_progress.py (and manual CSV inspection) has the full picture
    # of a run -- population means, the best candidate, and the actual center
    # theta -- without silently dropping anything. Columns are grouped by
    # concern; plot_progress.py tolerates missing/empty cells per-column, so
    # older runs and non-vision_schedule modes still plot cleanly.
    FIELDS = [
        # --- identity / ES state ---
        "run_id", "gen", "sigma_used",
        # --- fitness (population spread + the actual center theta) ---
        "best", "mean", "std", "spread", "theta_fitness",
        # --- outcome rates across the population (fractions in [0, 1]) ---
        "clear_rate", "timeout_rate", "dead_rate",
        # --- center-theta outcome this generation ---
        "theta_clear", "theta_time", "theta_damage", "theta_acc",
        "theta_screens_cleared",
        # --- best candidate this generation ---
        "best_time", "best_damage", "best_acc",
        # --- population-mean outcomes ---
        "mean_time", "mean_damage", "mean_acc",
        "mean_shots_fired", "mean_shots_hit",
        # Multi-screen tracking (MULTI_CLEAR_BONUS in config.py). mean/max are
        # population aggregates; theta_screens_cleared (above) is the center.
        "mean_screens_cleared", "max_screens_cleared",
        # --- cover / peek behaviour ---
        "mean_peek_flips", "mean_peek_hold", "mean_cover_time",
        # --- trigger / cover discipline (per-episode tick counts, pop mean) ---
        "mean_dry_fire", "mean_no_shot_exposed", "mean_hesitated_cover",
        "mean_reload_correct", "mean_continue_ticks",
        # --- aim behaviour / lane usage ---
        "mean_aim_x_std", "mean_aim_y_std", "mean_aim_span_x", "mean_aim_span_y",
        "mean_aim_dx", "mean_aim_dy",
        "mean_shot_left_frac", "mean_shot_mid_frac", "mean_shot_right_frac",
        "mean_hit_rate_left", "mean_hit_rate_mid", "mean_hit_rate_right",
        # --- reaction / hit timing ---
        # mean_hit_delta: mean frames between the reticle reaching a target and
        # the hit registering. mean_reaction_latency: mean ticks between a
        # target becoming visible and the first shot at it (REACTION_LATENCY_
        # PENALTY in config.py). Both should trend DOWN.
        "mean_hit_delta", "mean_reaction_latency",
        # --- shared gain scalars (vision_schedule only; 0.0 otherwise) ---
        # Each pair is population-mean tanh(gain_logit) and the same on the
        # updated center theta. Watch for any collapsing toward 0 (ES learning
        # to ignore that signal) vs. staying meaningfully engaged.
        "mean_vision_gain", "theta_vision_gain",
        "mean_shoot_gain", "theta_shoot_gain",
        "mean_drift_gain", "theta_drift_gain",
        "mean_peek_gain", "theta_peek_gain",
        "mean_ammo_gain", "theta_ammo_gain",
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
