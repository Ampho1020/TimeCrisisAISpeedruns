"""Plot training_log.csv. Safe to run mid-training in a second terminal."""

import csv
import glob
import os
import sys

import matplotlib.pyplot as plt


def _latest_log_path(pattern: str = "training_log*.csv"):
    candidates = [p for p in glob.glob(pattern) if os.path.isfile(p)]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _resolve_paths(argv):
    """Resolve input/output paths.

    Modes:
      plot_progress.py
      plot_progress.py <csv_path>
      plot_progress.py <csv_path> <out_png>
      plot_progress.py --latest
      plot_progress.py --latest <out_png>
    """
    if len(argv) > 0 and argv[0] == "--latest":
        path = _latest_log_path("training_log*.csv")
        if path is None:
            raise FileNotFoundError("No training_log*.csv files found in the current directory.")
        out_path = argv[1] if len(argv) > 1 else "latest_plot.png"
        return path, out_path, True

    path = argv[0] if len(argv) > 0 else "training_log.csv"
    out_path = argv[1] if len(argv) > 1 else None
    return path, out_path, False


def main():
    try:
        path, out_path, used_latest = _resolve_paths(sys.argv[1:])
    except FileNotFoundError as e:
        print(str(e))
        return
    with open(path) as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        print("No data yet.")
        return

    gens = [int(r["gen"]) for r in rows]

    def col(k):
        return [float(r[k]) for r in rows if r.get(k, "") != ""]

    def col_gens(k):
        return [int(r["gen"]) for r in rows if r.get(k, "") != ""]

    def has_col(k):
        return any(r.get(k, "") != "" for r in rows)

    fig, ax = plt.subplots(4, 2, figsize=(12, 13))

    ax[0][0].plot(gens, col("best"), label="best")
    ax[0][0].plot(gens, col("mean"), label="mean")
    ax[0][0].set_title("fitness")
    ax[0][0].legend()

    ax[0][1].plot(gens, col("std"), color="tab:orange")
    ax[0][1].set_title("fitness std   (~0 => raise SIGMA)")

    ax[1][0].plot(gens, [100 * v for v in col("clear_rate")], color="tab:green")
    ax[1][0].set_title("clear rate %")
    ax[1][0].set_ylim(-5, 105)

    ax[1][1].plot(gens, col("best_time"), color="tab:red")
    ax[1][1].set_title("best clear time")

    # --- cover / peek behaviour ---
    ctime_g = col_gens("mean_cover_time")
    hold_g  = col_gens("mean_peek_hold")

    if ctime_g:
        ax[2][0].plot(ctime_g, col("mean_cover_time"), color="tab:purple")
        ax[2][0].axhline(y=900, color="grey", linestyle="--", linewidth=0.8,
                         label="always-cover baseline (900 ticks)")
        ax[2][0].set_title("mean ticks in cover   (↓ = less camping)")
        ax[2][0].legend(fontsize=8)
    else:
        ax[2][0].text(0.5, 0.5, "no cover data yet",
                      ha="center", va="center", transform=ax[2][0].transAxes)
        ax[2][0].set_title("mean ticks in cover per episode")

    if hold_g:
        ax[2][1].plot(hold_g, col("mean_peek_hold"), color="tab:cyan")
        ax[2][1].set_title("mean peek hold score   (\u2191 = proper peek cycles)")
    else:
        ax[2][1].text(0.5, 0.5, "no cover data yet",
                      ha="center", va="center", transform=ax[2][1].transAxes)
        ax[2][1].set_title("mean out-of-cover hold score")

    # --- aim behavior / lane usage ---
    if has_col("mean_aim_x_std") and has_col("mean_aim_y_std"):
        ax[3][0].plot(col_gens("mean_aim_x_std"), col("mean_aim_x_std"), label="aim_x std")
        ax[3][0].plot(col_gens("mean_aim_y_std"), col("mean_aim_y_std"), label="aim_y std")
        if has_col("mean_aim_dx"):
            ax[3][0].plot(col_gens("mean_aim_dx"), col("mean_aim_dx"), label="mean |aim dx|", linestyle="--")
        ax[3][0].set_title("aim movement variability")
        ax[3][0].legend(fontsize=8)
    else:
        ax[3][0].text(0.5, 0.5, "no aim-telemetry data yet",
                      ha="center", va="center", transform=ax[3][0].transAxes)
        ax[3][0].set_title("aim movement variability")

    if (
        has_col("mean_shot_left_frac")
        and has_col("mean_shot_mid_frac")
        and has_col("mean_shot_right_frac")
    ):
        ax[3][1].plot(col_gens("mean_shot_left_frac"),  [100 * v for v in col("mean_shot_left_frac")],  label="left")
        ax[3][1].plot(col_gens("mean_shot_mid_frac"),   [100 * v for v in col("mean_shot_mid_frac")],   label="mid")
        ax[3][1].plot(col_gens("mean_shot_right_frac"), [100 * v for v in col("mean_shot_right_frac")], label="right")
        ax[3][1].set_ylim(-5, 105)
        ax[3][1].set_title("shot lane distribution % (L/M/R)")
        ax[3][1].legend(fontsize=8)
    else:
        ax[3][1].text(0.5, 0.5, "no lane-telemetry data yet",
                      ha="center", va="center", transform=ax[3][1].transAxes)
        ax[3][1].set_title("shot lane distribution % (L/M/R)")

    for row in ax:
        for a in row:
            a.grid(alpha=0.3)
            a.set_xlabel("generation")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot -> {out_path}")
        if used_latest:
            print(f"Source log -> {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
