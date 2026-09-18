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

    def series(key, scale=1.0):
        """(generations, values) for one column, skipping blank/non-numeric
        cells so older logs and non-vision_schedule runs plot cleanly."""
        xs, ys = [], []
        for r in rows:
            v = r.get(key, "")
            if v == "":
                continue
            try:
                y = scale * float(v)
            except ValueError:
                continue
            xs.append(int(r["gen"]))
            ys.append(y)
        return xs, ys

    # Every panel is data-driven so adding a column to logger.py only needs a
    # new entry here. Each series is (label, csv_key, scale, linestyle).
    PCT = 100.0
    panels = [
        {"title": "fitness", "series": [
            ("best", "best", 1.0, "-"),
            ("mean", "mean", 1.0, "-"),
            ("center theta", "theta_fitness", 1.0, "--")]},
        {"title": "fitness std (~0 => raise SIGMA) / spread", "series": [
            ("std", "std", 1.0, "-"),
            ("spread", "spread", 1.0, ":")]},
        {"title": "outcome rate %  (clear / timeout / dead)", "ylim": (-5, 105), "series": [
            ("clear", "clear_rate", PCT, "-"),
            ("timeout", "timeout_rate", PCT, "-"),
            ("dead", "dead_rate", PCT, "-")]},
        {"title": "clear time (ticks)  (\u2193 = faster)", "series": [
            ("best", "best_time", 1.0, "-"),
            ("pop mean", "mean_time", 1.0, "-"),
            ("center theta", "theta_time", 1.0, "--")]},
        {"title": "screens cleared / episode  (\u2191 = chaining)", "series": [
            ("pop mean", "mean_screens_cleared", 1.0, "-"),
            ("pop max", "max_screens_cleared", 1.0, "-"),
            ("center theta", "theta_screens_cleared", 1.0, "--")]},
        {"title": "accuracy %", "ylim": (-5, 105), "series": [
            ("pop mean", "mean_acc", PCT, "-"),
            ("pop best", "best_acc", PCT, "-"),
            ("center theta", "theta_acc", PCT, "--")]},
        {"title": "shots per episode (pop mean)", "series": [
            ("fired", "mean_shots_fired", 1.0, "-"),
            ("hit", "mean_shots_hit", 1.0, "-")]},
        {"title": "damage per episode  (\u2193 = fewer hits taken)", "series": [
            ("pop mean", "mean_damage", 1.0, "-"),
            ("best", "best_damage", 1.0, "-"),
            ("center theta", "theta_damage", 1.0, "--")]},
        {"title": "mean ticks in cover  (\u2193 = less camping)",
            "baselines": [(900, "always-cover (900)")], "series": [
            ("cover ticks", "mean_cover_time", 1.0, "-")]},
        {"title": "peek behaviour", "series": [
            ("flips", "mean_peek_flips", 1.0, "-"),
            ("hold score", "mean_peek_hold", 1.0, "-")]},
        {"title": "trigger / cover discipline (ticks)", "series": [
            ("dry fire", "mean_dry_fire", 1.0, "-"),
            ("exposed no-shot", "mean_no_shot_exposed", 1.0, "-"),
            ("hesitated cover", "mean_hesitated_cover", 1.0, "-"),
            ("reload correct", "mean_reload_correct", 1.0, "-"),
            ("continue-screen", "mean_continue_ticks", 1.0, ":")]},
        {"title": "aim movement variability", "series": [
            ("aim_x std", "mean_aim_x_std", 1.0, "-"),
            ("aim_y std", "mean_aim_y_std", 1.0, "-"),
            ("|aim dx|", "mean_aim_dx", 1.0, "--"),
            ("|aim dy|", "mean_aim_dy", 1.0, "--")]},
        {"title": "aim span (max-min)", "series": [
            ("span x", "mean_aim_span_x", 1.0, "-"),
            ("span y", "mean_aim_span_y", 1.0, "-")]},
        {"title": "shot lane distribution % (L/M/R)", "ylim": (-5, 105), "series": [
            ("left", "mean_shot_left_frac", PCT, "-"),
            ("mid", "mean_shot_mid_frac", PCT, "-"),
            ("right", "mean_shot_right_frac", PCT, "-")]},
        {"title": "hit rate by lane % (L/M/R)", "ylim": (-5, 105), "series": [
            ("left", "mean_hit_rate_left", PCT, "-"),
            ("mid", "mean_hit_rate_mid", PCT, "-"),
            ("right", "mean_hit_rate_right", PCT, "-")]},
        {"title": "reaction timing (ticks)  (\u2193 = faster)", "series": [
            ("hit delta", "mean_hit_delta", 1.0, "-"),
            ("reaction latency", "mean_reaction_latency", 1.0, "-")]},
        {"title": "gains \u2014 population mean  (\u21920 = ignoring signal)",
            "baselines": [(0.0, None)], "series": [
            ("vision", "mean_vision_gain", 1.0, "-"),
            ("shoot", "mean_shoot_gain", 1.0, "-"),
            ("drift", "mean_drift_gain", 1.0, "-"),
            ("peek", "mean_peek_gain", 1.0, "-"),
            ("ammo", "mean_ammo_gain", 1.0, "-")]},
        {"title": "gains \u2014 center theta",
            "baselines": [(0.0, None)], "series": [
            ("vision", "theta_vision_gain", 1.0, "-"),
            ("shoot", "theta_shoot_gain", 1.0, "-"),
            ("drift", "theta_drift_gain", 1.0, "-"),
            ("peek", "theta_peek_gain", 1.0, "-"),
            ("ammo", "theta_ammo_gain", 1.0, "-")]},
        {"title": "ES mutation step size (sigma)", "series": [
            ("sigma", "sigma_used", 1.0, "-")]},
    ]

    ncols = 2
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.1 * nrows), squeeze=False)
    flat = [a for row in axes for a in row]

    for ax, panel in zip(flat, panels):
        plotted = 0
        for label, key, scale, style in panel["series"]:
            xs, ys = series(key, scale)
            if not xs:
                continue
            ax.plot(xs, ys, style, label=label, linewidth=1.3)
            plotted += 1
        labeled_baseline = False
        if plotted:
            for by, blabel in panel.get("baselines", []):
                kw = {"color": "grey", "linestyle": "--", "linewidth": 0.8}
                if blabel:
                    kw["label"] = blabel
                    labeled_baseline = True
                ax.axhline(y=by, **kw)
            if "ylim" in panel:
                ax.set_ylim(*panel["ylim"])
            if plotted > 1 or labeled_baseline:
                ax.legend(fontsize=7, ncol=2)
        else:
            ax.text(0.5, 0.5, "no data yet", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title(panel["title"], fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlabel("generation")

    # Hide any unused trailing axes (odd panel count).
    for ax in flat[len(panels):]:
        ax.axis("off")

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
