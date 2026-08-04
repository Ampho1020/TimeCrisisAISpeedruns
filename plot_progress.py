"""Plot training_log.csv. Safe to run mid-training in a second terminal."""

import csv
import sys

import matplotlib.pyplot as plt


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "training_log.csv"
    with open(path) as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        print("No data yet.")
        return

    gens = [int(r["gen"]) for r in rows]
    col = lambda k: [float(r[k]) for r in rows if r.get(k, "") != ""]
    col_gens = lambda k: [int(r["gen"]) for r in rows if r.get(k, "") != ""]

    fig, ax = plt.subplots(3, 2, figsize=(11, 10))

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

    # --- cover behaviour ---
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

    for row in ax:
        for a in row:
            a.grid(alpha=0.3)
            a.set_xlabel("generation")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
