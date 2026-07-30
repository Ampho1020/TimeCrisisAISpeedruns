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
    col = lambda k: [float(r[k]) for r in rows]

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))

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

    for row in ax:
        for a in row:
            a.grid(alpha=0.3)
            a.set_xlabel("generation")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
