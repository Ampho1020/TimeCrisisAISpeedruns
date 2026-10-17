"""Pool per-episode eval records written by ``run_eval.py --shot-diag-out``.

    python pool_eval.py evals.jsonl [--group-by tag|checkpoint] [--on-target 0.08]

One episode swings roughly +/-6 accuracy points, so single runs cannot prove
a change. This pools episodes per group and reports shot-weighted accuracy with
a cluster-bootstrap CI (episodes are the resampling unit, since shots within an
episode are correlated), the blind/off/on-target split, and each later group's
difference from the first.
"""

import argparse
import json
import random
import sys


def load_records(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _acc(episodes: list[dict]) -> float:
    fired = sum(e["shots_fired"] for e in episodes)
    return sum(e["shots_hit"] for e in episodes) / fired if fired else 0.0


def bootstrap_ci(episodes: list[dict], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of pooled accuracy, resampling whole episodes."""
    if len(episodes) < 2:
        return (_acc(episodes), _acc(episodes))
    rng = random.Random(seed)
    stats = sorted(
        _acc([rng.choice(episodes) for _ in episodes]) for _ in range(n)
    )
    return stats[int(0.025 * n)], stats[int(0.975 * n) - 1]


def diff_ci(a: list[dict], b: list[dict], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Accuracy(b) - accuracy(a) with a 95% cluster-bootstrap CI."""
    rng = random.Random(seed)
    stats = sorted(
        _acc([rng.choice(b) for _ in b]) - _acc([rng.choice(a) for _ in a])
        for _ in range(n)
    )
    return _acc(b) - _acc(a), stats[int(0.025 * n)], stats[int(0.975 * n) - 1]


def split_buckets(shots: list[dict], on_target: float) -> dict[str, tuple[int, int]]:
    """(fired, hit) per blind / off-target / on-target bucket."""
    out = {"blind": [0, 0], "off": [0, 0], "on": [0, 0]}
    for s in shots:
        if not s["had_detection"]:
            key = "blind"
        elif s["nearest_dist"] >= on_target:
            key = "off"
        else:
            key = "on"
        out[key][0] += s["fired"]
        out[key][1] += s["hit"]
    return {k: (v[0], v[1]) for k, v in out.items()}


def group_episodes(records: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(str(r.get(key, "")), []).append(r)
    return groups


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def report(groups: dict[str, list[dict]], on_target: float) -> None:
    first_name, first = next(iter(groups.items()))
    for name, eps in groups.items():
        fired = sum(e["shots_fired"] for e in eps)
        hits = sum(e["shots_hit"] for e in eps)
        lo, hi = bootstrap_ci(eps)
        clears = sum(1 for e in eps if e.get("cleared"))
        per_ep = ", ".join(_pct(e["shots_hit"] / e["shots_fired"]) if e["shots_fired"] else "--" for e in eps)
        print(f"[{name}] episodes={len(eps)} clears={clears} fired={fired} hits={hits} "
              f"accuracy={_pct(_acc(eps))} (95% CI {_pct(lo)}..{_pct(hi)})")
        print(f"    per-episode: {per_ep}")
        shots = [s for e in eps for s in e.get("shots", [])]
        if shots:
            parts = []
            for k, (f_, h_) in split_buckets(shots, on_target).items():
                parts.append(f"{k} {f_} shots hit {_pct(h_ / f_) if f_ else '--'}")
            print("    " + " | ".join(parts))
        if name != first_name:
            d, dlo, dhi = diff_ci(first, eps)
            verdict = "significant" if dlo > 0 or dhi < 0 else "within noise"
            print(f"    vs [{first_name}]: {100*d:+.1f} pts (95% CI {100*dlo:+.1f}..{100*dhi:+.1f}) {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("records", help="JSONL written by run_eval.py --shot-diag-out")
    ap.add_argument("--group-by", default="tag", choices=["tag", "checkpoint"])
    ap.add_argument("--on-target", type=float, default=0.08)
    args = ap.parse_args()
    records = load_records(args.records)
    if not records:
        sys.exit(f"{args.records} has no episode records")
    report(group_episodes(records, args.group_by), args.on_target)


if __name__ == "__main__":
    main()
