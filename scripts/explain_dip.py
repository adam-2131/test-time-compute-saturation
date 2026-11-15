"""Which problems change from correct to wrong (or back) between two k values.

Written to check why accuracy drops slightly at k = 1000: is it a few
low-confidence problems flipping, or a real decline?

    python scripts/explain_dip.py
    python scripts/explain_dip.py experiments_gsm8k --from-k 10 --to-k 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis import (
    accuracy,
    discover_k_values,
    flips,
    load_k,
    mcnemar_exact,
    shared_ids,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", nargs="?", default="experiments", type=Path)
    parser.add_argument("--from-k", type=int, default=None)
    parser.add_argument("--to-k", type=int, default=None)
    args = parser.parse_args()

    # default: the last two k values
    available = discover_k_values(args.run_dir)
    from_k = args.from_k if args.from_k is not None else available[-2]
    to_k = args.to_k if args.to_k is not None else available[-1]

    before, after = load_k(args.run_dir, from_k), load_k(args.run_dir, to_k)
    ids = shared_ids(before, after)
    n = len(ids)

    acc_before = accuracy(before, ids)
    acc_after = accuracy(after, ids)
    print(f"n = {n}")
    print(f"k={from_k}  accuracy {acc_before:.4f}")
    print(f"k={to_k} accuracy {acc_after:.4f}   "
          f"delta {100 * (acc_after - acc_before):+.2f} pp\n")

    lost, won = flips(before, after, ids)
    print(f"correct at {from_k}, wrong at {to_k} : {len(lost)}")
    print(f"wrong at {from_k}, correct at {to_k} : {len(won)}")
    print(f"net                           : {len(won) - len(lost)}\n")

    for label, group in (("LOST", lost), ("WON", won)):
        for i in group:
            print(f"{label:<5} {i:<14} "
                  f"conf {float(before[i]['confidence']):.3f}->"
                  f"{float(after[i]['confidence']):.3f}  "
                  f"entropy {float(before[i]['entropy']):.2f}->{float(after[i]['entropy']):.2f}  "
                  f"k_used {before[i]['k_used']}->{after[i]['k_used']}")

    changed = lost + won
    if changed:
        peak = max(float(after[i]["confidence"]) for i in changed)
        print(f"\nhighest confidence among changed problems: {peak:.3f}")

    if changed:
        p = mcnemar_exact(len(lost), len(won))
        print(f"\nMcNemar exact test on {len(changed)} discordant problems "
              f"({len(lost)} lost / {len(won)} won): p = {p:.3f}")

    se = (acc_after * (1 - acc_after) / n) ** 0.5
    print(f"standard error of accuracy at n={n}: {100 * se:.2f} pp")
    print(f"observed difference is {abs(acc_after - acc_before) / se:.2f} standard errors")


if __name__ == "__main__":
    main()
