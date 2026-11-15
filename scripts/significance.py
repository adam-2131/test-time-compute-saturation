
import argparse
from pathlib import Path

from analysis import accuracy, flips, load_run, mcnemar_exact


ALPHA = 0.05


def main():
    parser = argparse.ArgumentParser(description="McNemar test between neighbouring k values")
    parser.add_argument("run_dir", nargs="?", default="experiments", type=Path)
    args = parser.parse_args()

    by_k, shared = load_run(args.run_dir)
    ks = sorted(by_k)

    print(f"n = {len(shared)}\n")
    print(f"{'comparison':<18}{'accuracy':<20}{'diff':>8}{'won':>8}{'lost':>8}{'p':>9}")
    print("-" * 71)

    for lo, hi in zip(ks, ks[1:]):
        acc_lo = accuracy(by_k[lo], shared)
        acc_hi = accuracy(by_k[hi], shared)

        lost, won = flips(by_k[lo], by_k[hi], shared)
        p = mcnemar_exact(len(lost), len(won))

        verdict = "significant" if p < ALPHA else "not significant"

        print(
            f"k={lo:<4}-> k={hi:<7}"
            f"{100 * acc_lo:5.1f}% -> {100 * acc_hi:5.1f}%"
            f"{100 * (acc_hi - acc_lo):+8.2f}{len(won):>8}{len(lost):>8}"
            f"{p:>9.4f}   {verdict}"
        )


if __name__ == "__main__":
    main()
