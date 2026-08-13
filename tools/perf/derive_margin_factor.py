"""Derive `bench_instrument::RESOLUTION_MARGIN_FACTOR` from the committed H100 round logs.

    python tools/perf/derive_margin_factor.py            # from the repo root
    python tools/perf/derive_margin_factor.py <log-dir>

THE QUESTION. `bench_instrument`'s resolution threshold `T = max(|peer floor|, our spread)` is a
WITHIN-round statistic: it is computed from the five rounds of one container invocation. A published
claim answers a different question -- will this number still be there next round? -- and the two are
not the same. This script measures the gap, in units of T, so the margin band's constant is derived
from the campaign's own logs rather than chosen.

THE INPUT is the per-cell verdict blocks of every `wgmma_config_sweep` round log. They are parsed
rather than the wide "published table", whose columns shift whenever a cell reads `-- not measured`.

    bench wgmma_config_sweep@<row>/<shape>: floor +X.XX% (peer A-vs-C), contender spread +Y.YY% ...
      field       A          C          B        C/A-1    B/A-1   B spread  verdict
      gemm_ms  1.272022   1.264329   1.246756   -0.60%   -1.99%    +1.65%   moved

Older rounds print the header without the `contender spread` term; those cells contribute
`T = |control gap|` and are handled.

THE OUTPUT is two cuts, both reported because they bound the answer from opposite sides:
  * ALL PAIRS -- every unordered pair of rounds in which a given (row, shape) was measured. Widest
    corpus, and it includes drift from code changes between waves as well as from the machine, so it
    is the pessimistic bound.
  * THE TIGHT CUT -- only the two consecutive same-format rounds (2026-08-11 w2-r3 and 2026-08-12
    w3-r2), one day apart. Fewest confounds, so it is the optimistic bound.

Both are printed with the fraction of cells whose between-round drift exceeded k*T. The band's
constant should be the smallest integer k that the DATA supports, and both cuts should agree on it.
"""

import os
import re
import sys

LOGS = [
    "2026-08-10-h100-act2-r2-cluster-sweep.log",
    "2026-08-10-h100-act2-r3-bmulticast.log",
    "2026-08-11-h100-w2-r3-config-sweep.log",
    "2026-08-12-h100-w3-r2-config-sweep.log",
]
TIGHT = ("2026-08-11-h100-w2-r3-config-sweep.log", "2026-08-12-h100-w3-r2-config-sweep.log")

BENCH_NEW = re.compile(
    r"bench\s+(\S+?)@(\S+?)/(\S+?):\s*floor\s*([+-][\d.]+)%\s*\(peer A-vs-C\),\s*"
    r"contender spread\s*([+-][\d.]+)%\s*\(ours\)"
)
BENCH_OLD = re.compile(r"bench\s+(\S+?)@(\S+?)/(\S+?):\s*floor\s*([+-][\d.]+)%\s*\(bar")
GEMM = re.compile(
    r"^\s*(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)%\s+([+-][\d.]+)%"
    r"(?:\s+([+-][\d.]+)%)?\s+(\w+)\s*$"
)


def parse(root):
    """Every per-cell verdict block in every log, as dicts. Percent units throughout."""
    cells = []
    for name in LOGS:
        path = os.path.join(root, name)
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read().replace("\r\n", "\n").replace("\r", "\n")
        pending = None
        for line in text.split("\n"):
            m = BENCH_NEW.search(line) or BENCH_OLD.search(line)
            if m:
                spread = float(m.group(5)) if m.re is BENCH_NEW else None
                pending = (m.group(2), m.group(3), float(m.group(4)), spread)
                continue
            if pending is None:
                continue
            g = GEMM.match(line)
            if g and g.group(1) != "field":
                arm, shape, gap, hdr_spread = pending
                cells.append(
                    dict(
                        log=name,
                        arm=arm,
                        shape=shape,
                        a=float(g.group(2)),
                        b=float(g.group(4)),
                        gap=gap,
                        effect=float(g.group(6)),
                        # The header's `contender spread` and the row's `B spread` are the same
                        # number; take the row's when present, since it is per-field.
                        spread=float(g.group(7)) if g.group(7) is not None else hdr_spread,
                        verdict=g.group(8),
                    )
                )
                pending = None
    return cells


def thr(c):
    """The within-round resolution threshold, percent: max(|peer floor|, our spread)."""
    return max(abs(c["gap"]), c["spread"] or 0.0)


def ratio(c):
    """The published ratio A/B on a millisecond field == (%-of-peer)/100."""
    return c["a"] / c["b"]


def quantile(xs, q):
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def report(label, pairs):
    """One cut: the distribution of between-round drift measured in units of T."""
    if not pairs:
        print("  %s: no comparable pairs" % label)
        return
    rs = sorted(p[1] for p in pairs)
    ds = sorted(p[0] for p in pairs)
    print("\n%s  (n = %d round-pairs)" % (label, len(rs)))
    print("  drift |R_i/R_j - 1|, percent : p50 %.3f  p75 %.3f  p90 %.3f  max %.3f"
          % (quantile(ds, .5), quantile(ds, .75), quantile(ds, .9), ds[-1]))
    print("  drift / T (margin units)    : p50 %.2f  p75 %.2f  p90 %.2f  max %.2f"
          % (quantile(rs, .5), quantile(rs, .75), quantile(rs, .9), rs[-1]))
    for k in (1.0, 1.5, 2.0, 3.0):
        n = sum(1 for r in rs if r > k)
        print("    drift exceeds %.1fxT for %5.1f%% of pairs   (%d/%d)"
              % (k, 100.0 * n / len(rs), n, len(rs)))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join("bench", "gpu", "h100")
    cells = parse(root)
    print("parsed %d per-cell verdict blocks from %d logs" % (len(cells), len(LOGS)))
    for name in LOGS:
        print("  %4d  %s" % (sum(1 for c in cells if c["log"] == name), name))

    by_key = {}
    for c in cells:
        by_key.setdefault((c["arm"], c["shape"]), []).append(c)
    replicated = {k: v for k, v in by_key.items() if len(v) >= 2}
    hist = {}
    for v in replicated.values():
        hist[len(v)] = hist.get(len(v), 0) + 1
    print("\n%d (row, shape) keys measured in two or more rounds:" % len(replicated))
    for n in sorted(hist):
        print("  in %d rounds: %2d keys  -> %d unordered pairs each"
              % (n, hist[n], n * (n - 1) // 2))

    every, tight = [], []
    for cs in replicated.values():
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                ci, cj = cs[i], cs[j]
                t = max(thr(ci), thr(cj))
                if t <= 0.0:
                    continue  # a round that measured no noise cannot scale anything
                drift = abs(ratio(ci) / ratio(cj) - 1.0) * 100.0
                every.append((drift, drift / t))
                if {ci["log"], cj["log"]} == set(TIGHT):
                    tight.append((drift, drift / t))

    report("ALL PAIRS (pessimistic: includes code drift between waves)", every)
    report("TIGHT CUT (w2-r3 vs w3-r2, one day apart, same log format)", tight)

    # The two audited cells, and what a claimed margin buys against the measured drift.
    rs = sorted(p[1] for p in every)
    print("\nP(between-round drift alone exceeds a claimed margin m), ALL PAIRS:")
    for m in (0.96, 1.21, 1.5, 2.0, 3.0):
        n = sum(1 for r in rs if r > m)
        print("  m = %.2fx -> %5.1f%%" % (m, 100.0 * n / len(rs)))
    print("\nthe two cells the 2026-08-13 retraction is about:")
    for c in cells:
        if c["log"].startswith("2026-08-12") and c["arm"] == "w1_mcb_v2_r16_p" \
                and c["shape"] in ("sq8192", "gpt_d4096_up"):
            print("  %-16s effect %+.2f%%  T %.2f%%  margin %.2fx  verdict %-8s  %.1f%% of peer"
                  % (c["shape"], c["effect"], thr(c), abs(c["effect"]) / thr(c), c["verdict"],
                     100.0 * ratio(c)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
