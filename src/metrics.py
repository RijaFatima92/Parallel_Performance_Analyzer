"""Reads results/raw_timings.csv, computes S, E, C, T_o, f for every (n, p)."""
import math
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw_timings.csv")
OUT = os.path.join(ROOT, "results", "metrics.csv")


def speedup(ts, tp):            return ts / tp
def efficiency(s, p):           return s / p
def cost(p, tp):                return p * tp
def total_overhead(p, tp, ts):  return p * tp - ts
def serial_fraction(s, p):
    """f = (1/S - 1/p) / (1 - 1/p); undefined for p = 1."""
    return float("nan") if p == 1 else (1 / s - 1 / p) / (1 - 1 / p)


def compute(raw_path=RAW):
    df = pd.read_csv(raw_path)
    med = df.groupby(["workload", "n", "p"])["time_sec"].median().reset_index()
    ts = med[med.workload.str.endswith("serial")].set_index("n")["time_sec"]
    par = med[med.workload.str.endswith("parallel")]
    rows = []
    for _, r in par.sort_values(["n", "p"]).iterrows():
        n, p, tp = int(r.n), int(r.p), r.time_sec
        t_s = ts[n]
        s = speedup(t_s, tp)
        rows.append(dict(n=n, p=p, T_S=t_s, T_P=tp, speedup=s,
                         efficiency=efficiency(s, p), cost=cost(p, tp),
                         overhead=total_overhead(p, tp, t_s),
                         serial_fraction=serial_fraction(s, p)))
    return pd.DataFrame(rows)


def isoefficiency_estimate(m, target=0.6):
    """Bonus: n needed for E >= target at each p, from the largest measured n.
    E = W/(W+T_o) >= target  =>  W >= target/(1-target) * T_o,  with W = T_S ~ n,
    so n_needed = n * (target/(1-target)) * T_o / T_S (assumes T_o stays fixed as n grows,
    i.e. a lower bound on the n needed)."""
    out = []
    big = m[m.n == m.n.max()]
    for _, r in big.iterrows():
        if r.p == 1:
            continue
        need = r.n if r.overhead <= 0 else r.n * (target / (1 - target)) * r.overhead / r.T_S
        out.append((int(r.p), max(need, float(r.n))))
    return out


def main():
    m = compute()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    m.to_csv(OUT, index=False)
    hdr = f"{'n':>12} {'p':>2} {'T_S(s)':>9} {'T_P(s)':>9} {'S':>6} {'E':>6} {'Cost(s)':>9} {'T_o(s)':>9} {'f':>7}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for _, r in m.iterrows():
        f = "n/a" if math.isnan(r.serial_fraction) else f"{r.serial_fraction:.4f}"
        print(f"{int(r.n):>12,} {int(r.p):>2} {r.T_S:9.4f} {r.T_P:9.4f} {r.speedup:6.2f} "
              f"{r.efficiency:6.2f} {r.cost:9.4f} {r.overhead:9.4f} {f:>7}")
    print("\nBonus - estimated n needed for E >= 0.6 (isoefficiency, from largest measured n):")
    for p, need in isoefficiency_estimate(m):
        print(f"  p = {p}: n >= {need:,.0f}")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
