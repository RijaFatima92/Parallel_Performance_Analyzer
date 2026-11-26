"""Run everything end-to-end:  python src/main.py
Workload: Option 3 - Monte-Carlo estimation of pi with n samples.
Steps: benchmark (serial + parallel) -> metrics -> plots.
Set env QUICK=1 for a fast smoke test (small n, 2 repeats)."""
import csv
import os
import sys
import time
import multiprocessing as mp

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics  # noqa: E402
import plot     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw_timings.csv")
QUICK = os.environ.get("QUICK") == "1"
SIZES = [10**4, 10**5, 10**6] if QUICK else [10**6, 10**7, 10**8]
PROCS = [1, 2, 4, 8]
REPEATS = 2 if QUICK else 5
CHUNK = 2_000_000          # samples per vectorised batch (bounds memory)


def count_inside(args):
    """Count points inside the unit quarter-circle for `n` samples (vectorised)."""
    n, seed = args
    rng = np.random.default_rng(seed)
    inside, left = 0, n
    while left > 0:
        m = min(CHUNK, left)
        x = rng.random(m)
        y = rng.random(m)
        inside += int(np.count_nonzero(x * x + y * y <= 1.0))
        left -= m
    return inside


def serial_pi(n, seed=0):
    """T_S: best single-threaded version (NumPy-vectorised, no process overhead)."""
    return 4.0 * count_inside((n, seed)) / n


def parallel_pi(n, p, pool, seed=0):
    """Split n samples over p workers; independent RNG stream per worker; reduce counts."""
    base, extra = divmod(n, p)
    jobs = [(base + (1 if i < extra else 0), seed + i + 1) for i in range(p)]
    return 4.0 * sum(pool.map(count_inside, jobs)) / n


def timed(fn, *a):
    t0 = time.perf_counter()
    val = fn(*a)
    return time.perf_counter() - t0, val


def benchmark():
    os.makedirs(os.path.dirname(RAW), exist_ok=True)
    rows = []
    ctx = mp.get_context("spawn" if sys.platform == "win32" else "fork")
    for n in SIZES:
        for run in range(1, REPEATS + 1):
            t, est = timed(serial_pi, n)
            rows.append(("mc_pi_serial", n, 1, run, t))
        print(f"[serial]   n={n:>11,}  last pi~{est:.5f}")
        for p in PROCS:
            with ctx.Pool(p) as pool:
                pool.map(count_inside, [(10, 0)] * p)   # warm-up: start workers
                for run in range(1, REPEATS + 1):
                    t, est = timed(parallel_pi, n, p, pool)
                    rows.append(("mc_pi_parallel", n, p, run, t))
            print(f"[parallel] n={n:>11,} p={p}  last pi~{est:.5f}")
    with open(RAW, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "n", "p", "run", "time_sec"])
        w.writerows(rows)
    print(f"Saved {len(rows)} measurements -> {RAW}")


if __name__ == "__main__":
    benchmark()
    metrics.main()
    plot.main()
