#!/usr/bin/env python3
"""Benchmark simulate.py scaling and plot speedup versus worker count."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt


def parse_core_counts(raw: str | None) -> list[int]:
    if raw:
        counts = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            value = int(token)
            if value < 1:
                raise ValueError("Core counts must be positive integers")
            counts.append(value)
        if not counts:
            raise ValueError("No valid core counts provided")
        return sorted(set(counts))

    cpu_total = os.cpu_count() or 1
    defaults = [1, 2, 4, 8, 16]
    counts = [c for c in defaults if c <= cpu_total]
    if 1 not in counts:
        counts.insert(0, 1)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure speedup of simulate.py as worker count increases"
    )
    parser.add_argument("--num-buildings", "-n", type=int, default=10,
                        help="Number of buildings passed to simulate.py")
    parser.add_argument("--repeats", "-r", type=int, default=3,
                        help="Benchmark repetitions per worker count")
    parser.add_argument("--cores", type=str, default=None,
                        help="Comma-separated worker counts, e.g. 1,2,4,8")
    parser.add_argument("--max-iter", type=int, default=20_000,
                        help="Maximum Jacobi iterations")
    parser.add_argument("--abs-tol", type=float, default=1e-4,
                        help="Absolute tolerance for Jacobi convergence")
    parser.add_argument("--csv", type=Path, default=Path("output/speedup_results.csv"),
                        help="Path to CSV output")
    parser.add_argument("--plot", type=Path, default=Path("output/speedup_plot.png"),
                        help="Path to speedup plot image")
    parser.add_argument("--dynamic", action="store_true",
                        help="Use dynamic scheduling instead of static")
    return parser.parse_args()


def run_once(project_dir: Path, num_buildings: int, workers: int, max_iter: int, abs_tol: float, dynamic: bool) -> float:
    cmd = [
        sys.executable,
        str(project_dir / "simulate.py"),
        str(num_buildings),
        "--workers",
        str(workers),
        "--max-iter",
        str(max_iter),
        "--abs-tol",
        str(abs_tol),
        "--no-plots",
    ]
    if dynamic:
        cmd.append("--dynamic")

    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=project_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start

    if proc.returncode != 0:
        raise RuntimeError(
            f"simulate.py failed for workers={workers} with exit code {proc.returncode}\n"
            f"stderr:\n{proc.stderr}"
        )

    return elapsed


def save_csv(rows: list[dict[str, float | int]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["workers", "mean_time_s", "std_time_s", "speedup", "efficiency"],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict[str, float | int]], plot_path: Path) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    workers = [int(row["workers"]) for row in rows]
    speedups = [row["speedup"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(workers, speedups, marker="o", linewidth=2, label="Measured speedup")
    plt.plot(workers, workers, "--", linewidth=1.5, label="Ideal speedup")
    plt.xlabel("Workers")
    plt.ylabel("Speedup (T1 / Tp)")
    plt.title("simulate.py Strong Scaling")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()

    if args.num_buildings < 1:
        raise ValueError("--num-buildings must be at least 1")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    project_dir = Path(__file__).resolve().parent
    core_counts = parse_core_counts(args.cores)
    if 1 not in core_counts:
        core_counts = [1] + core_counts

    results: list[dict[str, float | int]] = []

    for workers in core_counts:
        timings = [
            run_once(project_dir, args.num_buildings, workers, args.max_iter, args.abs_tol, args.dynamic)
            for _ in range(args.repeats)
        ]
        mean_time = statistics.mean(timings)
        std_time = statistics.pstdev(timings) if len(timings) > 1 else 0.0
        results.append(
            {
                "workers": workers,
                "mean_time_s": mean_time,
                "std_time_s": std_time,
                "speedup": 0.0,
                "efficiency": 0.0,
            }
        )
        print(f"workers={workers:>2} mean={mean_time:.3f}s std={std_time:.3f}s")

    baseline = next(row["mean_time_s"] for row in results if int(row["workers"]) == 1)
    for row in results:
        workers = int(row["workers"])
        speedup = baseline / row["mean_time_s"]
        row["speedup"] = speedup
        row["efficiency"] = speedup / workers

    save_csv(results, args.csv)
    save_plot(results, args.plot)

    print("\nSummary")
    for row in results:
        workers = int(row["workers"])
        print(
            f"workers={workers:>2} time={row['mean_time_s']:.3f}s "
            f"speedup={row['speedup']:.2f} efficiency={row['efficiency']:.2f}"
        )
    print(f"\nCSV written to: {args.csv}_{args.dynamic and 'dynamic' or 'static'}.csv")
    print(f"Plot written to: {args.plot}_{args.dynamic and 'dynamic' or 'static'}.csv")


if __name__ == "__main__":
    main()
