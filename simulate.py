from os.path import join
import argparse

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from line_profiler import profile
from multiprocessing.pool import ThreadPool


def parse_args():
    parser = argparse.ArgumentParser(description="Run building temperature simulation")
    parser.add_argument("num_buildings", nargs="?", type=int, default=1,
                        help="Number of buildings to simulate")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Number of worker threads")
    parser.add_argument("--max-iter", type=int, default=20_000,
                        help="Maximum Jacobi iterations")
    parser.add_argument("--abs-tol", type=float, default=1e-4,
                        help="Absolute tolerance for Jacobi convergence")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip saving comparison plots")
    parser.add_argument("--dynamic", action="store_true",
                        help="Use dynamic scheduling instead of static")
    return parser.parse_args()

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def visualise_temperature(u, interior_mask, bid=None, save_path=None, show=True):
    old_u = None
    old_mask = None
    if bid:
        old_u, old_mask = load_data(LOAD_DIR, bid)

    padded_mask = np.zeros_like(u, dtype=bool)
    padded_mask[1:-1, 1:-1] = interior_mask

    interior_only = np.ma.masked_where(~padded_mask, u)
    walls = np.zeros_like(u, dtype=int)
    walls[u == 5] = 1
    walls[u == 25] = 2
    walls = np.ma.masked_where(walls == 0, walls)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)

    if old_u is not None and old_mask is not None:
        old_padded_mask = np.zeros_like(old_u, dtype=bool)
        old_padded_mask[1:-1, 1:-1] = old_mask
        old_categories = np.zeros_like(old_u, dtype=int)
        old_categories[old_u == 5] = 1
        old_categories[old_u == 25] = 2
        old_categories[old_padded_mask] = 3

        floorplan_cmap = ListedColormap(["white", "#4c78a8", "#f58518", "#54a24b"])
        floorplan_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], floorplan_cmap.N)
        floorplan_plot = axes[0].imshow(old_categories, origin="lower", cmap=floorplan_cmap, norm=floorplan_norm)
        axes[0].set_title(f"Floor Plan: {bid}")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        colorbar = fig.colorbar(floorplan_plot, ax=axes[0], shrink=0.85, ticks=[0, 1, 2, 3])
        colorbar.ax.set_yticklabels(["outside", "load-bearing wall", "inside wall", "interior"])
    else:
        axes[0].axis("off")

    overlay_plot = axes[1].imshow(interior_only, origin="lower", cmap="coolwarm")
    wall_cmap = ListedColormap(["#4c78a8", "#f58518"])
    wall_norm = BoundaryNorm([0.5, 1.5, 2.5], wall_cmap.N)
    axes[1].imshow(walls, origin="lower", cmap=wall_cmap, norm=wall_norm, alpha=0.95)
    axes[1].set_title(f"Temperature of {bid}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(overlay_plot, ax=axes[1], shrink=0.85, label="Temperature")

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes

@profile
def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)

    for _ in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break
    return u

def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }


if __name__ == '__main__':
    args = parse_args()

    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    N = args.num_buildings
    if N < 1:
        raise ValueError("num_buildings must be at least 1")

    building_ids = building_ids[:N]

    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask
        

    # Run jacobi iterations for each floor plan
    MAX_ITER = args.max_iter
    ABS_TOL = args.abs_tol

    num_workers = max(1, args.workers)
    all_u = np.empty_like(all_u0)

    def worker_task(chunk):
        return [jacobi(all_u0[i], all_interior_mask[i], MAX_ITER, ABS_TOL) for i in chunk]

    if args.dynamic:
        # Dynamic scheduling: each task gets a single building for maximum load balancing
        with ThreadPool(num_workers) as pool:
            results = pool.map(worker_task, [[i] for i in range(N)])
    else:
        # Static scheduling: pre-split floor plans into fixed chunks per worker
        chunks = [chunk for chunk in np.array_split(np.arange(N), num_workers) if chunk.size > 0]
        with ThreadPool(num_workers) as pool:
            results = pool.map(worker_task, chunks)

    # Flatten results back into all_u
    idx = 0
    for chunk_result in results:
        for u in chunk_result:
            all_u[idx] = u
            idx += 1

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
        if not args.no_plots:
            visualise_temperature(u, interior_mask, bid, save_path=f"{bid}_comparison_floorplan.png", show=False)
