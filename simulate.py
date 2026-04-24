from os.path import join
import argparse
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import cupy as cp
from line_profiler import profile
from multiprocessing.pool import ThreadPool
from time import perf_counter
from numba import cuda


def load_data(load_dir, bid):
    SIZE = 512
    u = cp.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = cp.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = cp.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def visualise_temperature(u, interior_mask, bid=None, save_path=None, show=True):
    old_u = None
    old_mask = None
    if bid:
        old_u, old_mask = load_data(LOAD_DIR, bid)

    padded_mask = cp.zeros_like(u, dtype=bool)
    padded_mask[1:-1, 1:-1] = interior_mask

    interior_only = cp.ma.masked_where(~padded_mask, u)
    walls = cp.zeros_like(u, dtype=int)
    walls[u == 5] = 1
    walls[u == 25] = 2
    walls = cp.ma.masked_where(walls == 0, walls)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)

    if old_u is not None and old_mask is not None:
        old_padded_mask = cp.zeros_like(old_u, dtype=bool)
        old_padded_mask[1:-1, 1:-1] = old_mask
        old_categories = cp.zeros_like(old_u, dtype=int)
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


@cuda.jit
def cuda_kernel(u, u_new, interior_mask):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x # computing the index for the rows
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y # computing the index for the columns
    b = cuda.blockIdx.z # index of the building it is calculating

    if i >= 512 or j >= 512: # if the indexes of the points are outside the grid, it shouldn't do anything
        return
    if not interior_mask[b, i, j]: # if the indexes of the points are outside of our mask, we also don't want to do anything with them
        return 
    new_val = 0.25 * (u[b, i, j+1] + u[b, i+2, j+1] + u[b, i+1, j] + u[b, i+1, j+2])
    u_new[b, i+1, j+1] = new_val # Assign to new u to prevent bleeding over effects

def get_bpg(n, tpb):
    return (n + (tpb - 1)) // tpb

def jacobi(N, all_u, interior_mask, max_iter, atol=1e-6):
    u_new = all_u.copy()
    
    TPB = (16, 16, 1)
    bpg = (get_bpg(512, 16), get_bpg(512, 16), N)
    for _ in range(max_iter):
        cuda_kernel[bpg, TPB](all_u, u_new, interior_mask)
        u_new, all_u = all_u, u_new # pointer swap so previous u becomes freed

        # check for early termination
        # delta = cp.abs(u_new[:, 1:-1, 1:-1][interior_mask] - all_u[:, 1:-1,1:-1][interior_mask])
        # if delta.size == 0 or delta.max() < atol:
        #     break

    return u_new


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = cp.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = cp.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }


if __name__ == '__main__':
    # Load data

    test_function = cp.zeros((1, 514, 514)) # These functions just need to be of the same type as the one we're going to use later
    test_function_2 = cp.zeros((1, 512, 512), dtype='bool')
    jacobi(1, test_function, test_function_2, 20, 1e-4) # We need to run the function once for it to compile before we time everything

    start_timer = perf_counter()

    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = len(building_ids) # Process all
    else:
        N = int(sys.argv[1])
    
    building_ids = building_ids[:N]
    print(f'Processing {N} buildings...')
    print(f'Building IDs: {building_ids}')
    all_u = cp.empty((N, 514, 514))

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header

    CHUNK_SIZE = 100
    for c in range(0, N, CHUNK_SIZE): # Break it to process in chunks of CHUNK_SIZE to reduce mem used
        start = c
        end = min(c + CHUNK_SIZE, N)
        # Load floor plans
        all_u0 = cp.empty((CHUNK_SIZE, 514, 514))
        all_interior_mask = cp.empty((CHUNK_SIZE, 512, 512), dtype='bool')
        for i, bid in enumerate(building_ids[start:end]):
            u0, interior_mask = load_data(LOAD_DIR, bid)
            all_u0[i] = u0
            all_interior_mask[i] = interior_mask
        

        # Run jacobi iterations for each floor plan
        MAX_ITER = 20_000
        ABS_TOL = 1e-4

        u_chunk = jacobi(CHUNK_SIZE, all_u0, all_interior_mask, MAX_ITER, ABS_TOL)

        for bid, u, interior_mask in zip(building_ids[start:end], u_chunk, all_interior_mask):
            stats = summary_stats(u, interior_mask)
            print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
    
    end_time = perf_counter() - start_timer
    print(f'Total time taken: {end_time}')