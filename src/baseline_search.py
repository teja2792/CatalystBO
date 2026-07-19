"""
baseline_search.py

Baseline strategies to compare against Bayesian optimization in Phase 4:
  - random_search: uniform random sampling, budget-matched to BO (fair comparison)
  - full_grid_search: exhaustive coarse grid, reported separately as a
    "resolution cost" reference -- NOT budget-matched, shows how many
    evaluations grid search needs to cover the space at comparable resolution

Ground truth: the oracle is a product of independent per-variable factors,
each individually maximized at 1.0. The analytical global optimum is:
    Cu2O-Pd, particle_size_nm=48, metal_loading_wt_pct=2.0,
    facet="111", calcination_temp_C=350, defect_density="medium"
    -> noiseless activity = 97.0%
Used below as TRUE_OPTIMUM_ACTIVITY for regret curves.
"""

import csv
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(__file__))
from oracle import oracle

TRUE_OPTIMUM_ACTIVITY = 97.0

COMPOSITIONS = ["Cu2O", "Cu2O-Pd", "Cu2O-Au", "Cu2O-Ag"]
FACETS = ["111", "110", "100"]
DEFECTS = ["low", "medium", "high"]


def _sample_random_candidate(rng):
    return dict(
        composition=rng.choice(COMPOSITIONS),
        particle_size_nm=rng.uniform(20.0, 300.0),
        metal_loading_wt_pct=rng.uniform(0.1, 5.0),
        facet=rng.choice(FACETS),
        calcination_temp_C=rng.uniform(200.0, 600.0),
        defect_density=rng.choice(DEFECTS),
    )


def run_random_search(budget=50, seed=0, out_csv="results/random_search_log.csv"):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    best_so_far = -np.inf
    rows = []
    for i in range(1, budget + 1):
        candidate = _sample_random_candidate(rng)
        result = oracle(**candidate, rng=rng)
        activity = result["Activity_pct"]
        best_so_far = max(best_so_far, activity)
        regret = TRUE_OPTIMUM_ACTIVITY - best_so_far
        rows.append({
            "iteration": i, "method": "random_search",
            **candidate, "activity": round(activity, 2),
            "best_so_far": round(best_so_far, 2), "regret": round(regret, 2),
        })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Random search done. Best found: {best_so_far:.1f}% "
          f"(true optimum: {TRUE_OPTIMUM_ACTIVITY:.1f}%, "
          f"final regret: {TRUE_OPTIMUM_ACTIVITY - best_so_far:.1f})")
    print(f"Log saved to {out_csv}")
    return rows


def run_full_grid_search(out_csv="results/grid_search_log.csv",
                          size_levels=5, loading_levels=5, calc_levels=5, seed=0):
    """
    Exhaustive coarse grid. NOT budget-matched -- establishes how many
    evaluations a grid strategy needs at this resolution, reported as
    context in the README, not as a head-to-head curve against random/BO.
    """
    rng = np.random.default_rng(seed)
    sizes = np.linspace(20.0, 300.0, size_levels)
    loadings = np.linspace(0.1, 5.0, loading_levels)
    calcs = np.linspace(200.0, 600.0, calc_levels)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    best_so_far = -np.inf
    rows = []
    i = 0
    for composition in COMPOSITIONS:
        for size in sizes:
            for loading in loadings:
                for facet in FACETS:
                    for calc in calcs:
                        for defect in DEFECTS:
                            i += 1
                            candidate = dict(
                                composition=composition, particle_size_nm=size,
                                metal_loading_wt_pct=loading, facet=facet,
                                calcination_temp_C=calc, defect_density=defect,
                            )
                            result = oracle(**candidate, rng=rng)
                            activity = result["Activity_pct"]
                            best_so_far = max(best_so_far, activity)
                            rows.append({
                                "iteration": i, "method": "grid_search",
                                **candidate, "activity": round(activity, 2),
                                "best_so_far": round(best_so_far, 2),
                            })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Full grid search done. {i} total evaluations. "
          f"Best found: {best_so_far:.1f}% (true optimum: {TRUE_OPTIMUM_ACTIVITY:.1f}%)")
    print(f"Log saved to {out_csv}")
    return rows


if __name__ == "__main__":
    run_random_search(budget=50, seed=0)
    run_full_grid_search()