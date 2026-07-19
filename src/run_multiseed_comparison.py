"""
run_multiseed_comparison.py

A single BO run can get lucky/unlucky from its random initial points and
GP noise. This repeats both random_search and bayesian_optimization across
multiple seeds and reports mean +/- std final regret, plus a combined
long-format CSV for averaged regret curves with error bands in Phase 4.

Each row also carries the full candidate parameters (composition,
particle_size_nm, metal_loading_wt_pct, facet, calcination_temp_C,
defect_density) so this same log can be reused in Phase 5 to train a
SHAP-explainable surrogate on everything the BO loop evaluated.
"""

import csv
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(__file__))
from oracle import oracle
from search_space import SEARCH_SPACE, DIMENSION_NAMES
from skopt import Optimizer

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


def random_search_run(budget, seed):
    rng = np.random.default_rng(seed)
    best_so_far = -np.inf
    rows = []
    for i in range(1, budget + 1):
        candidate = _sample_random_candidate(rng)
        activity = oracle(**candidate, rng=rng)["Activity_pct"]
        best_so_far = max(best_so_far, activity)
        rows.append({"method": "random_search", "seed": seed, "iteration": i,
                      **candidate,
                      "activity": round(activity, 2), "best_so_far": round(best_so_far, 2),
                      "regret": round(TRUE_OPTIMUM_ACTIVITY - best_so_far, 2)})
    return rows


def bo_run(budget, seed, n_initial_points=10):
    rng = np.random.default_rng(seed)
    opt = Optimizer(dimensions=SEARCH_SPACE, base_estimator="GP", acq_func="EI",
                     n_initial_points=n_initial_points, random_state=seed)
    best_so_far = -np.inf
    rows = []
    for i in range(1, budget + 1):
        x = opt.ask()
        candidate = dict(zip(DIMENSION_NAMES, x))
        activity = oracle(**candidate, rng=rng)["Activity_pct"]
        opt.tell(x, -activity)
        best_so_far = max(best_so_far, activity)
        rows.append({"method": "bayesian_optimization", "seed": seed, "iteration": i,
                      **candidate,
                      "activity": round(activity, 2), "best_so_far": round(best_so_far, 2),
                      "regret": round(TRUE_OPTIMUM_ACTIVITY - best_so_far, 2)})
    return rows


def main(n_seeds=10, budget=50, out_csv="results/multiseed_log.csv"):
    all_rows = []
    random_finals, bo_finals = [], []

    for seed in range(n_seeds):
        r_rows = random_search_run(budget, seed)
        b_rows = bo_run(budget, seed)
        all_rows.extend(r_rows)
        all_rows.extend(b_rows)
        random_finals.append(r_rows[-1]["regret"])
        bo_finals.append(b_rows[-1]["regret"])
        print(f"seed {seed}: random final regret = {r_rows[-1]['regret']:.1f} | "
              f"BO final regret = {b_rows[-1]['regret']:.1f}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n=== {n_seeds}-seed summary (budget={budget}) ===")
    print(f"Random search final regret: {np.mean(random_finals):.1f} +/- {np.std(random_finals):.1f}")
    print(f"Bayesian optimization final regret: {np.mean(bo_finals):.1f} +/- {np.std(bo_finals):.1f}")
    print(f"Log saved to {out_csv}")


if __name__ == "__main__":
    main()