"""
bo_search.py

The actual Bayesian optimization loop. Uses a Gaussian Process surrogate
with Expected Improvement acquisition (the canonical BO setup) to decide
which candidate to evaluate next, given everything observed so far.

Budget-matched to random_search (50 evaluations) for a fair regret
comparison in Phase 4.

Cold-start note: a GP can't be fit meaningfully with zero data, so the
first n_initial_points candidates are chosen by random space-filling
design (same as random_search would pick), and only after that does the
GP-fitted acquisition function start guiding candidate selection.
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


def run_bayesian_optimization(budget=50, seed=0, n_initial_points=10,
                               out_csv="results/bo_log.csv"):
    rng = np.random.default_rng(seed)

    opt = Optimizer(
        dimensions=SEARCH_SPACE,
        base_estimator="GP",
        acq_func="EI",
        n_initial_points=n_initial_points,
        random_state=seed,
    )

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    best_so_far = -np.inf
    rows = []

    for i in range(1, budget + 1):
        x = opt.ask()
        candidate = dict(zip(DIMENSION_NAMES, x))
        result = oracle(**candidate, rng=rng)
        activity = result["Activity_pct"]

        # skopt's Optimizer MINIMIZES by default -- negate since we want
        # to maximize activity.
        opt.tell(x, -activity)

        best_so_far = max(best_so_far, activity)
        regret = TRUE_OPTIMUM_ACTIVITY - best_so_far
        rows.append({
            "iteration": i, "method": "bayesian_optimization",
            **candidate, "activity": round(activity, 2),
            "best_so_far": round(best_so_far, 2), "regret": round(regret, 2),
        })

        phase = "random-init" if i <= n_initial_points else "GP-guided"
        print(f"iter {i:2d} [{phase:10s}] {candidate['composition']:8s} | "
              f"size={candidate['particle_size_nm']:5.0f}nm | "
              f"loading={candidate['metal_loading_wt_pct']:.2f}wt% | "
              f"facet={candidate['facet']:3s} | "
              f"calc={candidate['calcination_temp_C']:5.0f}C | "
              f"defect={candidate['defect_density']:6s} -> {activity:5.1f}% "
              f"(best: {best_so_far:5.1f}%, regret: {regret:5.1f})")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nBO done. Best found: {best_so_far:.1f}% "
          f"(true optimum: {TRUE_OPTIMUM_ACTIVITY:.1f}%, "
          f"final regret: {TRUE_OPTIMUM_ACTIVITY - best_so_far:.1f})")
    print(f"Log saved to {out_csv}")
    return rows


if __name__ == "__main__":
    run_bayesian_optimization(budget=50, seed=0)