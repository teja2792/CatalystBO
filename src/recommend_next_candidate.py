"""
recommend_next_candidate.py

Phase 6: the "what should I synthesize next" output -- the concrete demo
this repo exists to produce.

Runs one full BO loop (budget=50, same design as bo_search.py) and reports:
  1. The best candidate ACTUALLY EVALUATED during the loop (empirical result)
  2. The single next candidate the GP surrogate recommends evaluating, with
     its PREDICTED mean and uncertainty (posterior std) -- the optimizer's
     live "try this next" suggestion, not something it has tested yet.

The uncertainty on #2 is the GP's own belief about itself, reported as a
predicted range -- not a guarantee.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(__file__))
from oracle import oracle
from search_space import SEARCH_SPACE, DIMENSION_NAMES
from skopt import Optimizer

TRUE_OPTIMUM_ACTIVITY = 97.0


def run_and_recommend(budget=50, seed=0, n_initial_points=10):
    rng = np.random.default_rng(seed)
    opt = Optimizer(dimensions=SEARCH_SPACE, base_estimator="GP", acq_func="EI",
                     n_initial_points=n_initial_points, random_state=seed)

    best_activity = -np.inf
    best_candidate = None

    for i in range(1, budget + 1):
        x = opt.ask()
        candidate = dict(zip(DIMENSION_NAMES, x))
        activity = oracle(**candidate, rng=rng)["Activity_pct"]
        opt.tell(x, -activity)
        if activity > best_activity:
            best_activity = activity
            best_candidate = candidate

    # Ask the fitted GP for its next recommended point -- not yet tested.
    next_x = opt.ask()
    next_candidate = dict(zip(DIMENSION_NAMES, next_x))
    gp_model = opt.models[-1]
    x_transformed = opt.space.transform([next_x])
    pred_mean, pred_std = gp_model.predict(x_transformed, return_std=True)
    predicted_activity = -pred_mean[0]  # model was trained on -activity

    print("=" * 70)
    print(f"BEST CANDIDATE EVALUATED DURING OPTIMIZATION ({budget} syntheses tested)")
    print("=" * 70)
    for k, v in best_candidate.items():
        print(f"  {k}: {v}")
    print(f"  -> measured activity: {best_activity:.1f}%")
    print(f"  (true noiseless optimum: {TRUE_OPTIMUM_ACTIVITY:.1f}%, "
          f"gap: {TRUE_OPTIMUM_ACTIVITY - best_activity:.1f} pts)")

    print()
    print("=" * 70)
    print("NEXT CANDIDATE RECOMMENDED BY THE OPTIMIZER (not yet synthesized)")
    print("=" * 70)
    for k, v in next_candidate.items():
        print(f"  {k}: {v}")
    print(f"  -> predicted activity: {predicted_activity:.1f}% "
          f"+/- {pred_std[0]:.1f}% (GP posterior belief, unverified)")
    print()
    print("Interpretation: after 50 simulated syntheses, don't synthesize a")
    print("51st random catalyst -- this specific candidate is where the")
    print("model's evidence says the next test is most likely to help.")

    return best_candidate, best_activity, next_candidate, predicted_activity, pred_std[0]


if __name__ == "__main__":
    run_and_recommend()