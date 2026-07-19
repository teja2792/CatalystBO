"""
plot_regret_curves.py

Phase 4: convergence comparison. Plots mean best-activity-found-so-far vs.
iteration for random search and Bayesian optimization (10-seed mean +/- 1
std band). Grid search's single result (4500 evaluations, NOT budget-
matched) is shown as a horizontal reference line for context only.
"""

import pandas as pd
import matplotlib.pyplot as plt

TRUE_OPTIMUM_ACTIVITY = 97.0
GRID_SEARCH_BEST = 85.1        # from results/grid_search_log.csv, 4500 evaluations
GRID_SEARCH_EVALS = 4500

df = pd.read_csv("results/multiseed_log.csv")

fig, ax = plt.subplots(figsize=(8, 5.5))

colors = {"random_search": "tab:gray", "bayesian_optimization": "tab:blue"}
labels = {"random_search": "Random search (10 seeds)",
          "bayesian_optimization": "Bayesian optimization (10 seeds)"}

for method in ["random_search", "bayesian_optimization"]:
    sub = df[df["method"] == method]
    stats = sub.groupby("iteration")["best_so_far"].agg(["mean", "std"]).reset_index()
    ax.plot(stats["iteration"], stats["mean"], label=labels[method],
            color=colors[method], linewidth=2)
    ax.fill_between(stats["iteration"], stats["mean"] - stats["std"],
                     stats["mean"] + stats["std"], color=colors[method], alpha=0.15)

ax.axhline(TRUE_OPTIMUM_ACTIVITY, color="black", linestyle="--", linewidth=1,
           label=f"True optimum ({TRUE_OPTIMUM_ACTIVITY:.0f}%, noiseless)")
ax.axhline(GRID_SEARCH_BEST, color="tab:orange", linestyle=":", linewidth=1.5,
           label=f"Grid search best ({GRID_SEARCH_BEST:.1f}%, {GRID_SEARCH_EVALS} evals -- not budget-matched)")

ax.set_xlabel("Evaluation (synthesis + test) number")
ax.set_ylabel("Best activity found so far (%)")
ax.set_title("Sample efficiency: Bayesian optimization vs. random search\n"
              "(mean +/- 1 std across 10 seeds, budget = 50 evaluations)")
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(0, 105)
fig.tight_layout()
fig.savefig("results/regret_curves.png", dpi=150)
print("Saved results/regret_curves.png")