"""
search_space.py

Defines the mixed categorical/continuous search space for Bayesian
optimization, shared by the baseline (random/grid) and BO (skopt) scripts
in later phases.
"""

from skopt.space import Real, Categorical

SEARCH_SPACE = [
    Categorical(["Cu2O", "Cu2O-Pd", "Cu2O-Au", "Cu2O-Ag"], name="composition"),
    Real(20.0, 300.0, name="particle_size_nm"),
    Real(0.1, 5.0, name="metal_loading_wt_pct"),
    Categorical(["111", "110", "100"], name="facet"),
    Real(200.0, 600.0, name="calcination_temp_C"),
    Categorical(["low", "medium", "high"], name="defect_density"),
]

DIMENSION_NAMES = [dim.name for dim in SEARCH_SPACE]