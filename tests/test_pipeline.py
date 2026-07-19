"""
tests/test_pipeline.py

Oracle/search-space tests are statistical (oracle has genuine +/-3% noise
by design) -- tolerances are set wide enough to be robust across runs
while still catching real regressions.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from oracle import oracle
from search_space import SEARCH_SPACE, DIMENSION_NAMES
from baseline_search import _sample_random_candidate, COMPOSITIONS, FACETS, DEFECTS
from bo_search import run_bayesian_optimization


def _average_activity(candidate, n=200, seed=0):
    rng = np.random.default_rng(seed)
    activities = [oracle(**candidate, rng=rng)["Activity_pct"] for _ in range(n)]
    return np.mean(activities)


NEAR_OPTIMAL_PD = dict(
    composition="Cu2O-Pd", particle_size_nm=48.0, metal_loading_wt_pct=2.0,
    facet="111", calcination_temp_C=350.0, defect_density="medium",
)


def test_composition_ordering_matches_grounding():
    """
    STRONG-tier claim: Cu2O-Pd >> Cu2O (real, from the source paper).
    MODERATE-tier claim: Cu2O-Pd > Cu2O-Ag > Cu2O-Au given the base
    activities chosen in oracle.py.
    """
    activities = {comp: _average_activity(dict(NEAR_OPTIMAL_PD, composition=comp))
                  for comp in COMPOSITIONS}
    assert activities["Cu2O-Pd"] > activities["Cu2O-Ag"] > activities["Cu2O-Au"] > activities["Cu2O"]
    assert activities["Cu2O-Pd"] - activities["Cu2O"] > 50


def test_known_optimum_recovers_analytical_value():
    """
    Every oracle factor is an independent multiplicative term maxed at 1.0,
    so this candidate should average close to the documented 97.0%
    (noiseless) value.
    """
    avg = _average_activity(NEAR_OPTIMAL_PD, n=500)
    assert 94.0 < avg < 100.0


def test_temperature_rise_is_a_decoy():
    """
    STRONG-tier claim: the ~10C temperature rise is real and independent
    of catalyst/activity -- confirm it doesn't differ between a
    high-activity and low-activity candidate.
    """
    rng = np.random.default_rng(1)
    high = [oracle(**NEAR_OPTIMAL_PD, rng=rng)["Temp_Rise_C"] for _ in range(200)]
    low_candidate = dict(NEAR_OPTIMAL_PD, composition="Cu2O")
    low = [oracle(**low_candidate, rng=rng)["Temp_Rise_C"] for _ in range(200)]

    assert 7.0 < np.mean(high) < 13.0
    assert 7.0 < np.mean(low) < 13.0
    assert abs(np.mean(high) - np.mean(low)) < 2.0


def test_search_space_matches_documented_bounds():
    assert DIMENSION_NAMES == [
        "composition", "particle_size_nm", "metal_loading_wt_pct",
        "facet", "calcination_temp_C", "defect_density",
    ]
    assert SEARCH_SPACE[1].low == 20.0 and SEARCH_SPACE[1].high == 300.0
    assert SEARCH_SPACE[2].low == 0.1 and SEARCH_SPACE[2].high == 5.0
    assert SEARCH_SPACE[4].low == 200.0 and SEARCH_SPACE[4].high == 600.0


def test_random_candidate_respects_search_space():
    rng = np.random.default_rng(2)
    for _ in range(100):
        c = _sample_random_candidate(rng)
        assert c["composition"] in COMPOSITIONS
        assert 20.0 <= c["particle_size_nm"] <= 300.0
        assert 0.1 <= c["metal_loading_wt_pct"] <= 5.0
        assert c["facet"] in FACETS
        assert 200.0 <= c["calcination_temp_C"] <= 600.0
        assert c["defect_density"] in DEFECTS


def test_bo_loop_runs_and_produces_valid_rows():
    """
    Smoke test with a small budget -- not a convergence check (the 10-seed
    multiseed run covers that), just confirms the loop runs end to end and
    returns well-formed rows.
    """
    rows = run_bayesian_optimization(budget=12, seed=42, n_initial_points=10,
                                      out_csv="results/_test_bo_log.csv")
    assert len(rows) == 12
    for row in rows:
        assert row["composition"] in COMPOSITIONS
        assert 0.0 <= row["activity"] <= 100.0
    os.remove("results/_test_bo_log.csv")