import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from oracle import oracle

rng = np.random.default_rng(42)

candidates = [
    dict(composition="Cu2O", particle_size_nm=48, metal_loading_wt_pct=0,
         facet="111", calcination_temp_C=350, defect_density="medium"),
    dict(composition="Cu2O-Pd", particle_size_nm=48, metal_loading_wt_pct=2.0,
         facet="111", calcination_temp_C=350, defect_density="medium"),
    dict(composition="Cu2O-Au", particle_size_nm=48, metal_loading_wt_pct=2.0,
         facet="111", calcination_temp_C=350, defect_density="medium"),
    dict(composition="Cu2O-Ag", particle_size_nm=48, metal_loading_wt_pct=2.0,
         facet="111", calcination_temp_C=350, defect_density="medium"),
    dict(composition="Cu2O-Pd", particle_size_nm=150, metal_loading_wt_pct=4.5,
         facet="100", calcination_temp_C=550, defect_density="high"),
]

for c in candidates:
    result = oracle(**c, rng=rng)
    print(c["composition"], "| size:", c["particle_size_nm"], "nm |",
          "loading:", c["metal_loading_wt_pct"], "wt% |", "facet:", c["facet"],
          "->", round(result["Activity_pct"], 1), "% activity |",
          "temp rise:", round(result["Temp_Rise_C"], 1), "C")