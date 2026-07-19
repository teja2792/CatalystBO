# CatalystBO

**Closed-loop Bayesian optimization for photocatalyst discovery.** Given a
mixed categorical/continuous design space (composition, particle size,
metal loading, facet, calcination temperature, defect density), this repo
doesn't predict the activity of catalysts you already made — it recommends
the *next* catalyst to synthesize, using a Gaussian Process surrogate and
Expected Improvement to decide where testing is most likely to help.

> Imagine you've synthesized Cu2O, Cu2O-Pd, Cu2O-Au, Cu2O-Ag catalysts.
> Instead of guessing candidate #301, the model says: **Cu2O-Pd, 20 nm,
> 2.07 wt% Pd, facet (111), 359°C calcination, medium defect density →
> 98.3% predicted activity.** That's the closed loop this repo builds.

Part of a three-repo portfolio demonstrating ML applied to catalysis
research: [`CatalystML`](https://github.com/teja2792/CatalystML) (property
prediction), [`ExplainableCatML`](https://github.com/teja2792/ExplainableCatML)
(cross-method explainability), and this repo (sequential experimental
design). This one is the only one of the three that decides what to test
next rather than explaining a fixed dataset.

---

## Highlights

- Bayesian optimization cut mean regret by ~91% vs. random search at an
  identical 50-evaluation budget, validated across 10 seeds (33.9 ± 7.6
  vs. 3.2 ± 6.0), not a single favorable run.
- It beat a 4,500-evaluation grid search using 90x fewer evaluations.
- Every oracle relationship is explicitly tagged by evidence tier with
  citations — real data from my own published work where available,
  literature-adapted anchors where not, and clearly-labeled illustrative
  assumptions where no grounding exists.
- Real limitations of the GP+EI approach (boundary-seeking, over-
  exploitation, epistemic-vs-aleatoric uncertainty) were diagnosed from
  the actual run logs and reported rather than hidden.
- SHAP applied to a purpose-built surrogate — not the GP directly —
  correctly recovers the oracle's dominant designed effect, cross-
  validated with a held-out R².

---

## Technical Skills Demonstrated (For Interviewers)

**Bayesian optimization / sequential experimental design** — Gaussian
Process surrogate modeling, Expected Improvement acquisition, mixed
categorical/continuous search spaces via `scikit-optimize`, budget-matched
benchmarking against random and grid-search baselines, regret/convergence
curve analysis, and multi-seed statistical validation rather than
single-run claims.

**Data engineering & literature grounding** — extracted real quantitative
data from a peer-reviewed paper's PDF/SI (`pdftotext` + targeted parsing),
performed a systematic literature search for analogous-system anchors when
direct data wasn't available, and enforced an explicit STRONG/MODERATE/
ILLUSTRATIVE tagging discipline throughout rather than fabricating numbers.

**Explainability applied to an optimizer, not just a predictor** — SHAP
`TreeExplainer` on a deliberately-chosen standalone surrogate (with an
explicit justification for why the GP itself isn't a good SHAP target),
validated with a held-out test R² rather than a training-only score.

**Critical statistical evaluation** — diagnosed GP-EI's boundary-seeking
and over-exploitation behavior directly from raw iteration logs, correctly
explained negative regret as a noise artifact against a noiseless
reference, and distinguished epistemic from aleatoric uncertainty rather
than overselling a GP confidence interval.

**Software engineering** — modular design separating the oracle, search
space, and optimization strategy; a pytest suite spanning statistical
regression tests, structural/bounds checks, and an end-to-end smoke test;
seeded reproducibility throughout; standard git workflow discipline.

---

## Advantages

- **Statistically validated, not a lucky single run.** The headline
  91% regret reduction holds across 10 independent seeds with a large
  effect size relative to the variance, not one favorable trace.
- **Handles a genuinely mixed search space** (4 composition classes, 2
  continuous variables with different units, 2 more categoricals) in one
  unified GP + EI framework.
- **Full uncertainty quantification**, not just point predictions — every
  recommendation carries a GP posterior estimate.
- **Explainability extends to the optimizer's own evidence**, not just a
  static prediction model, and is cross-validated with a held-out R²
  rather than reported on training data alone.
- **Every relationship is citation-tagged** (STRONG/MODERATE/ILLUSTRATIVE)
  — nothing is presented as measured when it's a literature-adapted or
  generic assumption.
- **Modular and reproducible**: oracle, search space, and optimization
  strategy are cleanly separated; every run is seeded; 6/6 tests passing.

## Disadvantages / Limitations

- **This is a simulation, not real experimental validation.** The entire
  "91% improvement" is measured against a synthetic oracle I designed —
  not a real synthesis-and-test loop. Treat this as a methodology
  demonstration, not a materials discovery claim.
- **Facet, calcination temperature, defect density, and the Au/Ag
  composition scaling are illustrative or adapted from different systems.**
  Real experiments could show meaningfully different relationships.
- **GP + Expected Improvement showed real, documented failure modes even
  within the simulation**: particle size converged to the search space's
  lower bound (20 nm) rather than the oracle's designed 48 nm peak — a
  combination of a deliberately weak/wide signal there and a known
  boundary-seeking artifact in GP-based EI. In ~30% of the 10 seeds, the
  optimizer proposed an exact duplicate of a previously-evaluated point,
  caught by skopt's own duplicate check — evidence of late-stage
  over-exploitation, not a bug.
- **Reported prediction uncertainty (±0.6%) is epistemic, not a real
  measurement interval.** It reflects the GP's confidence about the mean
  function value at a well-sampled point, not the ~3% spread you'd expect
  from a single new noisy measurement. Conflating the two would overstate
  precision by roughly 5x.
- **Code duplication**: the BO loop is implemented twice (`bo_search.py`
  and `run_multiseed_comparison.py`) rather than factored into a shared
  module — acceptable at this prototype scale, would consolidate before
  production use.
- **One-hot encoding all composition categories (no dropped baseline)**
  introduces mild collinearity in the SHAP surrogate's individual dummy
  attributions. Only the dominant Cu2O-Pd effect should be taken at face
  value; the smaller categories' exact SHAP signs are less reliable.
- **50-evaluation budget on a 6-dimensional space** is a proof-of-concept
  scale — some higher-order interactions are likely under-sampled.

---

## Results

### Sample efficiency (10 seeds, budget = 50 evaluations each)

| Method | Final regret (mean ± std) | Evaluations |
|---|---|---|
| Random search | 33.9 ± 7.6 | 50 |
| Bayesian optimization | **3.2 ± 6.0** | 50 |
| Grid search (context only, not budget-matched) | 11.9 (single run) | 4,500 |

BO reduced mean regret by ~91% relative to random search at equal budget,
and beat a grid search using 90x fewer evaluations.

![Regret curves](results/regret_curves.png)

### What the optimizer's evidence says matters

A RandomForest trained on all 500 (candidate, activity) pairs BO evaluated
across 10 seeds (test R² = 0.971, held-out from a 400/100 split) confirms
composition dominates, consistent with it being the most strongly-grounded
effect in the oracle:

![SHAP bar](results/surrogate_shap_bar.png)
![SHAP beeswarm](results/surrogate_shap_beeswarm.png)

### Demo: recommend the next candidate
BEST CANDIDATE EVALUATED DURING OPTIMIZATION (50 syntheses tested)
composition: Cu2O-Pd | particle_size_nm: 20.0 | metal_loading_wt_pct: 2.07
facet: 111 | calcination_temp_C: 359.3 | defect_density: medium
-> measured activity: 98.3%
NEXT CANDIDATE RECOMMENDED BY THE OPTIMIZER (not yet synthesized)
composition: Cu2O-Pd | particle_size_nm: 20.0 | metal_loading_wt_pct: 1.99
facet: 111 | calcination_temp_C: 353.0 | defect_density: medium
-> predicted activity: 92.7% +/- 0.6% (GP posterior belief, unverified)
---

## Why this is a different ML contribution, not a rediscovery

`CatalystML` and `ExplainableCatML` both fit models to a fixed, already-
collected dataset. This repo instead treats catalyst discovery as a
**sequential decision problem under uncertainty**: at each step, the
optimizer chooses which candidate to "synthesize" next based on everything
learned so far, and is scored on how few evaluations it needs to find a
near-optimal catalyst — not on prediction accuracy alone.

---

## Data grounding — what's real, what's illustrative

Every oracle relationship is tagged by evidence tier. Full citations and
reasoning are in [`DATA_GROUNDING.md`](DATA_GROUNDING.md); summary below.

| Variable | Tier | Basis |
|---|---|---|
| Composition (Cu2O vs. Cu2O-Pd) | **STRONG** | Addanki Tirumala et al., *ACS Appl. Mater. Interfaces* 2025 (10.1021/acsami.5c01996). Cu2O-Pd reaches ~100% conversion by 6h; Cu2O alone shows minimal conversion under identical conditions. |
| Composition (Cu2O-Au, Cu2O-Ag) | MODERATE | Not measured in the source paper. Adapted from Xiao et al., *ACS Catal.* 2014 (Au-Pd alloy, 37% thermal conversion); a 2022 *ACS Appl. Nano Mater.* PdAu nanosheet study (30%→98% under visible light); and an Ag-Pd alloy Suzuki-coupling photocatalyst study (96% conversion). Different host oxide and reaction — directional anchors only. |
| Particle size peak (48 nm) | **STRONG** (peak location only) | The only particle size tested in the source paper, performs well. Peak *shape*/width is illustrative — no size sweep exists in this system. |
| Metal loading | ILLUSTRATIVE | No wt% figure reported in the source paper. Peak at 2.0 wt% is a generic literature-typical value. |
| Facet | ILLUSTRATIVE | Source paper DFT-tested (111)/(110) without explicit relative activity numbers. (100) included via a cited but different-reaction paper (Zhang et al., *Nanoscale* 2021). |
| Calcination temperature | ILLUSTRATIVE | Generic materials-science volcano trade-off (crystallinity vs. sintering). No sweep in this system. |
| Defect density | ILLUSTRATIVE | General photocatalysis trade-off, deliberately kept low-magnitude given no real anchor. |
| Temperature-rise confound (~10°C) | **STRONG** | Measured in source paper SI. Reaction mixture warms ~10°C under light regardless of catalyst; dark control rules out heating as the mechanism. Included as a decoy variable with zero true effect on activity. |

**No number in this repo is presented as a real measurement unless tagged STRONG.**

---

## Method

1. **Oracle** (`src/oracle.py`) — simulates "true" catalyst activity as a
   product of independent per-variable factors (composition base rate ×
   size × loading × facet × calcination × defect), each tagged by evidence
   tier above, plus realistic ±3% observation noise and the temperature
   decoy.
2. **Search space** (`src/search_space.py`) — mixed categorical/continuous
   space via `skopt.space`.
3. **Baselines** (`src/baseline_search.py`) — random search (budget-matched
   to BO) and a full 4,500-point grid search (reported separately, since
   it's not budget-matched).
4. **Bayesian optimization** (`src/bo_search.py`) — `skopt.Optimizer` with
   a Gaussian Process surrogate and Expected Improvement acquisition,
   10 random-init points followed by GP-guided selection.
5. **Multi-seed robustness** (`src/run_multiseed_comparison.py`) — repeats
   random search and BO across 10 seeds. A single BO run can get lucky;
   this is the number that's actually defensible.
6. **Surrogate explanation** (`src/explain_surrogate.py`) — SHAP on a
   standalone RandomForest trained on everything BO evaluated (not the GP
   itself — see Limitations).
7. **Recommendation demo** (`src/recommend_next_candidate.py`) — the actual
   "what to synthesize next" output.
8. **Tests** (`tests/test_pipeline.py`) — 6 tests covering oracle grounding,
   the analytical optimum, the decoy variable, search-space bounds, sampler
   validity, and an end-to-end BO smoke test. All passing.

---

## How to run
pip install -r requirements.txt
python scripts/test_oracle.py                    # Phase 1: sanity-check the oracle
python src/baseline_search.py                     # Phase 2: random + grid baselines
python src/bo_search.py                            # Phase 3: single BO run
python src/run_multiseed_comparison.py             # Phase 4: 10-seed robustness
python src/plot_regret_curves.py                   # Phase 4: regret curve plot
python src/explain_surrogate.py                    # Phase 5: SHAP on surrogate
python src/recommend_next_candidate.py              # Phase 6: recommendation demo
pytest tests/test_pipeline.py -v                    # Phase 7: test suite
## Repository structure
CatalystBO/
├── src/
│   ├── oracle.py                    # grounded simulation of catalyst activity
│   ├── search_space.py              # mixed categorical/continuous space
│   ├── baseline_search.py           # random + grid search baselines
│   ├── bo_search.py                 # single Bayesian optimization run
│   ├── run_multiseed_comparison.py  # 10-seed robustness comparison
│   ├── plot_regret_curves.py        # convergence plot
│   ├── explain_surrogate.py         # SHAP on post-hoc surrogate
│   └── recommend_next_candidate.py  # closed-loop recommendation demo
├── scripts/
│   └── test_oracle.py               # Phase 1 sanity check
├── tests/
│   └── test_pipeline.py             # 6 tests: statistical + structural + smoke
├── results/                         # logs, CSVs, plots
├── DATA_GROUNDING.md                # full citation list per variable
└── README.md
## References

- Addanki Tirumala et al., *ACS Appl. Mater. Interfaces* 2025, 10.1021/acsami.5c01996 (source paper for the Cu2O-Pd system this oracle is grounded in)
- Xiao, Q. et al., *ACS Catal.* 2014, 4, 1725–1734 (Au-Pd alloy anchor)
- Ag-Pd alloy nanoparticle photocatalyst, Suzuki-Miyaura coupling (Ag-Pd anchor)
- Zhang et al., *Nanoscale* 2021 (Cu2O{100} facet context)