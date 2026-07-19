"""
oracle.py

Simulates the "true" photocatalytic activity of a Cu2O-based catalyst as a
function of composition, particle size, metal loading, facet, calcination
temperature, and defect density.

This oracle stands in for an actual synthesis-and-test cycle. It is NOT a
fitted model of real experimental sweeps for most variables — real catalyst
science rarely has every variable independently swept in one paper. Each
contribution below is explicitly tagged with its evidence tier:

    STRONG        = directly measured in Addanki Tirumala et al.,
                     ACS Appl. Mater. Interfaces 2025, 10.1021/acsami.5c01996
                     (this repo author's own paper + SI)
    MODERATE      = real, cited literature value, but from a DIFFERENT
                     catalytic system (Au-Pd or Ag-Pd alloys, different
                     reaction), adapted here as a directional anchor only
    ILLUSTRATIVE  = general materials-science principle (e.g. volcano-shaped
                     loading/calcination trends, facet coordination effects),
                     no fitted curve, no literature sweep for THIS system

See DATA_GROUNDING.md for the full citation list and reasoning per variable.
"""

import numpy as np

# ---------------------------------------------------------------------------
# STRONG: composition base activity
# Source: Addanki Tirumala et al. 2025, Fig. 3 / Table 1 (main text + SI)
#   Cu2O-Pd + light  -> ~95% conversion at 4 h, ~100% by 6 h, ~100% selectivity
#   Cu2O alone + light -> "minimal" reaction progress even at 8 h
# MODERATE: Cu2O-Au / Cu2O-Ag are NOT measured in this paper. Values below
# are directional anchors adapted from analogous Pd-alloy photocatalytic
# coupling literature (different host oxide, different reaction):
#   Xiao et al., ACS Catal. 2014, 4, 1725  -> Au-Pd alloy, 37% thermal
#       conversion at Au:Pd = 1:1.86, improved further under light
#   PdAu nanosheets, ACS Appl. Nano Mater. 2022 -> 30%->98% (1h->3h), visible light
#   Ag-Pd alloy nanoparticle photocatalyst (Suzuki coupling) -> 96% conversion
#       under optimized conditions
# These numbers are for different reactions/hosts, so treat Au/Ag figures
# here as "plausible, literature-informed," not "measured for this system."
# ---------------------------------------------------------------------------
BASE_ACTIVITY = {
    "Cu2O": 8.0,        # STRONG (real, "minimal" conversion in paper)
    "Cu2O-Pd": 97.0,    # STRONG (real, ~100% by 6h)
    "Cu2O-Au": 45.0,    # MODERATE (adapted from Xiao et al. 2014 Au-Pd anchor)
    "Cu2O-Ag": 80.0,    # MODERATE (adapted from Ag-Pd alloy anchor, discounted
                         # for different host oxide)
}

# ---------------------------------------------------------------------------
# STRONG: particle size peak location
# The paper's ONLY tested particle size is 48 +/- 5 nm (Cu2O-Pd), and it
# performs very well. We anchor the assumed size-activity peak AT this real
# value. But no size sweep exists in the paper -- the fact that a peak
# (rather than a plateau or monotonic trend) exists at all, and its width,
# is ILLUSTRATIVE, not measured.
# ---------------------------------------------------------------------------
SIZE_PEAK_NM = 48.0
SIZE_WIDTH_NM = 60.0  # wide/forgiving -- reflects genuine uncertainty

# ---------------------------------------------------------------------------
# ILLUSTRATIVE: metal loading volcano
# No wt% Pd figure is reported in the paper (deliberately not fabricated).
# 2.0 wt% is a generic "typical" loading for supported noble-metal
# photocatalysts in the broader literature, used only as a plausible peak.
# ---------------------------------------------------------------------------
LOADING_PEAK_WT_PCT = 2.0
LOADING_WIDTH_WT_PCT = 1.2

# ---------------------------------------------------------------------------
# ILLUSTRATIVE: facet effect
# The paper DFT-tested (111) and (110) facets (no explicit relative activity
# numbers extracted). (100) is included because it appears in a paper the
# authors cite (Zhang et al., Nanoscale 2021, Pd/Cu2O{100} for CO2
# photoreduction) -- different reaction, not a direct activity comparison.
# Relative ordering below reflects a general, commonly-cited claim in Cu2O
# catalysis literature that under-coordinated (111) surface Cu sites tend to
# be more catalytically active than more close-packed facets -- a defensible
# general principle, NOT a fitted result for this reaction.
# ---------------------------------------------------------------------------
FACET_FACTOR = {"111": 1.00, "110": 0.85, "100": 0.75}

# ---------------------------------------------------------------------------
# ILLUSTRATIVE: calcination temperature volcano
# Classic materials-science trade-off (crystallinity/activity gain vs.
# sintering/surface-area loss at high temp). Peak location is a generic
# textbook estimate, not fit to any sweep in this system.
# ---------------------------------------------------------------------------
CALCINATION_PEAK_C = 350.0
CALCINATION_WIDTH_C = 150.0

# ---------------------------------------------------------------------------
# ILLUSTRATIVE: defect density
# Defects can supply extra active sites (helpful) or act as recombination
# centers (harmful) -- a well-known trade-off in photocatalysis broadly.
# Kept deliberately low-magnitude since there is no real anchor at all.
# ---------------------------------------------------------------------------
DEFECT_FACTOR = {"low": 0.92, "medium": 1.00, "high": 0.88}

# ---------------------------------------------------------------------------
# STRONG: temperature-rise confound
# Real, measured: reaction mixture warms ~10 C under illumination regardless
# of catalyst identity or activity (paper's dark-control rules out heating
# as the mechanism). Included as a decoy feature -- true relationship to
# activity is ZERO by design. Mirrors the confound pattern in ExplainableCatML.
# ---------------------------------------------------------------------------
TEMP_RISE_MEAN_C = 10.0
TEMP_RISE_STD_C = 1.0


def _gaussian_factor(x, peak, width):
    """Returns 1.0 at the peak, decaying smoothly away from it. Never zero."""
    return np.exp(-((x - peak) ** 2) / (2 * width ** 2))


def oracle(
    composition: str,
    particle_size_nm: float,
    metal_loading_wt_pct: float,
    facet: str,
    calcination_temp_C: float,
    defect_density: str,
    rng: np.random.Generator = None,
):
    """
    Returns a dict with the simulated activity (%) and the decoy temperature
    rise (C) for one candidate catalyst.

    composition: one of "Cu2O", "Cu2O-Pd", "Cu2O-Au", "Cu2O-Ag"
    particle_size_nm: float
    metal_loading_wt_pct: float (ignored if composition == "Cu2O")
    facet: one of "111", "110", "100"
    calcination_temp_C: float
    defect_density: one of "low", "medium", "high"
    """
    if rng is None:
        rng = np.random.default_rng()

    base = BASE_ACTIVITY[composition]

    if composition == "Cu2O":
        # No metal present -- loading has no meaning here.
        loading_factor = 1.0
    else:
        loading_factor = 0.5 + 0.5 * _gaussian_factor(
            metal_loading_wt_pct, LOADING_PEAK_WT_PCT, LOADING_WIDTH_WT_PCT
        )

    size_factor = 0.6 + 0.4 * _gaussian_factor(
        particle_size_nm, SIZE_PEAK_NM, SIZE_WIDTH_NM
    )
    facet_factor = FACET_FACTOR[facet]
    calc_factor = 0.6 + 0.4 * _gaussian_factor(
        calcination_temp_C, CALCINATION_PEAK_C, CALCINATION_WIDTH_C
    )
    defect_factor = DEFECT_FACTOR[defect_density]

    activity = base * loading_factor * size_factor * facet_factor * calc_factor * defect_factor

    # Realistic experimental noise (~3% std, consistent with typical
    # replicate variability reported in photocatalysis conversion studies).
    activity += rng.normal(0, 3.0)
    activity = float(np.clip(activity, 1.0, 100.0))

    # Decoy variable -- independent of every input above, by design.
    temp_rise_C = float(rng.normal(TEMP_RISE_MEAN_C, TEMP_RISE_STD_C))

    return {"Activity_pct": activity, "Temp_Rise_C": temp_rise_C}