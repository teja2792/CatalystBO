# Data Grounding

Every variable in this repo's oracle function is tagged by evidence tier. No
number here is presented as a real measurement unless it is explicitly
marked STRONG.

| Variable | Tier | Basis |
|---|---|---|
| Composition (Cu2O vs Cu2O-Pd) | **STRONG** | Addanki Tirumala et al., ACS Appl. Mater. Interfaces 2025 (10.1021/acsami.5c01996). Cu2O-Pd reaches ~100% conversion by 6h; Cu2O alone shows minimal conversion under identical conditions. |
| Composition (Cu2O-Au, Cu2O-Ag) | MODERATE | Not measured in the source paper. Adapted from Xiao et al., ACS Catal. 2014, 4, 1725 (Au-Pd alloy, 37% thermal conversion at Au:Pd 1:1.86); a 2022 ACS Appl. Nano Mater. PdAu nanosheet study (30%->98%, visible light); and a Ag-Pd alloy Suzuki-coupling photocatalyst study (96% conversion). Different host oxide and reaction -- directional anchors only. |
| Particle size peak (48 nm) | **STRONG** (peak location only) | Only particle size tested in the source paper (48+/-5 nm), performs well. Peak *shape*/width is illustrative -- no size sweep exists in this system. |
| Metal loading | ILLUSTRATIVE | No wt% figure reported in the source paper. Peak at 2.0 wt% is a generic literature-typical value, not fit to data. |
| Facet | ILLUSTRATIVE | Source paper DFT-tested (111)/(110) without explicit relative activity numbers. (100) included via Zhang et al., Nanoscale 2021 (different reaction: CO2 photoreduction). Relative ordering reflects general Cu2O coordination-chemistry arguments, not a fitted result. |
| Calcination temperature | ILLUSTRATIVE | Generic materials-science volcano trade-off (crystallinity vs. sintering). No sweep in this system. |
| Defect density | ILLUSTRATIVE | General photocatalysis trade-off (extra active sites vs. recombination centers). No real anchor. Effect size kept deliberately small. |
| Temperature-rise confound (~10 C) | **STRONG** | Measured in source paper SI. Reaction mixture warms ~10C under light regardless of catalyst; dark control at same temperature shows no reaction, ruling out heating as the mechanism. Included here as a decoy variable with zero true effect on activity. |