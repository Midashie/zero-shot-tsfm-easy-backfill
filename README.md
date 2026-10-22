# Zero-shot time series foundation models for EASY-backfilling walltime estimates

Code and aggregate results for the manuscript "Zero-shot time series foundation models do not improve walltime estimates for EASY backfilling" (pre-registered negative result).

## What is in this repository

| Folder | Content |
| --- | --- |
| `taskb/` | F-DATA scheduling contexts, forecasts and recorded-time replay |
| `pm100/` | PM100 contexts, forecasts and replay |
| `theta/` | Theta (ALCF) context building and node-size selection (70% utilization rule) |
| `rt/` | Replay-timeline contexts and replay |
| `online/` | Exact online replay, SafeWall-style baseline, kill-and-requeue mode, analysis scripts and run queues |
| `g3/` | Completion-order classical estimators (Tsafrir last-2, history q90) used by the recorded-time replay |
| `manifest/` | Input manifest and the F-DATA replay job list builder |
| `paper/build_tables.py` | Builds Tables 5 to 7 from `results/online` and asserts the headline counts |
| `results/` | Aggregate result JSON files behind every table (no per-job data) |

## What is not in this repository

- Raw traces. Download them from the sources below.
- Per-job outputs (contexts, forecasts, replay logs). Rebuild them with the scripts.
- The CENTILE code. The replay imports it. Clone it separately and check out commit `c237f821fb119c01a0055d550d804f8a54e6a2b4` (clone origin: https://github.com/ZzZTripleZzZ/all-in-one). Pass the clone path as the first argument of `online_sim.py` and `build_rt.py`.

## Data sources

- F-DATA (file 21_04): https://doi.org/10.5281/zenodo.11467483
- PM100: https://doi.org/10.5281/zenodo.10127767 (CC BY 4.0)
- ALCF Theta job data, 2017 to 2020: https://reports.alcf.anl.gov/data/ANL-ALCF-DJC-THETA_20170701_20171231.html (and the 2018, 2019 and 2020 pages)

This data was generated from resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357.

## Known limitations of this release

1. **Paths.** The scripts read three environment variables and fall back to folders next to the code: `DATA_DIR` (raw traces, default `./data`), `OUT_DIR` (outputs, default `./outputs`) and `CENTILE_DIR` (the CENTILE clone, default `./centile`). Inside `DATA_DIR` the loaders expect `ZeroShotSchedule/21_04.parquet` (F-DATA), `ZeroShotScheduleLocal/job_table.parquet` (PM100), `ZeroShotSchedule/derived/cesnet_hourly.npz` (only `manifest/build_manifest.py` reads it, to hash it and to build CESNET window lists that the paper does not use), the model folders `ZeroShotSchedule/moirai-2.0` and `ZeroShotSchedule/timesfm-2.5`, and `ANL-ALCF-DJC-THETA_<start>_<end>.csv` for each Theta year. The scripts apart from these path edits are the code that produced the paper's numbers. After the path edit I checked that every file compiles, and that `paper/build_tables.py` reproduces the headline counts from `results/`. I have not rerun the replays from the edited copy.
2. **Separate Python environments.** Set `PY_M2`, `PY_TTM` and `PY_TF` to the Python interpreters of three environments, one per model library (Moirai-2, TTM, TimesFM). They default to `python3`.
3. **Tested versions.** Python 3.13, numpy 2.5.3, pandas 3.0.5, pyarrow 26.0.0, scipy 1.18.1, torch 2.14.1, xgboost 3.4.1. Model library versions in the three model environments are not recorded here.
4. **Run order for the Theta experiments.** `online/queue_theta.sh`, `queue_theta_years.sh`, `queue_swb.sh`, `queue_enf.sh`, `queue_q80.sh` and `queue_rep.sh`. Each queue skips finished steps. `build_tables.py` fails its assertions if the counts differ from the paper.
5. **Run time.** The Chronos-Bolt replays dominate the cost. The queues ran for many hours.

The CENTILE loader that `g3/g3_fdata.py` calls reads `~/nfm/data/fdata/*.parquet`, so put or link the F-DATA file there.

## Citation

Add the final citation here after acceptance.
