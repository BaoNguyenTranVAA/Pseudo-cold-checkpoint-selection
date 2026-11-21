# Pseudo-cold validation leakage and checkpoint-selection fidelity: reproducibility package

This repository accompanies the article *Checkpoint-Selection Failure Under Pseudo-Cold Validation Leakage: Matched Controls and Sealed-Holdout Evidence*, prepared for submission to IEEE Access. It contains the exact analysis code, the locked result tables, and the study protocol. With them, the confirmatory results (P1, P2) and the exploratory TEST tables in the paper can be recomputed and checked against the locked values.

**Authors**

- Bao Nguyen Tran, Vietnam Aviation Academy, Vietnam
- Luong Minh Tri Nguyen, The University of Queensland, Australia
- Huu Hiep Nguyen, Vietnam Aviation Academy, Vietnam
- Loc Tran, Vietnam Aviation Academy, Vietnam

> **What this package does and does not reproduce.** It reproduces the reported analysis statistics covered by the released derived tables (per-trajectory regrets, split-level effects and per-state cold-TEST scores), as detailed below, and it includes the plotted data of Figure 3 with their provenance. The pretraining simulation of Appendix B (Table B-I) and the values quoted from the earlier conference paper are outside its scope. It does **not** re-run data construction, model training, or TEST scoring: the raw review data, the split archives, the sealed cold-TEST arrays and about 35 GB of checkpoints are not distributed here (see *Not included*).

## Quick check (about one minute, CPU only)

```bash
pip install "numpy>=1.26" "pandas>=2.2" "scipy>=1.11"
python scripts/reproduce_confirmatory.py
```

The script:
1. checks the SHA-256 of the released inputs;
2. pairs A3 with A0 (P1) and A3 with A5 (P2) on (domain, data split, training seed);
3. averages the five seeds into 16 split effects;
4. runs the exact two-sided sign-flip test over all 2^16 sign assignments.

It then compares everything with the locked files and prints `ALL MATCH`. Expected values:

| Set | Contrast | T | p (exact sign-flip) | Positive domains |
|---|---|---|---|---|
| DEV | P1 (A3 − A0) | 0.003766 | 0.00140 | 4/4 |
| DEV | P2 (A3 − A5) | 0.003364 | 0.00989 | 4/4 |
| cold-TEST | P1 | 0.008477 | 0.00058 | 4/4 |
| cold-TEST | P2 | 0.008194 | 0.00269 | 4/4 |

DEV uses the full regret at rule R1 and horizon 120. Cold-TEST uses the grid-restricted regret, which is a lower bound on each arm's full regret. A difference of grid-restricted regrets is not a lower bound on the difference of full regrets.

To regenerate the exploratory TEST tables (21 precommitted contrasts with Holm adjustment, the arm table, and the selected-loss decomposition):
```bash
python exploratory_test/explore_test.py test_results_locked dev_results_locked out
```
The script checks its input hashes. `out/exploratory_family.csv`, `out/arm_table_test.csv` and `out/decomposition.csv` are byte-identical to the files in `exploratory_test/out/`. Tested with Python 3.11 (numpy 1.26, pandas 2.2) and Python 3.13 (numpy 2.x, pandas 3.0).

## Checking the numbers in the manuscript

```bash
python scripts/verify_manuscript_numbers.py
```

The script makes 312 checks of the manuscript against the files of this repository (values, counts, and the row order of Table C-I), cross-checks the locked result fields, checks the Figure 3 data, and prints `RESULT: ALL MATCH`. The exit code is 0 only if everything matches and 1 otherwise. It writes two reports into the current folder: `manuscript_check.csv` and `consistency_check.csv`.

What it does and does not check:
- **The manuscript values are transcribed by hand into the script.** They are not read from the PDF, and a value that is not transcribed is not checked. The script does not cover Table B-I (simulation) or the values quoted from earlier work; for figures, only the Figure 3 data are checked (Part 3).
- **Each computed value is rounded to the decimals printed in the manuscript.** It is then compared as text.
- **Column `basis` of `manuscript_check.csv` says where each value comes from:**
  - `recomputed` (117 checks): computed from the per-trajectory, per-state or per-unit rows.
  - `locked-field` (189 checks): read from a field of `results.json`, `test_results.json` or `exploratory_family.csv`.
  - `ratio of locked fields` / `arithmetic` (6 checks).
- **Part 2 cross-checks the locked fields (427 checks, tolerance 1e-12).** Every locked contrast field that has unit rows is recomputed from those rows:
  - T, the mean of the 16 split effects;
  - the exact sign-flip p, by enumeration and with `code/stats_confirm.py`;
  - the 95% CI by test inversion;
  - positive splits and domains;
  - Holm over the 21 exploratory contrasts.
  
  The two rule-interaction contrasts are rebuilt from `per_trajectory.csv`.
- **Part 3 checks the Figure 3 data in `figure_data/`** (10 checks; see *Figure 3 data* below).

The script needs numpy, pandas and scipy. It was tested with Python 3.11 (numpy 1.26, pandas 2.2) and Python 3.13 (numpy 2.x, pandas 3.0), with identical results.

## Figure 3 data

`figure_data/FIG3_curves.csv` holds the values plotted in Figure 3. There is one row per grid arm (A0, A3, A5, A7) and saved grid epoch (30 epochs), with the mean over the 80 trajectories of:
- the pseudo-validation NDCG@10 (`mean_pv_ndcg10`, panel a);
- the cold-TEST NDCG@10 (`mean_test_ndcg10`, panel b).

The figure draws A0, A3 and A5; A7 is included for completeness. The vertical lines of panel (b) are the A3 medians in `FIG3_trajectory_medians.json`.

Provenance:
- **Script and run.** `figure_data/fig3_curves.py` produced the files on the original workstation on 20 Nov 2026, after the cold-TEST lock. It is a post-lock descriptive step: the curves are computed from the archived per-state scores, no checkpoint was loaded and nothing was re-scored. The manifest documents the provenance.
- **Manifest.** `FIG3_MANIFEST.json` records the SHA-256 of the script, the inputs and the outputs, and the software environment.
- **Panel (b).** Recomputed from the released `test_scores.csv`; `verify_manuscript_numbers.py` (Part 3) checks it.
- **Panel (a).** Comes from the per-state PV scores of the DEV-only dry run (`dryrun_scores.csv`, SHA-256 `4130f20e…`, recorded in the audit trail before the cold-TEST was opened). That file is not released, so panel (a) can be traced through the manifest hash but not recomputed from this repository.
- **Checks before the run.** The script checks that the 9,600 grid states have the same checkpoint identity (state and weight hashes) in the dry-run and cold-TEST score files.
- **Match with the printed figure.** Both panels were checked against the printed figure by overlaying the values on the published raster. The A3 markers lie within 1 px of their values (about 0.0001 NDCG@10). The overlapping A0/A5 markers lie within the 2-px tolerance of the check.

## Contents

| Path | Role |
|---|---|
| `scripts/reproduce_confirmatory.py` | Stand-alone recomputation of P1/P2 (DEV and cold-TEST, rules R1 and R2) from the released tables |
| `scripts/verify_manuscript_numbers.py` | Check of the manuscript's quantitative values against this repository (see above) |
| `figure_data/FIG3_curves.csv`, `FIG3_trajectory_medians.json`, `FIG3_MANIFEST.json`, `fig3_curves.py` | Plotted data of Figure 3, with the script and manifest that produced them (see *Figure 3 data*) |
| `code/analysis_dev.py` | DEV analysis as locked and run (confirmatory P1/P2, secondary contrasts, descriptives) |
| `code/analysis_test.py` | Sealed cold-TEST analysis as frozen before TEST was opened |
| `code/metrics.py` | NDCG@10 and Recall@10 on the cold holdout catalog |
| `code/test_schema.py` | Schema and integrity checks for the TEST score files and receipts |
| `code/stats_confirm.py` | Exact sign-flip test, decision rule, and confidence interval by test inversion |
| `code/selection.py` | Checkpoint selection rules R1 (argmax) and R2 (trailing 5-epoch mean) and regret |
| `code/common.py`, `code/constants.py`, `code/trajectory_schema.py` | Modules imported by the analysis scripts (configuration, arms, schema) |
| `dev_results_locked/units.csv` | 16 split effects per DEV contrast |
| `dev_results_locked/per_trajectory.csv` | Per-trajectory DEV selection outcomes (1,200 trajectories × rules × horizons) |
| `dev_results_locked/results.json` | Locked DEV results |
| `test_results_locked/test_units.csv` | 16 split effects per cold-TEST contrast |
| `test_results_locked/test_per_trajectory.csv` | Per-trajectory cold-TEST outcomes |
| `test_results_locked/test_scores.csv` | Cold-TEST NDCG@10 / Recall@10 of all 12,000 scored checkpoints, with checkpoint hashes |
| `test_results_locked/test_results.json` | Locked cold-TEST results |
| `exploratory_test/explore_test.py`, `exploratory_test/out/*.csv` | Post-lock exploratory TEST tables and the script that produced them |
| `protocol/PROTOCOL.md` | Study protocol V1.3, locked 28 Oct 2026 before training |
| `protocol/DEVIATIONS.md` | Ledger of pre-lock amendments (part A), data-stage notes (B1–B4) and post-lock deviations made before training (C1–C9) |
| `post_lock/TEST_ANALYSIS_PLAN.md` | Cold-TEST analysis plan, frozen before TEST was opened |
| `LICENSE`, `LICENSE-DATA.md` | MIT License for the code; CC BY 4.0 for the result tables and documents (see *License*) |
| `SHA256SUMS` | Hashes of every file in this package |

`protocol/` and `post_lock/` are working documents written in Vietnamese. They are released unchanged so that their hashes match the audit trail.

## Design in brief

- **Data:** four Amazon review domains (Video Games, Clothing/Shoes/Jewelry, Sports and Outdoors, Toys and Games). Each domain has a fixed core and four item-cold splits; per split there are 700 cold-DEV items and 300 sealed cold-TEST items.
- **Training:** 15 arms × 5 training seeds per split, 1,200 trajectories of 120 epochs.
- **Arms of the confirmatory contrasts:**
  - A0: clean;
  - A3: pseudo-validation (PV) pairs inserted at q ≈ 0.335 per batch;
  - A5: replacement-matched clean control.
- **Confirmatory test:** fixed sequence P1 → P2, α = 0.05. The statistic is the mean of 16 split effects. The test is the exact sign-flip, with at least 3 of 4 domains required to be positive.
- **Inference scope:** inference is conditional on the four fixed domain cores. Splits use independent, domain-seeded random streams, and every (domain, split, seed) combination has its own training seed. Generalization to unseen domains is not tested (see the paper's Appendix B).

## Timeline of locks

| Event (UTC) | Record |
|---|---|
| 28 Oct 2026 | Protocol V1.3 locked (`PROTOCOL.md` SHA-256 `c5c2913b…`) |
| 30 Oct 2026 04:40:51 | DEV results locked (`DEV_RESULTS_LOCK.json` `0775d9fb…`) |
| 30 Oct 2026 10:19:26 | TEST analysis plan frozen (`TEST_PLAN_FREEZE.json` `841e3c6a…`) |
| 30 Oct 2026 16:54:40 | Cold-TEST opened once (opening id `OPEN-TEST-20261030T165440Z`, no resume) |
| 30 Oct 2026 17:07:44 | Cold-TEST results locked (`05_LOCKED.json` `5d868299…`) |

The cold-TEST plan was frozen after the DEV results were known, so the TEST analysis is a *precommitted sealed-TEST replication informed by DEV*.

Note on `DEVIATIONS.md`: the DEV lock pins an earlier revision of this file (SHA-256 `e6ad3dd4…`). The released revision (`cdc09956…`) additionally records correction B4, a guard fix made before training. The full audit trail (receipts, audit log, reviewer verdicts) is available from the corresponding author on request.

## Running the original pipeline

The scripts in `code/` are the exact versions used; their SHA-256 values are listed in `SHA256SUMS` and match the locked records. They are released for inspection. To run end to end they need the full project tree:
- the data cores and splits;
- the trajectory logs in `runs/`;
- for the TEST scorer, the checkpoints and the sealed cold-TEST arrays.

The project root is set with the environment variable `JV_ROOT`. The defaults in `code/constants.py` point to the original Windows workstation. The remaining pipeline (data construction, sampler, trainer, guarded TEST opener) is available from the corresponding author on request.

## Not included

- The raw Amazon review data and item metadata. They are not redistributed; see the paper for the source and preprocessing.
- The split archives (`core.npz`, `dev.npz`) and the sealed cold-TEST arrays.
- The model checkpoints (about 35 GB) and the per-epoch trajectory logs.

## Citation

Citation information will be added once the article is published.

## License

- **Code:** `code/`, `scripts/`, `exploratory_test/explore_test.py` and `figure_data/fig3_curves.py` are released under the MIT License (`LICENSE`).
- **Result tables and documents:** the tables in `dev_results_locked/`, `test_results_locked/`, `exploratory_test/out/` and `figure_data/`, and the documents in `protocol/` and `post_lock/`, are released under CC BY 4.0 (`LICENSE-DATA.md`).
- **Not covered:** the Amazon Reviews 2023 data and metadata. They are not part of this repository, and these licenses do not extend to them.

The locked files carry no license header, so that their hashes stay identical to the audit trail; the license applies through these files and this section.

## Contact

Corresponding author: Bao Nguyen Tran, Vietnam Aviation Academy, Vietnam.
