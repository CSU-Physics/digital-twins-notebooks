# Digital Twins for Semiconductor Manufacturing: notebooks and media (SemiAcademy)

Lab notebooks, slides and interactive explorers for the SemiAcademy micro-credential *Digital Twins for Semiconductor Manufacturing* (Canvas course 57). The labs run **inside Canvas, in the learner's own browser**: GitHub Pages serves a JupyterLite site, and each Canvas lab page embeds one notebook. Python runs in the browser through Pyodide, so learners install nothing and need no account. Each lab also has an optional Google Colab copy.

Site: https://csu-physics.github.io/digital-twins-notebooks/

Every lab works with one **simulated** thermal oxidation furnace; Module 2 also uses one real, public data set, SECOM (see [Data](#data)). Its model, `furnace.py`, sits next to each notebook and grows module by module. Every value it produces is simulated, and each learner's KEY (a whole number from 100 to 999, given by Question 1 of the lab check) selects that learner's own data.

## Notebooks

| Module | Notebook | Canvas page (course 57) | Lab address |
|---|---|---|---|
| 0 | `content/m0/DT-M0-lab-furnace-log.ipynb` | Module 0 Lab: Notebooks, NumPy and pandas on a Furnace Log | `lab/index.html?path=m0/DT-M0-lab-furnace-log.ipynb` |
| 1 | `content/m1/DT-M1-lab-first-twin.ipynb` | Module 1 Lab: A First Twin of the Furnace | `lab/index.html?path=m1/DT-M1-lab-first-twin.ipynb` |
| 2 | `content/m2/DT-M2-lab-fab-data.ipynb` | Module 2 Lab: Screening Fab Data and Joining Records | `lab/index.html?path=m2/DT-M2-lab-fab-data.ipynb` |
| 3 | `content/m3/DT-M3-lab-oxidation-model.ipynb` | Module 3 Lab: Calibrating a Deal-Grove Oxidation Model | `lab/index.html?path=m3/DT-M3-lab-oxidation-model.ipynb` |
| 4 | `content/m4/DT-M4-lab-virtual-metrology.ipynb` | Module 4 Lab: Building and Testing a Virtual Metrology Model | `lab/index.html?path=m4/DT-M4-lab-virtual-metrology.ipynb` |
| 5 | `content/m5/DT-M5-lab-run-to-run.ipynb` | Module 5 Lab: Run-to-Run Control of a Drifting Furnace | `lab/index.html?path=m5/DT-M5-lab-run-to-run.ipynb` |

Each module folder holds the notebook, its copy of `furnace.py` (1.0.0 in `m0`; 1.1.0 in `m1`, which adds `temperature_stream()`; 1.2.0 in `m2`, which adds `fab_records()`, the MES, trace and metrology records of 24 runs; 1.3.0 in `m3`, which adds `textbook_rates()` and `oxidation_runs()`, oxide thickness after dry oxidation at 1000 °C, including a thin-oxide run that grows faster than Deal-Grove; 1.4.0 in `m4`, which adds `vm_runs()`, 67 production runs with the furnace's trace summaries and the measured thickness of five sampled wafers per run, a new 140-minute recipe and one run that was not measured; 1.5.0 in `m5`, which adds `r2r_setup()` and `r2r_furnace()`, the recipe run by run while the thermocouple bias grows, in a drift and a failing scenario with the same disturbances for every controller; earlier functions are unchanged), any helpers the lab supplies (`twin_tools.py` in `m1`, `secom_tools.py` in `m2`, `oxide_tools.py` in `m3`, `vm_tools.py` in `m4`, `r2r_tools.py` in `m5`) and its check file (`dt_mK_check.py`). The check cell only imports the check file, so the reference solutions are not on screen; the check prints the learner's verification value only after every hidden test passes.

The Colab copies are in `colab/mK/`. They are the same notebooks with one extra first cell that downloads `furnace.py` and the check file from this repository. Open one with `https://colab.research.google.com/github/CSU-Physics/digital-twins-notebooks/blob/main/colab/m0/DT-M0-lab-furnace-log-colab.ipynb`.

## Course media (slides and explorers)

The folder `media/` holds self-contained HTML pages that Canvas embeds next to the readings. The deploy workflow copies it into the site, so each file is served at `https://csu-physics.github.io/digital-twins-notebooks/media/...`. They use no outside libraries and no network calls.

| File | What it is |
|---|---|
| `media/m0/m0-slides.html` | Module 0 slide deck, 19 slides: the course and the badge, the simulated furnace, NumPy arrays and the sample standard deviation, pandas tables and missing values, plotting with units, the moving average and its lag, debugging, the KEY and the verification value. Arrow keys or the buttons move between slides; **Print / PDF** gives one slide per page; `#s7` opens slide 7. Every number and curve is computed from `furnace.py` by `tools/build_dt_m0_media.py`. |
| `media/m0/m0-explore.html` | Module 0 explorer: smoothing and lag, with noise and window sliders, the hold period or the whole run, and a trailing average against the true temperature (`#smooth`); missing readings, comparing leaving them out, a forward fill and filling with zero (`#missing`). Every slider has a number box and works with the arrow keys. |
| `media/m1/m1-slides.html` | Module 1 slide deck, 20 slides: the Digital Twin Consortium and NIST IR 8356 definitions, the three parts of a twin, the model, shadow and twin classification, three levels of coupling, fidelity and update frequency, ISO 23247 in outline, the first-order furnace model and the fit of τ, the model alone against the synced twin, update interval and gain. Every number is computed by `tools/build_dt_m1_media.py`. |
| `media/m1/m1-explore.html` | Module 1 explorer: eight fab systems to classify as model, shadow or twin, with their data flows and level of coupling (`#sort`); a twin synced every 10 to 300 s with an adjustable gain and noise, with and without a disturbance (`#sync`). |
| `media/m2/m2-slides.html` | Module 2 slide deck, 21 slides: a wafer's path through the fab, equipment traces, metrology and the MES, SECS/GEM, Interface A and OPC UA, one run's data path, clocks and summaries, joining by run and wafer IDs, late and missing results, SECOM and screening, the accuracy trap, splitting by time, leakage, association and cause, NIST IR 8356 and SEMI E187. Built by `tools/build_dt_m2_media.py` from SECOM and `furnace.py`. |
| `media/m2/m2-explore.html` | Module 2 explorer: one wafer's data path in six hops, with a metrology delay against the twin's 12-hour limit (`#path`); a controller clock that disagrees with the MES and the hold mean it spoils (`#clock`); selection leakage with made-up records, ranked on training records only or on all records (`#leak`). |
| `media/m3/m3-slides.html` | Module 3 slide deck, 21 slides: how dry oxide grows, the Deal-Grove model and its two regimes, τ for a starting oxide, temperature and ambient, calibration data, the straight-line form and errors in variables, fitting x(t) with `curve_fit`, correlated estimates, residuals, confidence and prediction bands, extrapolation, the thin-oxide regime and the model's range. Built by `tools/build_dt_m3_media.py`. |
| `media/m3/m3-explore.html` | Module 3 explorer: oxide growth against time with temperature, time, starting oxide, dry or wet and (100) or (111) wafers, with the linear and parabolic limits and simulated measurements (`#grow`); a calibration design with the number of runs, σ and the longest run, a weighted fit and the confidence and prediction bands, inside and outside the calibrated range (`#bands`). |
| `media/m4/m4-slides.html` | Module 4 slide deck, 21 slides: virtual metrology and sampled metrology, the trace features, a linear model with five features, correlated features and ridge regression, overfitting, training, tuning and test data, cross-validation with preprocessing inside the folds, folds of whole runs, the baseline, physics plus a learned correction, the test runs and a new recipe, and the range check. Built by `tools/build_dt_m4_media.py`. |
| `media/m4/m4-explore.html` | Module 4 explorer: polynomial fits of degree 1 to 10 with the number of training points, the noise and a ridge penalty, and the training and test errors against the degree (`#complexity`); a straight line and a hybrid (Deal-Grove plus a learned correction) fitted to training runs and compared with the simulated furnace inside and outside the training range, with the range check (`#extrapolate`). |
| `media/m5/m5-slides.html` | Module 5 slide deck, 21 slides: thermocouple drift and its cost in thickness, the offset d between the furnace and its twin, a fixed recipe on a drifting furnace, the run-to-run loop, the EWMA and its weights, the recipe rule from the inverted Deal-Grove model, the one-variable Kalman filter and its settling gain, three controllers on the same disturbances, choosing λ, missing measurements, advisory and automatic mode, and the recalibration and maintenance decisions. Built by `tools/build_dt_m5_media.py`. |
| `media/m5/m5-explore.html` | Module 5 explorer: an EWMA run-to-run controller and a fixed recipe on a drifting furnace, with the bias growth, the noise and λ, the achieved thickness and oxidation time run by run, and the mean absolute error, largest deviation and runs held at a limit (`#loop`); EWMA and Kalman estimates of the same offset, with the drift, the noise, λ, Q, R and missing runs, and the Kalman gain run by run (`#estimate`). |

## Building and testing

- `tools/make_dt_m0.py` writes the Module 0 notebook and its Colab copy, and copies `checks/dt_m0_check.py` next to the notebook.
- `tools/build_dt_m0_media.py` writes the Module 0 slides from `tools/deck_head.html`, `tools/deck_bar.html` and `tools/deck_ctrl.js`.
- `python3 checks/test_dt_m0_check.py` runs the check against 16 correct and 13 deliberately wrong combinations of the learner's functions for three KEYs, and checks the messages a learner sees. It must report 0 wrong verdicts.
- `tools/make_dt_m0_sets.py` writes the 200 answer sets for Question 1 of the lab check to `dt_m0_solutions.json`. That file, like every answer key, is excluded by `.gitignore` and is never committed.
- Module 1 has the same set: `tools/make_dt_m1.py`, `tools/build_dt_m1_media.py`, `checks/test_dt_m1_check.py` (30 correct and 45 wrong combinations of `update_prediction()` and the fitted τ over three KEYs, 0 wrong verdicts) and `tools/make_dt_m1_sets.py` (`dt_m1_solutions.json`, gitignored).
- Module 2: `tools/make_dt_m2.py`, `tools/build_dt_m2_media.py`, `checks/test_dt_m2_check.py` (15 correct and 97 wrong combinations of the four functions over three KEYs, 0 wrong verdicts) and `tools/make_dt_m2_sets.py` (`dt_m2_solutions.json`, gitignored). The Module 2 explorer is written by hand.
- Module 3: `tools/make_dt_m3.py`, `tools/build_dt_m3_media.py`, `checks/test_dt_m3_check.py` (9 correct and 76 wrong combinations of `dg_tau()`, `dg_thickness()` and the fit over three KEYs, 0 wrong verdicts) and `tools/make_dt_m3_sets.py` (`dt_m3_solutions.json`, gitignored). The Module 3 explorer is written by hand; its fit was checked against `oxide_tools.fit_deal_grove()`.
- Module 4: `tools/make_dt_m4.py`, `tools/build_dt_m4_media.py`, `checks/test_dt_m4_check.py` (72 correct and 102 wrong combinations of `baseline_recent()`, `make_ridge()`, `hybrid_predict()` and `outside_range()` over three KEYs, 0 wrong verdicts) and `tools/make_dt_m4_sets.py` (`dt_m4_solutions.json`, gitignored; KEYs whose value is close to a rounding boundary, or whose two best ridge penalties are almost tied, are skipped). The Module 4 explorer is written by hand.
- Module 5: `tools/make_dt_m5.py`, `tools/build_dt_m5_media.py`, `checks/test_dt_m5_check.py` (81 correct and 93 wrong combinations of `ewma_update()`, `kalman_update()`, `next_time()` and `alarm_run()`, 0 wrong verdicts) and `tools/make_dt_m5_sets.py` (`dt_m5_solutions.json`, gitignored; KEYs whose value is close to a rounding boundary are skipped). The Module 5 explorer is written by hand.

The software versions are fixed in `requirements.txt` (JupyterLite 0.6.3 with the Pyodide kernel 0.6.1). Before a module is released, its notebook and every stored answer set are run in Pyodide 0.27.7 under Node. The kernel loads Pyodide 0.27.6 from its CDN in the learner's browser, from the same 0.27 release line, so each module is also opened once in a real browser before release.

## Data

`content/m2/data/` holds the SECOM data set: M. McCann and A. Johnston (2008), *SECOM*, UCI Machine Learning Repository, https://doi.org/10.24432/C54305, licensed CC BY 4.0. `secom.data` is stored compressed with gzip (`secom.data.gz`) so that it loads faster in the browser; unzipped it is byte for byte the UCI file. `secom_labels.data` and `secom.names` are unchanged. The folder's README gives the checksums.

## Licenses

- Course materials written for this course (notebook text, slides, explorers, figures and the simulated data sets) are licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Suggested credit: "Digital Twins for Semiconductor Manufacturing, SemiAcademy / ISEP-CSU, CC BY 4.0."
- The license of the course code (the simulator, check files and build scripts) is still to be confirmed by the course owner, to match the quantum computing notebooks repository.
- The SECOM data keep their own license, CC BY 4.0, with the credit above.
- JupyterLite (BSD 3-Clause), Pyodide (MPL 2.0), NumPy, pandas, SciPy and scikit-learn (BSD 3-Clause) and matplotlib (its PSF-based license) keep their own licenses; they are installed by the deploy workflow or loaded by Pyodide, and not stored here.

Developed through the Intel Semiconductor Education Program at Central State University (ISEP-CSU). Questions: mhadizadeh@centralstate.edu
