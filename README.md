# Digital Twins for Semiconductor Manufacturing: notebooks and media (SemiAcademy)

Lab notebooks, slides and interactive explorers for the SemiAcademy micro-credential *Digital Twins for Semiconductor Manufacturing* (Canvas course 57). The labs run **inside Canvas, in the learner's own browser**: GitHub Pages serves a JupyterLite site, and each Canvas lab page embeds one notebook. Python runs in the browser through Pyodide, so learners install nothing and need no account. Each lab also has an optional Google Colab copy.

Site: https://csu-physics.github.io/digital-twins-notebooks/

Every lab works with one **simulated** thermal oxidation furnace. Its model, `furnace.py`, sits next to each notebook and grows module by module. Every value it produces is simulated, and each learner's KEY (a whole number from 100 to 999, given by Question 1 of the lab check) selects that learner's own data.

## Notebooks

| Module | Notebook | Canvas page (course 57) | Lab address |
|---|---|---|---|
| 0 | `content/m0/DT-M0-lab-furnace-log.ipynb` | Module 0 Lab: Notebooks, NumPy and pandas on a Furnace Log | `lab/index.html?path=m0/DT-M0-lab-furnace-log.ipynb` |

Each module folder holds the notebook, its copy of `furnace.py` and its check file (`dt_mK_check.py`). The check cell only imports the check file, so the reference solutions are not on screen; the check prints the learner's verification value only after every hidden test passes.

The Colab copies are in `colab/mK/`. They are the same notebooks with one extra first cell that downloads `furnace.py` and the check file from this repository. Open one with `https://colab.research.google.com/github/CSU-Physics/digital-twins-notebooks/blob/main/colab/m0/DT-M0-lab-furnace-log-colab.ipynb`.

## Course media (slides and explorers)

The folder `media/` holds self-contained HTML pages that Canvas embeds next to the readings. The deploy workflow copies it into the site, so each file is served at `https://csu-physics.github.io/digital-twins-notebooks/media/...`. They use no outside libraries and no network calls.

| File | What it is |
|---|---|
| `media/m0/m0-slides.html` | Module 0 slide deck, 19 slides: the course and the badge, the simulated furnace, NumPy arrays and the sample standard deviation, pandas tables and missing values, plotting with units, the moving average and its lag, debugging, the KEY and the verification value. Arrow keys or the buttons move between slides; **Print / PDF** gives one slide per page; `#s7` opens slide 7. Every number and curve is computed from `furnace.py` by `tools/build_dt_m0_media.py`. |
| `media/m0/m0-explore.html` | Module 0 explorer: smoothing and lag, with noise and window sliders, the hold period or the whole run, and a trailing average against the true temperature (`#smooth`); missing readings, comparing leaving them out, a forward fill and filling with zero (`#missing`). Every slider has a number box and works with the arrow keys. |

## Building and testing

- `tools/make_dt_m0.py` writes the Module 0 notebook and its Colab copy, and copies `checks/dt_m0_check.py` next to the notebook.
- `tools/build_dt_m0_media.py` writes the Module 0 slides from `tools/deck_head.html`, `tools/deck_bar.html` and `tools/deck_ctrl.js`.
- `python3 checks/test_dt_m0_check.py` runs the check against 16 correct and 13 deliberately wrong combinations of the learner's functions for three KEYs, and checks the messages a learner sees. It must report 0 wrong verdicts.
- `tools/make_dt_m0_sets.py` writes the 200 answer sets for Question 1 of the lab check to `dt_m0_solutions.json`. That file, like every answer key, is excluded by `.gitignore` and is never committed.

The software versions are fixed in `requirements.txt` (JupyterLite 0.6.3 with the Pyodide kernel 0.6.1). Before a module is released, its notebook and every stored answer set are run in Pyodide 0.27.7 under Node. The kernel loads Pyodide 0.27.6 from its CDN in the learner's browser, from the same 0.27 release line, so each module is also opened once in a real browser before release.

## Licenses

- Course materials written for this course (notebook text, slides, explorers, figures and the simulated data sets) are licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Suggested credit: "Digital Twins for Semiconductor Manufacturing, SemiAcademy / ISEP-CSU, CC BY 4.0."
- The license of the course code (the simulator, check files and build scripts) is still to be confirmed by the course owner, to match the quantum computing notebooks repository.
- JupyterLite (BSD 3-Clause), Pyodide (MPL 2.0), NumPy, pandas and SciPy (BSD 3-Clause) and matplotlib (its PSF-based license) keep their own licenses; they are installed by the deploy workflow and not stored here.

Developed through the Intel Semiconductor Education Program at Central State University (ISEP-CSU). Questions: mhadizadeh@centralstate.edu
