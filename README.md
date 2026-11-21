# Aircraft Engine Remaining Useful Life

CMPS 261 · American University of Beirut · Group 16

Moustapha Haydoura, Wassim Fawaz, and Mohamad Khalil

We use simulated turbofan sensor readings from NASA’s C-MAPSS FD001 dataset to estimate how many operating cycles an engine has left. We compare models that see only the latest reading, models that use summary statistics from the last 30 cycles, and CNN and LSTM models that take the full 30-cycle sequence.

Adding sensor history brought the random forest’s test RMSE down from 18.44 to 13.51 cycles. The CNN trained with mean squared error had the lowest validation RMSE, so we used it for the prediction script. Its test RMSE was 14.93 cycles.

For the full comparison, see the [final report](Final_Report.pdf), the [notebook](aircraft_rul_pipeline.ipynb), or the [test metrics](artifacts/test_metrics.csv). FD001 covers one operating condition and one fault mode.

## Setup

Clone the repository and run the commands below from its root. We ran the experiments on Windows with Python 3.14.8 on CPU. The setup uses `uv` to install the pinned dependencies.

```powershell
uv venv .venv --python 3.14
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

On macOS or Linux, use `.venv/bin/python` in place of `.venv/Scripts/python.exe`.

## View the saved results

Open [aircraft_rul_pipeline.ipynb](aircraft_rul_pipeline.ipynb) and leave `RUN_TRAINING = False` to load the saved results. The [pipeline source](src/rul_pipeline.py) contains the preprocessing, training, and evaluation code.

## Run a prediction

```powershell
.venv/Scripts/python.exe scripts/predict.py test_FD001.txt --output predictions.csv
```

The script uses the fitted model and scaler in `models/`. It accepts unlabeled histories in the 26-column C-MAPSS format, with consecutive cycles starting at 1 for each engine. The output contains one RUL estimate and a nominal 90% prediction interval per engine at its last recorded cycle.

## Reproduce everything

Keep `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` in the repository root. The first three commands run the tests, train and select the models, then score them against the official test labels. A full rerun overwrites saved models, results, and generated documents.

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe src/rul_pipeline.py --stage develop --seeds 42 7 21 --epochs 35
.venv/Scripts/python.exe src/rul_pipeline.py --stage evaluate
```

To rebuild the figures, report, and notebook and package the submission, continue with these commands.

```powershell
.venv/Scripts/python.exe scripts/make_figures.py
.venv/Scripts/python.exe scripts/build_report.py
.venv/Scripts/python.exe scripts/build_notebook.py
.venv/Scripts/python.exe scripts/execute_notebook.py
.venv/Scripts/python.exe scripts/predict.py test_FD001.txt --output artifacts/inference_predictions.csv
.venv/Scripts/python.exe scripts/validate_submission.py
.venv/Scripts/python.exe scripts/package_submission.py
```

## Data

[NASA C-MAPSS FD001](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) is credited to A. Saxena and K. Goebel (2008), NASA Ames Prognostics Center of Excellence.
