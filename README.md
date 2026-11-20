# Aircraft engine remaining useful life — Group 16

CMPS 261, American University of Beirut. Moustapha Haydoura, Wassim Fawaz, and Mohamad Khalil.

This project compares classical and sequence models on NASA C-MAPSS FD001. The central comparison separates the benefit of sensor history from the benefit of learning a temporal representation. The final report and notebook use the same saved experimental results.

## Deliverables

- `Final_Report.pdf` — illustrated final report, including all selected models, tuning outcomes, failure analysis, uncertainty, and computational cost.
- `Final_Report.docx` — editable report with the same content.
- `Final_Submission.zip` — report, executed notebook, data, code, trained models, run artifacts, and a verified SHA-256 manifest.
- `aircraft_rul_pipeline.ipynb` — executed notebook, retaining the original six-step workflow and embedding the full pipeline source.
- `src/rul_pipeline.py` — preprocessing, feature engineering, models, tuning, calibration, and final evaluation.
- `artifacts/` — candidate scores, engine splits, frozen configuration, predictions, calibration residuals, training histories, bootstrap intervals, and data checksums.
- `models/` — fitted scaler, classical estimators, and neural checkpoints.
- `figures/` — reproducible PNG and vector PDF plots.
- `tests/` — regression checks for scoring, engine separation, endpoint alignment, causal features, short histories, and interval quantiles.

## Setup

The completed run uses Windows, Python 3.14.8, and CPU execution. Four threads are used for the fitted models. Install the exact package versions recorded in `requirements.txt`:

```powershell
uv venv .venv --python 3.14
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

On other systems, use your Python executable and a virtual environment. The source contains no absolute local paths. PyTorch and numerical-library differences can change timing and small numerical details; the saved predictions are the exact record behind the submitted report.

## Review the completed run

Open `aircraft_rul_pipeline.ipynb` from the project root. `RUN_TRAINING=False` loads the saved experiment and displays its results. The notebook embeds the full pipeline source and can restore `src/rul_pipeline.py` when the file is missing. Edit the source and rebuild the notebook if you change the implementation.

## Reproduce everything

Keep the three supplied FD001 text files in the project root. The pipeline does not download data or require a GPU.

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe src/rul_pipeline.py --stage develop --seeds 42 7 21 --epochs 35
.venv/Scripts/python.exe src/rul_pipeline.py --stage evaluate
.venv/Scripts/python.exe scripts/make_figures.py
.venv/Scripts/python.exe scripts/build_report.py
.venv/Scripts/python.exe scripts/build_notebook.py
.venv/Scripts/python.exe scripts/execute_notebook.py
.venv/Scripts/python.exe scripts/predict.py test_FD001.txt --output artifacts/inference_predictions.csv
.venv/Scripts/python.exe scripts/validate_submission.py
.venv/Scripts/python.exe scripts/package_submission.py
```

Development completes and serializes all choices before evaluation reads the official target file. A full rerun overwrites derived artifacts, models, and documents. Copy the completed submission first if you want to preserve its exact run. `--stage all` combines development and evaluation. Reduced epochs or fewer seeds are useful for debugging but are not the submitted experiment.

## Predict without labels

The validation-selected model is CNN MSE. Download its [PyTorch checkpoint](models/cnn_mse.pt) and [fitted scaler](models/scaler.joblib). The command below loads the [complete frozen model bundle](models/development_models.joblib) and the fitted scaler. Cloning the repository provides all three files as well as the frozen configuration and the other trained models.

```powershell
.venv/Scripts/python.exe scripts/predict.py test_FD001.txt --output predictions.csv
```

The input follows the 26-column C-MAPSS schema, grouped by engine with consecutive cycles beginning at one. The command loads the validation-selected model and scaler, constructs the final window for each engine, and outputs RUL plus nominal 90% interval bounds. It never loads the RUL target file. Short histories are left-padded with the first reading.

## Experimental protocol

The 100 development engines are partitioned 60/20/20 into fitting, validation, and calibration sets. Only fitting engines determine scaling and weights. Labels are capped at 125 for fitting; raw RUL is the primary validation/test target. Validation uses three random prefixes per engine, calibration one prefix per engine, and test one official endpoint per engine. Exact IDs and endpoints are saved.

All learned models use the same 30-cycle training endpoints. Classical inputs are current readings, current/mean/std, or current/mean/std/slope. CNN and LSTM inputs retain the full ordered sequence. XGBoost sensor ablations exclude settings or retain six channels ranked by fitting-only gain. The strongest validation RMSE determines the retained inference model. The asymmetric CNN is selected within its family using validation PHM to study that alternative objective.

PHM uses `exp(-error/13)-1` for underestimation and `exp(error/10)-1` for overestimation, where error is prediction minus truth. The original notebook reversed these constants. Point predictions are clipped to [0,125]; interval upper bounds are not capped. Capped-target RMSE is reported separately and must not be confused with raw RMSE.

Seeds 7 and 21 are sensitivity refits using seed-42 hyperparameters, with their own engine partitions, prefix draws, and neural checkpoint selection. They are not independent nested CV. Twenty separate calibration engines determine the finite-sample residual quantile for each model. Nominal coverage depends on exchangeability; synthetic prefix selection may differ from official test censoring.

The progress report already inspected this public test set. The revised workflow prevents new test-driven tuning but cannot erase that historical exposure. FD001 is simulated and limited to one operating condition and fault mode; these experiments do not validate real aircraft maintenance decisions.

## Report editing

Edit `scripts/build_report.py`, then rerun it to regenerate PDF and DOCX from the saved CSVs. Edit `scripts/make_figures.py` to change charts. Both scripts accept an optional `--poster` flag for the separate poster workflow. Run the figure command first when generating documents.

To insert the course-required repository link, save the actual GitHub URL as the sole line of `repository_url.txt` and rerun `scripts/build_report.py`. Do not submit a fabricated or uncreated repository URL. The repository should include the notebook, code, requirements, artifacts, and trained model files so the final fitted model is downloadable.

## Data and references

The supplied `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` are credited to A. Saxena and K. Goebel (2008), NASA Ames Prognostics Center of Excellence. Dataset provenance: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

The report contains primary references for C-MAPSS, PHM evaluation, XGBoost, random forests, LSTM, convolutional RUL prediction, and conformal intervals. Do not use the old progress-report scores or original figure as final experimental evidence.
