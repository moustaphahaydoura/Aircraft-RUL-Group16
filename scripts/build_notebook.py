"""Build the submission notebook with embedded source and reproducible outputs."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):cells.append(nbf.v4.new_code_cell(s))

md('# Aircraft engine remaining useful life prediction\n## NASA C-MAPSS FD001 · CMPS 261 · Group 16\n'
   '**Moustapha Haydoura · Wassim Fawaz · Mohamad Khalil**\n\n'
   'Research question: does learning a temporal representation improve on a tuned classical model with the same historical information, and at what computational cost?\n\n'
   'This final notebook preserves the original six-step workflow and includes the complete implementation. '
   'The notebook and final report use the same saved experiment artifacts.')
md('### 1. Environment setup and complete implementation\n'
   'Install `requirements.txt` in Python 3.14 (the tested environment). Run from the project root. '
   'The notebook embeds the complete pipeline and can restore its source file when the file is missing. '
   'Training uses four CPU threads, seeded random generators, deterministic PyTorch operations, and no GPU requirement. '
   'Set `RUN_TRAINING=True` to regenerate experiments; the default review mode loads the completed run.')
code("from pathlib import Path\nimport sys, json\nimport numpy as np\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT=Path.cwd()\nassert (ROOT/'train_FD001.txt').exists(), 'Run this notebook from the project folder.'\n(ROOT/'src').mkdir(exist_ok=True)\nRUN_TRAINING=False")
source=(ROOT/'src/rul_pipeline.py').read_bytes().decode('utf-8')
embedded=source.replace('\\','\\\\').replace('"""','\\"""').replace('\r','\\r')
code('pipeline_source = """\\\n'+embedded+'"""\n\n'
     "pipeline_path=ROOT/'src/rul_pipeline.py'\n"
     "if not pipeline_path.exists():\n    pipeline_path.write_bytes(pipeline_source.encode('utf-8'))\n"
     "assert pipeline_path.read_bytes()==pipeline_source.encode('utf-8'), 'Rebuild the notebook after editing the pipeline source.'")
code("sys.path.insert(0,str(ROOT/'src'))\nimport rul_pipeline as pipeline\nprint('Python:',sys.version)\nprint('PyTorch:',pipeline.torch.__version__)\nprint('Feature count:',len(pipeline.FEATURES))")
md('### 2. Data audit, preprocessing, and piecewise RUL targets\n'
   'The 100 development engines are split 60/20/20 into fitting, validation, and independent calibration engines. '
   'Scaling is fitted on fitting engines only. Training labels are `min(last_cycle - current_cycle, 125)`. '
   'The target is at the window endpoint. Each window contains 30 observations from one engine. '
   'The retained inputs are 14 sensors plus two operating settings. Sensor 6 is near-constant, not exactly constant. '
   'The ID and cycle count are excluded from predictors. Validation uses three synthetic prefixes per held-out engine; '
   'calibration uses one prefix per independent engine. Official test RUL is never used for tuning.')
code("train_df=pipeline.load_frame(ROOT/'train_FD001.txt')\ndisplay(train_df.head())\nprint('Training rows/engines:',len(train_df),train_df.unit_id.nunique())\nprint('Missing values:',int(train_df.isna().sum().sum()))\nsplit=pipeline.engine_split(train_df,42)\nprint({k:len(v) for k,v in split.items()})\nassert not set(split['train']) & set(split['validation'])\nassert not set(split['train']) & set(split['calibration'])\ndisplay(train_df[pipeline.COLUMNS[2:]].agg(['nunique','std']).T)")
md('### 3. Evaluation metrics and safety asymmetry\n'
   'Primary evaluation uses **raw** remaining cycles: RMSE, MAE, and PHM sum. Capped RMSE is reported separately. '
   'For `d = prediction - truth`, PHM contributes `exp(-d/13)-1` when `d<0` and `exp(d/10)-1` otherwise. '
   'Thus the same positive error is more costly. The supplied notebook reversed those denominators. '
   'All model predictions are clipped to [0,125], consistently with the fitted target; prediction intervals have no upper cap.')
code("early,late=pipeline.phm_terms(np.array([50.,50.]),np.array([40.,60.]))\nprint(f'10 cycles early: {early:.4f}; 10 cycles late: {late:.4f}')\nassert late>early")
md('### 4. Classical baselines, historical features, and model selection\n'
   'Compare linear regression, random forest, and XGBoost. Instantaneous features have 16 dimensions; '
   'current + rolling mean + population standard deviation have 48; adding least-squares slope gives 64. '
   'All models share exactly the same training endpoints. Sensor ablations retain 14 sensors or the top six channels '
   'ranked by training XGBoost gain, without test-driven selection. Every candidate and its validation metrics are saved. '
   'RMSE selects ordinary models. The asymmetric CNN uses validation PHM to test a deliberately different objective. '
   'A final model is selected across variants by validation raw RMSE, never by its test rank.')
md('### 5. Sequence models, fitting, and independent uncertainty calibration\n'
   'The CNN retains two temporal convolutions, pooling, and a dense head. The LSTM restores the proposal comparison. '
   'Widths 32/64 and learning rates 0.001/0.0005 are paired in a bounded search. Standard MSE is compared with '
   'an overestimation-weighted MSE (ratios 2 and 5). Targets are divided by 125 for numerical conditioning. '
   'Adam, weight decay 1e-5, batch size 256, gradient clipping 5, at most 35 epochs and patience 8 are used. '
   'The best validation checkpoint is restored. Twenty unused calibration engines set nominal 90% residual intervals. '
   'Two additional engine splits assess sensitivity using hyperparameters selected on seed 42; this is not nested CV. '
   'No model is refitted on calibration or test engines.')
code("if RUN_TRAINING:\n    manifest=pipeline.fit_development(ROOT,seeds=(42,7,21),max_epochs=35)\nelse:\n    manifest=json.loads((ROOT/'artifacts/frozen_development.json').read_text())\nprint('Validation-selected model:',manifest['selected_model'])\ndisplay(pd.read_csv(ROOT/'artifacts/tuning_results.csv'))\ndisplay(pd.read_csv(ROOT/'artifacts/split_sensitivity.csv'))")
md('### 6. Final test evaluation, failure analysis, complexity, and visuals\n'
   'All 100 test engines receive one prediction at their final observed cycle, aligned by engine ID. '
   'Short histories are supported with left padding instead of silently dropping engines. '
   'The historical progress report already inspected this public test set: the present protocol freezes all new choices '
   'before evaluation, but it cannot undo that earlier exposure. Results describe FD001, not operational aircraft safety. '
   'Test-bootstrap intervals quantify sampling uncertainty for the fitted predictors; they do not quantify training variation.')
code("if RUN_TRAINING:\n    pipeline.evaluate_test(ROOT)\nresults=pd.read_csv(ROOT/'artifacts/test_metrics.csv')\ndisplay(results.round(3))\ndisplay(pd.read_csv(ROOT/'artifacts/bootstrap_intervals.csv').round(3))")
code("for name in ['dataset','model_comparison','ablations','learning_curves','failure_scatter','failure_stages','uncertainty','complexity_sensitivity']:\n    display(Image(filename=str(ROOT/f'figures/{name}.png'),width=1000))")
code("predictions=pd.read_csv(ROOT/'artifacts/test_predictions.csv')\nchosen=manifest['selected_model']\nerrors=predictions.assign(error=predictions[chosen]-predictions.true_rul)\ndisplay(errors[['unit_id','endpoint','true_rul',chosen,'error']].reindex(errors.error.abs().sort_values(ascending=False).index).head(10))\ndisplay(pd.read_csv(ROOT/'artifacts/stage_metrics.csv').round(3))\nprint('Interpretation and limitations: see Final_Report.pdf, sections 5-6.')")
md('### Reproduce figures, documents, and regression checks\n'
   'After a full rerun, execute `python scripts/make_figures.py` and `python scripts/build_report.py` to regenerate '
   'the figures and final report from the new artifacts. Run `python -m pytest -q` for leakage-boundary, '
   'metric-direction, window/label, short-history, feature-slope, shape, and calibration-quantile tests. '
   'Use `python scripts/predict.py test_FD001.txt --output predictions.csv` for label-free inference. '
   'The complete figure and document generation code is included in the repository under `scripts/`. '
   'References are listed in the final report; data are credited to NASA Ames PCoE (Saxena and Goebel, 2008).')
nb.cells=cells
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.14.8'}}
nbf.write(nb,ROOT/'aircraft_rul_pipeline.ipynb')
print('Built notebook with complete source and six-step workflow.')
