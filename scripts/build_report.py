"""Create the illustrated final report (PDF/DOCX) and optionally an A1 poster.

All numeric claims are read from experiment artifacts. Edit prose here, then rerun.
Optional repository URL: put the URL alone in repository_url.txt before building.
"""
import argparse
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr
import numpy as np
import pandas as pd
from matplotlib import get_data_path
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, A1, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.pdfgen import canvas
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
BLUE='#173c59'; TEAL='#147d78'; LIGHT='#edf3f6'
AUTHORS='Moustapha Haydoura · Wassim Fawaz · Mohamad Khalil'
REFERENCES=[
 ('[1] A. Saxena and K. Goebel (2008). Turbofan Engine Degradation Simulation Data Set. NASA Ames Prognostics Center of Excellence.',
  'https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/'),
 ('[2] A. Saxena, K. Goebel, D. Simon and N. Eklund (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. International Conference on Prognostics and Health Management.',
  'https://ntrs.nasa.gov/api/citations/20090029214/downloads/20090029214.pdf'),
 ('[3] T. Chen and C. Guestrin (2016). XGBoost: A scalable tree boosting system. KDD. DOI: 10.1145/2939672.2939785.',
  'https://arxiv.org/abs/1603.02754'),
 ('[4] L. Breiman (2001). Random forests. Machine Learning 45, 5-32.',
  'https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf'),
 ('[5] S. Hochreiter and J. Schmidhuber (1997). Long short-term memory. Neural Computation 9(8), 1735-1780.',
  'https://doi.org/10.1162/neco.1997.9.8.1735'),
 ('[6] X. Li, Q. Ding and J.-Q. Sun (2018). Remaining useful life estimation in prognostics using deep convolution neural networks. Reliability Engineering & System Safety 172, 1-11.',
  'https://escholarship.org/uc/item/5ns8r3fs'),
 ('[7] A. N. Angelopoulos and S. Bates (2021; revised 2022). A gentle introduction to conformal prediction and distribution-free uncertainty quantification.',
  'https://arxiv.org/abs/2107.07511')]


def content():
    m=pd.read_csv(ROOT/'artifacts/test_metrics.csv').set_index('model')
    f=json.loads((ROOT/'artifacts/frozen_development.json').read_text())
    s=json.loads((ROOT/'artifacts/test_summary.json').read_text())
    p=pd.read_csv(ROOT/'artifacts/test_predictions.csv')
    tuning=pd.read_csv(ROOT/'artifacts/tuning_results.csv')
    sens=pd.read_csv(ROOT/'artifacts/split_sensitivity.csv')
    boot=pd.read_csv(ROOT/'artifacts/bootstrap_intervals.csv').set_index('model')
    stage=pd.read_csv(ROOT/'artifacts/stage_metrics.csv')
    selected=f['selected_model']; best=m.rmse.idxmin()
    classical=m.loc[[n for n in m.index if n.startswith(('XGB','RF','Linear'))]].rmse.idxmin()
    neural=m.loc[['CNN MSE','CNN asymmetric','LSTM MSE']].rmse.idxmin()
    delta=m.loc[classical,'rmse']-m.loc[neural,'rmse']
    repo=(ROOT/'repository_url.txt').read_text(encoding='utf-8').strip() if (ROOT/'repository_url.txt').exists() else ''
    blocks=[]
    def title(t):blocks.append(('title',t))
    def h(t):blocks.append(('h',t))
    def sub(t):blocks.append(('sub',t))
    def para(t):blocks.append(('p',t))
    def fig(n,c):blocks.append(('fig',n,c))
    def table(headers,rows,widths=None):blocks.append(('table',headers,rows,widths))
    def page():blocks.append(('page',))
    title('Predicting aircraft engine\nremaining useful life')
    para('CMPS 261 · Final report · Group 16 · American University of Beirut\n'+AUTHORS+'\n20 November 2026')
    h('1. Abstract')
    para(f'We estimate remaining useful life (RUL) from the NASA C-MAPSS FD001 simulated engine trajectories and ask whether sequence learning improves on classical models given comparable history. A 60/20/20 engine split separates fitting, model selection, and interval calibration. We compare linear regression, random forests, XGBoost, a one-dimensional CNN, and an LSTM, with controlled historical-feature ablations. The model selected by validation, {selected}, achieves {m.loc[selected,"rmse"]:.2f} cycles RMSE and {m.loc[selected,"mae"]:.2f} cycles MAE on 100 official test endpoints. The strongest observed classical and neural test results are {m.loc[classical,"rmse"]:.2f} ({classical}) and {m.loc[neural,"rmse"]:.2f} ({neural}) cycles, respectively. Nominal 90% intervals for the selected model cover {100*m.loc[selected,"coverage90"]:.0f}% of test targets. We report tuning, split sensitivity, inference cost, and failure modes rather than treating one aggregate score as evidence of operational reliability.')
    h('2. Introduction')
    para('RUL is the number of operating cycles left before an engine reaches the end of its simulated trajectory. Predicting this quantity could help prioritize maintenance, but a useful predictor must distinguish normal variation from degradation and expose uncertainty when that distinction is weak. FD001 provides a manageable setting for studying those questions; it does not represent actual in-flight failures [1,2].')
    para('Our research question has two parts: does including recent sensor history improve prediction, and does learning a sequence representation add value beyond engineered summaries of that same history? The second part is essential. Comparing a CNN with a current-reading baseline alone would confound model architecture with access to information.')
    fig('workflow','Figure 1. The evaluation protocol separates engines and freezes decisions before the revised final test evaluation. Calibration engines are never used for fitting or model selection.')
    page()
    h('3. Dataset and preprocessing')
    para(f'The supplied FD001 files contain {f["train_rows"]:,} development observations from 100 run-to-failure engines and {s["test_rows"]:,} observations from 100 partially observed test engines. The subset represents one operating condition and high-pressure compressor degradation [1]. Each row contains an engine ID, a cycle index, three operating settings, and 21 sensors. The separate test label file provides remaining cycles after each engine’s last observed cycle. Rows are checked for finite values, duplicate engine-cycle keys, contiguous cycles, and the expected 26-column schema.')
    table(['Partition','Engines','Role'],[['Fitting','60','Fit scaler and candidate models'],['Validation','20','Three prefixes/engine; tune and select checkpoints'],['Calibration','20','One prefix/engine; set prediction interval width'],['Official test','100','One final observed endpoint/engine; report results']],[80,55,350])
    para('For an engine ending at cycle T, the raw target at cycle t is T - t. The fitted target is min(T - t,125). This piecewise target reflects a modeling assumption that early healthy operation offers limited information about the eventual failure time; the cap is fixed from the existing project rather than tuned on test results. The primary evaluation still uses raw RUL. A second, explicitly labeled capped RMSE shows how much the evaluation changes when that assumption is also imposed on the target.')
    para('The retained channels are operating settings 1 and 2 and sensors 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, and 21: 14 sensors plus two settings, not 16 sensors. The original exclusion list is retained for FD001. Sensor 6 is near-constant; the other excluded channels are constant or effectively constant in this subset. The full variability audit is supplied in feature_audit.csv. Removing these channels does not establish that the remaining predictors are independent.')
    para('Min-max scaling is fitted only on the 60 fitting engines. Validation, calibration, and test observations use those same extrema, without refitting or clipping scaled inputs. Thirty-cycle windows remain within engine boundaries. The training label belongs to the final cycle of each window. All classical and neural models use the same endpoints, avoiding an advantage from extra training examples. The engine ID and absolute cycle index are excluded from predictors.')
    fig('dataset','Figure 2. Observed lengths, the raw and capped target for training engine 1, and causal 15-cycle means of sensor 11 for three training engines. The smoothing in the right panel is descriptive; fitted windows use 30 cycles.')
    para(f'The shortest official test history has {s["min_history"]} cycles. The inference implementation nevertheless supports shorter histories by repeating the first observation on the left. No engine is silently dropped, and test targets are aligned by ID. There are {s["above_cap"]} test labels above 125; the target cap alone imposes an RMSE floor of {s["cap_only_rmse_floor"]:.2f} cycles for predictions restricted to [0,125].')
    page()
    h('4. Models, training, and validation')
    sub('4.1 Model ladder and historical information')
    para('The mean capped training target is a no-information baseline. Ordinary least-squares linear regression provides an interpretable reference. Random forests average randomized regression trees [4]. XGBoost builds regularized boosted trees [3]. For each historical representation, all features are computed causally from the same 30-cycle window: current readings (16 values), current + mean + population standard deviation (48), and current + mean + standard deviation + least-squares slope per cycle (64).')
    para('The CNN follows the existing notebook: Conv1d(16,w,3) → ReLU → Conv1d(w,2w,3) → ReLU → MaxPool(2) → flatten → dense(64) → ReLU → scalar. Padding preserves temporal length. The LSTM uses one recurrent layer of width w, the last hidden output, and a 32-unit ReLU head [5]. Both consume the ordered 30 × 16 input. Convolution is a reasonable local temporal model [6], while the LSTM tests the originally proposed recurrent alternative. Neither architecture is assumed to outperform a tuned tree model.')
    para('The proposal specified linear regression, random forest, and LSTM; the progress report substituted XGBoost and CNN. The final experiment retains both sets. Boosting provides a regularized nonlinear baseline for the engineered summaries, while convolution tests whether short local patterns can be learned with parallel temporal filters. Restoring the proposal models makes these changes comparative choices rather than unsupported replacements. The exponential-loss instability mentioned in the progress report is not reproduced here; the weighted MSE is evaluated directly as a bounded-cost surrogate.')
    sub('4.2 Bounded search and model selection')
    table(['Family','Candidate search'],[['XGBoost','6 tuples (trees, depth, learning rate, child weight): (250,3,.03,5); (400,3,.05,5); (250,5,.03,5); (400,5,.05,5); (250,3,.05,20); (400,5,.03,20)'],['Random forest','150 trees; max depth 12 or unlimited; minimum leaf size 5 or 15; feature fraction .8'],['CNN / LSTM','Paired (width, learning rate): (32,.001), (64,.0005); CNN asymmetric additionally compares penalty ratios 2 and 5']],[90,395])
    para('XGBoost uses histogram trees, row and column subsampling .8, and L2 regularization 5. Sensor-only and top-six ablations reuse the selected history-model configuration; they are not independently retuned. Neural targets are divided by 125. Adam uses weight decay 1e-5, batches of 256, gradient-norm clipping at 5, at most 35 epochs, and patience 8. The best checkpoint is restored. These small searches are documented experiments, not claims of exhaustive optimization.')
    para('With seed 42, 60 engines fit the models, 20 provide validation, and 20 remain reserved for calibration. For each validation engine, three distinct observed prefixes are drawn uniformly from endpoints allowing 30 observations and at most 150 remaining cycles. This produces 60 validation endpoints with equal engine representation. Candidate RMSE is computed against raw RUL. The asymmetric CNN deliberately selects by validation PHM instead; the final deployment candidate across model variants is still selected by validation raw RMSE. The prefix scheme approximates partial histories but is not guaranteed to match the official censoring distribution.')
    sub('4.3 Metrics and the asymmetric loss')
    para('For d = predicted RUL - true RUL, RMSE = sqrt(mean(d²)) and MAE = mean(|d|). The PHM sum is Σ[exp(-d/13)-1] for d < 0 and Σ[exp(d/10)-1] for d ≥ 0 [2]. Positive errors therefore receive the larger penalty. The supplied notebook had these denominators reversed. Training the asymmetric CNN instead uses the stable surrogate mean[d² × (r if d > 0 else 1)], with r = 2 or 5, not the exponential score itself. All point predictions are clipped to [0,125].')
    page()
    h('5. Results and analysis')
    sub('5.1 Final endpoint comparison')
    para(f'Table 1 reports every final model variant on the same 100 test endpoints. {selected} was selected before the revised test evaluation. The lowest observed test RMSE belongs to {best}; this retrospective rank does not change the frozen selection. All errors are in cycles except PHM, a dimensionless sum over 100 engines. Lower values are better.')
    table(['Model','Val. RMSE','Test RMSE','MAE','PHM','Capped RMSE'],[[n,f'{r.validation_rmse:.2f}',f'{r.rmse:.2f}',f'{r.mae:.2f}',f'{r.phm:.1f}',f'{r.capped_rmse:.2f}'] for n,r in m.iterrows()],[120,65,65,55,80,100])
    para('Table 1. Validation uses 60 prefixes from 20 engines; test uses 100 endpoints from 100 engines. Their PHM sums should not be compared directly because their sample counts differ. The last column alone uses capped truth.')
    fig('model_comparison','Figure 3. Test accuracy and asymmetric penalty. The logarithmic PHM axis prevents the weakest baseline from obscuring the remaining models. The green bar marks the validation-selected model where present.')
    para(f'The best observed neural result is {abs(delta):.2f} cycles {"lower" if delta>0 else "higher"} in RMSE than the best observed classical result. These are descriptive comparisons across several candidates. For the frozen selected model, an engine-bootstrap 95% interval for test RMSE is [{boot.loc[selected,"rmse_lo"]:.2f}, {boot.loc[selected,"rmse_hi"]:.2f}]. Paired differences for every candidate are supplied in bootstrap_intervals.csv. Resampling 100 test engines 5,000 times measures test-sample uncertainty conditional on the fitted models, not uncertainty from retraining.')
    page()
    sub('5.2 Historical features and sensor ablations')
    hist_delta=m.loc['XGB instant','rmse']-m.loc['XGB history','rmse']
    para(f'Adding mean, standard deviation, and slope to XGBoost changes raw test RMSE from {m.loc["XGB instant","rmse"]:.2f} to {m.loc["XGB history","rmse"]:.2f} cycles, a {abs(hist_delta):.2f}-cycle {"reduction" if hist_delta>0 else "increase"}. The mean/std-only variant reaches {m.loc["XGB mean_std","rmse"]:.2f}. Each representation was tuned separately within the same model family. From current readings to the full historical representation, linear regression improves from {m.loc["Linear instant","rmse"]:.2f} to {m.loc["Linear history","rmse"]:.2f} cycles and random forest from {m.loc["RF instant","rmse"]:.2f} to {m.loc["RF history","rmse"]:.2f}.')
    top=f['selected']['XGB top6 sensors']['config']['features']
    para(f'Removing operating settings leaves 14 sensor channels and 56 engineered values, with test RMSE {m.loc["XGB sensors only","rmse"]:.2f}. Keeping six sensor channels ({", ".join(t.replace("sensor_","s") for t in top)}) leaves 24 engineered values and gives {m.loc["XGB top6 sensors","rmse"]:.2f}. Channels are selected by the fitting model’s gain importance, summed over current, mean, standard deviation, and slope features. This is a predictive ranking, not evidence of a causal physical mechanism; correlated channels can share or substitute importance.')
    fig('ablations','Figure 4. XGBoost historical-feature and sensor ablations. Sensor subsets reuse the full-history hyperparameters. Training gain ranks channels without consulting test targets.')
    sub('5.3 Optimization behavior')
    fig('learning_curves','Figure 5. Loss and validation RMSE for the selected candidate within each neural family. The asymmetric training loss has a different scale; its checkpoint selection uses PHM, so its lowest-RMSE epoch need not be retained.')
    para('Validation curves provide a direct check on the fixed-epoch procedure in the original notebook. Training loss alone is insufficient for selecting a checkpoint, and an improvement in an overestimation-weighted loss need not improve RMSE. The saved histories include every epoch of each candidate, including trials that were not selected.')
    page()
    sub('5.4 Failure analysis and a tested mitigation')
    fig('failure_scatter','Figure 6. Endpoint predictions for historical XGBoost and both CNN objectives. Red points overestimate remaining life; blue points underestimate it. The horizontal dotted line is the fixed prediction cap.')
    fig('failure_stages','Figure 7. Error magnitude and bias by true remaining-life range. Positive bias corresponds to overestimation. Stage metrics use disjoint groups of the same official test engines.')
    a,b=m.loc['CNN MSE'],m.loc['CNN asymmetric']
    near=stage[stage.stage=='0-30'].set_index('model')
    para(f'The asymmetric CNN reduced the overall frequency of overestimation from {100*a.over_rate:.0f}% to {100*b.over_rate:.0f}%. For the {near.loc["CNN MSE","n"]} engines with 30 or fewer cycles left, that frequency fell from {100*near.loc["CNN MSE","over_rate"]:.0f}% to {100*near.loc["CNN asymmetric","over_rate"]:.0f}%. Total PHM rose from {a.phm:.1f} to {b.phm:.1f} and raw RMSE from {a.rmse:.2f} to {b.rmse:.2f} cycles, so fewer optimistic estimates did not translate to a lower aggregate penalty. The two CNN variants also differ in selected architecture and checkpoint, limiting what we can attribute to the loss alone.')
    para('The cap necessarily underestimates targets above 125. Even below that threshold, similar sensor states can correspond to different remaining horizons because trajectories begin with different health states and degradation rates. The plots and subgroup errors are consistent with that ambiguity but do not identify its physical cause. A practical next experiment would compare uncapped or multi-stage targets using development engines only, followed by evaluation on an additional untouched subset.')
    err=p[selected]-p.true_rul
    worst=p.loc[err.abs().nlargest(5).index]
    table(['Engine','Observed cycles','True RUL','Predicted RUL','Error'],[[int(r.unit_id),int(r.endpoint),f'{r.true_rul:.0f}',f'{r[selected]:.2f}',f'{r[selected]-r.true_rul:+.2f}'] for _,r in worst.iterrows()],[70,110,100,110,95])
    para(f'Table 2. The five largest absolute errors of {selected}; these examples are selected after evaluation for diagnosis, not for additional tuning. Full predictions and interval bounds are supplied in test_predictions.csv.')
    page()
    sub('5.5 Can the uncertainty estimates be trusted?')
    para('A separate set of 20 calibration engines supplies one randomly truncated endpoint each. For a fixed model, we calculate the absolute raw-RUL residual on each endpoint. The nominal 90% split-conformal radius is the 19th smallest of 20 residuals: ceil((20+1) × .9) = 19. Intervals are [max(0,prediction-q), prediction+q]. We do not cap the upper bound at 125, because the target is raw remaining life [7].')
    para(f'For {selected}, the radius is {f["selected"][selected]["radius90"]:.2f} cycles. On the official test engines, coverage is {100*m.loc[selected,"coverage90"]:.0f}% and mean width is {m.loc[selected,"mean_width90"]:.2f} cycles. These two quantities must be read together: a wide interval can cover many outcomes without providing a precise maintenance decision. Figure 8 checks whether aggregate coverage hides weaker performance in a particular life stage.')
    fig('uncertainty','Figure 8. Nominal 90% residual intervals for the validation-selected model, sorted by true test RUL for visualization only. Right: empirical coverage by life stage; no test labels are used to set interval widths.')
    table(['Model','Coverage (%)','Mean width','Radius q'],[[n,f'{100*m.loc[n,"coverage90"]:.0f}',f'{m.loc[n,"mean_width90"]:.2f}',f'{f["selected"][n]["radius90"]:.2f}'] for n in ['XGB history','CNN MSE','CNN asymmetric','LSTM MSE']],[150,110,110,115])
    para('Table 3. Interval widths are in cycles. Calibration residuals are independent of fitting and model selection, but only 20 independent engines determine each radius.')
    para('Split conformal prediction relies on exchangeability between calibration and deployment examples. Our synthetic prefix selection and the official test censoring process may differ, so nominal 90% coverage is not an unconditional guarantee here. It is also a marginal statement, not a promise for every engine or life stage. We therefore report observed coverage and widths, avoid describing the intervals as validated safety bounds, and recommend additional independent calibration under the intended censoring process.')
    page()
    sub('5.6 Computational complexity and split sensitivity')
    fig('complexity_sensitivity','Figure 9. Selected-fit cost versus test RMSE, and validation RMSE on three engine partitions. Seeds 7 and 21 reuse seed-42 hyperparameters and reselect neural checkpoints using their own validation engines.')
    names=['Linear history','RF history','XGB instant','XGB history','CNN MSE','CNN asymmetric','LSTM MSE']
    rows=[]
    for n in names:
        filename=n.lower().replace(' ','_')+('.pt' if n.startswith(('CNN','LSTM')) else '.joblib')
        size=(ROOT/'models'/filename).stat().st_size/1024
        rows.append([n,f'{m.loc[n,"fit_seconds"]:.2f}',f'{m.loc[n,"latency_ms_100"]:.2f}',f'{int(m.loc[n,"complexity"]):,}',f'{size:.1f}'])
    table(['Model','Fit (s)','100 inputs (ms)','Complexity count','File (KiB)'],rows,[125,70,100,110,80])
    para('Table 4. Complexity count means fitted coefficients for linear regression, tree nodes for forests/boosting, and trainable parameters for neural networks; these counts are not directly interchangeable. Latency is the median of 15 CPU calls after three warm-ups, including window-to-feature conversion and prediction but excluding file I/O, scaling, and window construction. Neural fit time includes validation passes and early stopping. The table shows only the selected fit, not the entire search.')
    para(f'Runs use Python {f["python"]}, PyTorch {f["torch"]}, four worker threads, and CPU execution on {f["processor"]}. Total development wall time across all candidates and three splits is {f["development_seconds"]/60:.1f} minutes. This is a local cost comparison, not a hardware-independent benchmark; background activity and small-batch overhead affect timings. All selected checkpoints and the scaler are included for label-free inference.')
    ss=sens[sens.model.isin(['XGB history','CNN MSE','CNN asymmetric','LSTM MSE'])].groupby('model').rmse.agg(['mean','std','min','max'])
    table(['Model','Mean val. RMSE','SD','Minimum','Maximum'],[[n,*[f'{v:.2f}' for v in r]] for n,r in ss.iterrows()],[150,100,75,80,80])
    para('Table 5. Sensitivity across development seeds 42, 7, and 21. The engine splits overlap, and additional splits reuse primary-split selected settings. This is a stress check on conclusions, not independent nested cross-validation or a confidence interval. Variation combines engine composition, random prefix selection, and model randomness. The official test is evaluated only for the frozen primary-split predictors.')
    page()
    h('6. Conclusion')
    xgb=m.loc['XGB history']; cnn=m.loc['CNN MSE']
    ci=boot.loc['XGB history']
    para(f'Historical features gave the clearest accuracy gain. XGBoost improved from {m.loc["XGB instant","rmse"]:.2f} cycles test RMSE with current readings to {xgb.rmse:.2f} with historical summaries. The validation-selected {selected} reached {m.loc[selected,"rmse"]:.2f}, while {classical} had the lowest observed test RMSE at {m.loc[classical,"rmse"]:.2f}. The paired 95% bootstrap interval for historical XGBoost minus {selected} spans [{ci.delta_vs_selected_lo:.2f}, {ci.delta_vs_selected_hi:.2f}] cycles, leaving the test advantage uncertain.')
    sens_pair=sens.pivot(index='seed',columns='model',values='rmse')
    gains=sens_pair['XGB history']-sens_pair['CNN MSE']
    para(f'The overall results give limited reason to pay the CNN’s fitting cost on FD001. Historical XGBoost minus CNN validation RMSE ranged from {gains.min():+.2f} to {gains.max():+.2f} cycles across the three splits. The selected CNN fit took {cnn.fit_seconds/xgb.fit_seconds:.1f} times as long as the XGBoost fit, while both predicted a prepared batch of 100 histories in under {math.ceil(max(cnn.latency_ms_100,xgb.latency_ms_100))} ms. For the 25 engines with 0–30 cycles remaining, the CNN reached {near.loc["CNN MSE","rmse"]:.2f} cycles RMSE, compared with {near.loc["XGB history","rmse"]:.2f} for historical XGBoost. A future study focused on near-failure decisions could validate directly on that horizon.')
    para('The principal methodological improvement is a comparison that controls information access and avoids test-driven tuning in this revision. The earlier RMSE 65.02 baseline and CNN RMSE 17.05 are not used as final evidence: no saved split, tuning history, checkpoint, or prediction file supports their reproduction, and the supplied scoring function reversed the PHM penalties. The new experiment reports both successful and unsuccessful variants, along with their resource costs and uncertainty.')
    para('The scope remains one simulated operating condition and fault mode. The 125-cycle cap creates a known long-horizon bias; calibration uses only 20 engines; repeated-split results are not independent CV; and the public test set had already been inspected in the progress report. Freezing the revised protocol cannot undo that historical exposure. Further work should use an untouched evaluation set, test uncapped or stage-specific targets, repeat training seeds within each split, and assess FD002-FD004 before making broader generalization claims.')
    sub('Submission and reproducibility')
    para('Final_Submission.zip contains the executed notebook, reusable Python modules, candidate scores and predictions, independent calibration residuals, environment versions, trained model files, and the fitted scaler. The README documents training, evaluation, figure regeneration, document generation, and label-free inference. Data files are credited to NASA Ames PCoE [1].')
    if repo:
        blocks.append(('link',repo))
    table(['Feedback / requirement','Where addressed'],[['Engine-disjoint validation and frozen selection','Sections 3–4, splits.json, frozen_development.json'],['Current versus historical classical models','Section 5.2, tuning_results.csv, Figure 4'],['Revisit weak XGBoost and show tuning','Sections 4.2 and 5.1, Appendices A–B'],['Explain changes from proposal','Section 4.1'],['Failure mitigation, uncertainty, complexity','Sections 5.4–5.6, Figures 6–9'],['Reproducible trained code','Notebook, src/, models/, README.md']],[200,285])
    page()
    h('7. References')
    for text,url in REFERENCES:
        para(text+'\n'+url)
    para('Course-specific requirements: supplied CMPS 261 course project description (project (1).pdf) and professor feedback (proffeedback.docx). The supplied proposal and progress report establish project history; their preliminary numerical claims are superseded by this final run.')
    page()
    h('Appendix A. Selected configurations')
    para('These settings were selected using development data. Exact settings and calibration residuals are machine-readable in frozen_development.json. The model is retained at the chosen validation checkpoint, without refitting on the validation or calibration engines.')
    rows=[]
    for n,info in f['selected'].items():
        cfg=info['config']
        st='No tuned hyperparameters' if not cfg else '; '.join(f'{k}={v}' for k,v in cfg.items())
        rows.append([n,st])
    table(['Model','Selected settings'],rows,[130,355])
    para('Every nonconstant model and the scaler have serialized artifacts in models/. For portability, neural .pt files include the state dictionary, architecture configuration, feature order, window length, and cap. XGBoost and random-forest bundles record their feature variant and sensor indices. Load local joblib files only from a trusted source, using the supplied environment.')
    page()
    h('Appendix B. Candidate tuning results')
    para('Candidate numbers follow the search order in Section 4.2. XGBoost candidates 0–5 use the six listed tuples. Random forest candidates 0–3 use (depth, minimum leaf) = (12,5), (unlimited,5), (12,15), (unlimited,15). Standard neural candidates 0/1 use the two paired settings. Asymmetric CNN candidates 0–3 use (32,.001,2), (32,.001,5), (64,.0005,2), (64,.0005,5). The sensitivity refit scores are in tuning_results.csv.')
    tt=tuning[tuning.seed==42]
    table(['Model','Trial','Val. RMSE','Val. MAE','Val. PHM','Fit (s)'],[[r.model,int(r.candidate),f'{r.rmse:.2f}',f'{r.mae:.2f}',f'{r.phm:.1f}',f'{r.fit_seconds:.2f}'] for _,r in tt.iterrows()],[140,45,75,75,80,70])
    return blocks,m,f,repo


def render_pdf(blocks,output_dir=ROOT):
    font_dir=Path(get_data_path())/'fonts'/'ttf'
    for name in ('DejaVuSans','DejaVuSans-Bold','DejaVuSans-Oblique'):
        pdfmetrics.registerFont(TTFont(name,str(font_dir/f'{name}.ttf')))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle('TitleCustom',fontName='DejaVuSans-Bold',fontSize=25,leading=29,textColor=colors.HexColor(BLUE),spaceAfter=14))
    styles.add(ParagraphStyle('BodyCustom',fontName='DejaVuSans',fontSize=9.5,leading=12.5,spaceAfter=7))
    styles.add(ParagraphStyle('HeadCustom',fontName='DejaVuSans-Bold',fontSize=15,leading=19,spaceBefore=6,spaceAfter=10,textColor=colors.HexColor(BLUE),keepWithNext=True))
    styles.add(ParagraphStyle('SubCustom',fontName='DejaVuSans-Bold',fontSize=11.5,leading=15,spaceBefore=7,spaceAfter=8,textColor=colors.HexColor(TEAL),keepWithNext=True))
    styles.add(ParagraphStyle('CaptionCustom',fontName='DejaVuSans-Oblique',fontSize=8,leading=10.5,spaceAfter=10,textColor=colors.HexColor('#495b66')))
    styles.add(ParagraphStyle('TableCustom',fontName='DejaVuSans',fontSize=8,leading=10))
    styles.add(ParagraphStyle('TableHeadCustom',fontName='DejaVuSans-Bold',fontSize=8,leading=10,textColor=colors.white))
    story=[]
    def p(t,style='BodyCustom'):return Paragraph(escape(str(t)).replace('\n','<br/>'),styles[style])
    for b in blocks:
        kind=b[0]
        if kind in ('title','h','sub','p'):
            story.append(p(b[1],{'title':'TitleCustom','h':'HeadCustom','sub':'SubCustom','p':'BodyCustom'}[kind]))
        elif kind=='link':
            story.append(Paragraph(f'GitHub repository<br/><link href={quoteattr(b[1])} color="{TEAL}"><u>{escape(b[1])}</u></link>',styles['BodyCustom']))
        elif kind=='page':story.append(PageBreak())
        elif kind=='fig':
            path=ROOT/f'figures/{b[1]}.png'
            w,h=PILImage.open(path).size
            im=Image(str(path),width=485,height=485*h/w)
            story.append(KeepTogether([Spacer(1,4),im,p(b[2],'CaptionCustom')]))
        elif kind=='table':
            data=[[p(x,'TableHeadCustom') for x in b[1]]]+[[p(x,'TableCustom') for x in row] for row in b[2]]
            t=Table(data,colWidths=b[3],repeatRows=1,hAlign='LEFT')
            padding=2.25 if len(data)>35 else 4
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(BLUE)),('VALIGN',(0,0),(-1,-1),'TOP'),
                                   ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor(LIGHT)]),
                                   ('BOTTOMPADDING',(0,0),(-1,-1),padding),('TOPPADDING',(0,0),(-1,-1),padding),
                                   ('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#b9c9d3'))]))
            story.extend([t,Spacer(1,9)])
    def footer(c,doc):
        c.saveState();c.setStrokeColor(colors.HexColor('#c7d6df'));c.line(55,43,A4[0]-55,43)
        c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#586b77'))
        c.drawString(55,30,'CMPS 261 · Group 16 | Aircraft engine RUL')
        c.drawRightString(A4[0]-55,30,str(doc.page));c.restoreState()
    doc=SimpleDocTemplate(str(output_dir/'Final_Report.pdf'),pagesize=A4,leftMargin=55,rightMargin=55,topMargin=42,bottomMargin=56,
                          title='Aircraft engine remaining useful life — Final report',author=AUTHORS)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def render_docx(blocks,output_dir=ROOT):
    d=Document();sec=d.sections[0]
    sec.top_margin=Inches(.65);sec.bottom_margin=Inches(.65);sec.left_margin=Inches(.75);sec.right_margin=Inches(.75)
    sec.page_width=Inches(8.27);sec.page_height=Inches(11.69)
    normal=d.styles['Normal'];normal.font.name='Calibri';normal.font.size=Pt(10)
    normal.paragraph_format.space_after=Pt(6)
    normal.paragraph_format.line_spacing=Pt(12.5)
    for name in ['Title','Heading 1','Heading 2']:
        d.styles[name].font.color.rgb=RGBColor.from_string('173C59')
    for b in blocks:
        if b[0]=='title':d.add_heading(b[1],0)
        elif b[0]=='h':d.add_heading(b[1],1)
        elif b[0]=='sub':d.add_heading(b[1],2)
        elif b[0]=='p':d.add_paragraph(b[1])
        elif b[0]=='link':
            p=d.add_paragraph('GitHub repository\n')
            hyperlink=OxmlElement('w:hyperlink')
            hyperlink.set(qn('r:id'),p.part.relate_to(b[1],RT.HYPERLINK,is_external=True))
            run=OxmlElement('w:r')
            properties=OxmlElement('w:rPr')
            style=OxmlElement('w:rStyle');style.set(qn('w:val'),'Hyperlink')
            properties.append(style);run.append(properties)
            label=OxmlElement('w:t');label.text=b[1];run.append(label)
            hyperlink.append(run);p._p.append(hyperlink)
        elif b[0]=='page':d.add_page_break()
        elif b[0]=='fig':
            d.add_picture(str(ROOT/f'figures/{b[1]}.png'),width=Inches(6.65))
            d.paragraphs[-1].paragraph_format.keep_with_next=True
            p=d.add_paragraph(b[2]);p.style='Caption'
        elif b[0]=='table':
            t=d.add_table(rows=1,cols=len(b[1]));t.style='Light Shading Accent 1'
            t.autofit=False
            for column,width in zip(t.columns,b[3]):column.width=Inches(6.65*width/sum(b[3]))
            t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
            for c,v in zip(t.rows[0].cells,b[1]):c.text=str(v)
            for row in b[2]:
                for c,v in zip(t.add_row().cells,row):c.text=str(v)
            for row in t.rows:
                row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
                for c,column in zip(row.cells,t.columns):
                    c.width=column.width
                    for p in c.paragraphs:
                        p.paragraph_format.space_after=Pt(3)
                        p.paragraph_format.space_before=Pt(3)
                        p.paragraph_format.line_spacing=Pt(10)
                        for r in p.runs:r.font.size=Pt(8)
    foot=sec.footer.paragraphs[0];foot.text='CMPS 261 · Group 16 | '
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');foot._p.append(field)
    d.core_properties.title='Predicting aircraft engine remaining useful life'
    d.core_properties.author=AUTHORS
    with (output_dir/'Final_Report.docx').open('wb') as output:
        d.save(output)


def poster(m,f,repo):
    W,H=landscape(A1);c=canvas.Canvas(str(ROOT/'Research_Poster.pdf'),pagesize=(W,H))
    c.setTitle('Aircraft engine RUL — Group 16 research poster');c.setAuthor(AUTHORS)
    c.setFillColor(colors.HexColor('#f5f8fa'));c.rect(0,0,W,H,fill=1,stroke=0)
    c.setFillColor(colors.HexColor(BLUE));c.rect(0,H-230,W,230,fill=1,stroke=0)
    c.setFillColor(colors.white);c.setFont('Helvetica-Bold',54)
    c.drawString(65,H-82,'Does learning the sequence improve engine RUL prediction?')
    c.setFont('Helvetica',28);c.drawString(65,H-136,'NASA C-MAPSS FD001 | Classical baselines, temporal models, and honest uncertainty')
    c.setFont('Helvetica',23);c.drawString(65,H-185,AUTHORS+'  |  CMPS 261 · Group 16 · AUB')
    gap=36;cw=(W-130-2*gap)/3;top=H-270
    styles=getSampleStyleSheet()
    body=ParagraphStyle('posterbody',fontName='Helvetica',fontSize=21,leading=29,textColor=colors.HexColor('#243746'),spaceAfter=15)
    heading=ParagraphStyle('posterhead',fontName='Helvetica-Bold',fontSize=28,leading=34,textColor=colors.HexColor(TEAL))
    def text_at(x,y,text,style=body):
        p=Paragraph(escape(text).replace('\n','<br/>'),style);_,hh=p.wrap(cw,2000);p.drawOn(c,x,y-hh);return y-hh-20
    def pic(x,y,name):
        path=ROOT/f'figures/{name}.png';w,h=PILImage.open(path).size;hh=cw*h/w
        c.drawImage(str(path),x,y-hh,width=cw,height=hh,mask='auto');return y-hh-20
    chosen=f['selected_model'];r=m.loc[chosen]
    x=65;y=top
    y=text_at(x,y,'QUESTION & DATA',heading)
    y=text_at(x,y,'Does a CNN or LSTM improve on a strong classical model when both can use the same 30-cycle history? What does any improvement cost?')
    y=text_at(x,y,'100 run-to-failure development engines + 100 partial test engines. One simulated operating condition and one fault mode. Inputs: 14 sensors + 2 settings. Training targets capped at 125; primary evaluation uses raw RUL.')
    y=text_at(x,y,'EVALUATION DESIGN',heading)
    y=text_at(x,y,'60 fitting / 20 validation / 20 calibration engines. Scaling uses fitting engines only. Hyperparameters and checkpoints are frozen before final test evaluation. One prediction per test engine.')
    y=text_at(x,y,'MODEL LADDER',heading)
    y=text_at(x,y,'Mean baseline → linear regression → random forest / XGBoost → CNN / LSTM. Current readings, mean/std summaries, and trend features isolate the benefit of history. An asymmetric CNN tests the cost of overestimation.')
    y=pic(x,y,'poster_ablations')
    x=65+cw+gap;y=top
    y=text_at(x,y,'RESULTS',heading)
    y=text_at(x,y,f'Validation-selected: {chosen}\nTest RMSE {r.rmse:.2f} cycles | MAE {r.mae:.2f}\nPHM {r.phm:.1f} across 100 test engines')
    y=pic(x,y,'poster_models')
    y=text_at(x,y,'FAIR COMPARISON',heading)
    y=text_at(x,y,f'XGBoost current-only: {m.loc["XGB instant","rmse"]:.2f} cycles RMSE. With historical summaries: {m.loc["XGB history","rmse"]:.2f}. CNN MSE: {m.loc["CNN MSE","rmse"]:.2f}. LSTM: {m.loc["LSTM MSE","rmse"]:.2f}. Architecture alone does not explain the difference.')
    y=pic(x,y,'poster_cost')
    y=text_at(x,y,'Selected-fit costs on the same CPU. Three engine splits check sensitivity; they are not nested cross-validation.',ParagraphStyle('small',parent=body,fontSize=17,leading=23))
    x=65+2*(cw+gap);y=top
    y=text_at(x,y,'FAILURES & UNCERTAINTY',heading)
    y=pic(x,y,'poster_failures')
    y=text_at(x,y,f'Asymmetric CNN: PHM {m.loc["CNN asymmetric","phm"]:.1f} versus {m.loc["CNN MSE","phm"]:.1f} for MSE. The target cap creates long-horizon underestimation. A lower overprediction rate is not an operational safety guarantee.')
    y=pic(x,y,'poster_intervals')
    y=text_at(x,y,f'Nominal 90% intervals for {chosen}: {100*r.coverage90:.0f}% observed coverage; {r.mean_width90:.1f}-cycle mean width. Only 20 calibration engines; synthetic censoring may differ from test censoring.')
    y=text_at(x,y,'CONCLUSION',heading)
    y=text_at(x,y,'Judge temporal learning against tuned historical baselines, measured costs, and split variation. These results support a bounded FD001 comparison, not deployment on real aircraft. The public test set was already exposed in the earlier progress report.')
    # Fixed footer is separate from the three content columns.
    c.setFillColor(colors.HexColor(BLUE));c.rect(0,0,W,126,fill=1,stroke=0)
    c.setFillColor(colors.white);c.setFont('Helvetica-Bold',18);c.drawString(65,94,'REFERENCES & REPRODUCIBILITY')
    c.setFont('Helvetica',16)
    c.drawString(65,65,'Saxena & Goebel (NASA, 2008); Saxena et al. (PHM, 2008); Chen & Guestrin (2016); Breiman (2001); Hochreiter & Schmidhuber (1997).')
    c.drawString(65,40,'Li, Ding & Sun (2018); Angelopoulos & Bates (2021). Full citations, notebook, predictions, checkpoints, and commands accompany the final report.')
    c.save()
    # Save editable poster source as an artifact of the same builder (the script itself).
    print('Poster content bottom positions checked during PDF layout review.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poster',action='store_true',help='Also render the poster from its figure panels')
    parser.add_argument('--output-dir',type=Path,default=ROOT,help='Directory for the two final reports')
    args=parser.parse_args()
    blocks,m,f,repo=content()
    (ROOT/'artifacts/report_content.json').write_text(json.dumps(blocks,indent=2,ensure_ascii=False),encoding='utf-8')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    render_pdf(blocks,args.output_dir);render_docx(blocks,args.output_dir)
    print('Created Final_Report.pdf and Final_Report.docx')
    if args.poster:
        poster(m,f,repo)
