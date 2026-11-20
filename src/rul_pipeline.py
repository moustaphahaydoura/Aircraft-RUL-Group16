"""Reproducible FD001 experiments. No test targets enter fitting or selection."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ['unit_id', 'time_cycles'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
# Sensor 6 is near-constant, not exactly constant; keep the original FD001 convention.
DROP = ['op_setting_3', 'sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
FEATURES = [c for c in COLUMNS[2:] if c not in DROP]
CAP = 125
WINDOW = 30
THREADS = 4


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)


def load_frame(path):
    frame = pd.read_csv(path, sep=r'\s+', header=None)
    if frame.shape[1] != 26:
        raise ValueError(f'{path}: expected 26 columns, got {frame.shape[1]}')
    frame.columns = COLUMNS
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Missing or nonfinite observations')
    if frame.duplicated(['unit_id', 'time_cycles']).any():
        raise ValueError('Duplicate engine/cycle keys')
    for _, g in frame.groupby('unit_id'):
        if not np.array_equal(g.time_cycles, np.arange(1, len(g) + 1)):
            raise ValueError('Cycles must be ordered, contiguous, and start at 1')
    frame[['unit_id', 'time_cycles']] = frame[['unit_id', 'time_cycles']].astype(int)
    return frame


def phm_terms(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 1 or len(y_true) == 0:
        raise ValueError('Metrics require equal, nonempty one-dimensional arrays')
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError('Nonfinite metric input')
    d = y_pred - y_true
    return np.expm1(np.where(d < 0, -d / 13.0, d / 10.0))


def metrics(y_true, y_pred):
    terms = phm_terms(y_true, y_pred)
    d = np.asarray(y_pred) - y_true
    return dict(rmse=float(np.sqrt(np.mean(d**2))), mae=float(np.mean(np.abs(d))),
                phm=float(terms.sum()), bias=float(d.mean()), over_rate=float(np.mean(d > 0)),
                over_20_rate=float(np.mean(d > 20)), n=len(d))


def engine_split(frame, seed=42):
    ids = np.sort(frame.unit_id.unique())
    ids = np.random.default_rng(seed).permutation(ids)
    n = len(ids)
    return dict(train=sorted(ids[:int(.6*n)].tolist()),
                validation=sorted(ids[int(.6*n):int(.8*n)].tolist()),
                calibration=sorted(ids[int(.8*n):].tolist()))


def make_windows(frame, scaler, mode='train', seed=42, features=FEATURES, window=WINDOW):
    """Windows never cross engines. Endpoint metadata preserve target alignment.

    Validation uses three random prefix endpoints per engine; calibration uses one.
    Draws use only the development engines, never official test lengths or labels.
    Test histories shorter than the window are left-padded with the first reading.
    """
    xs, ys, meta = [], [], []
    rng = np.random.default_rng(seed)
    for unit, g in frame.groupby('unit_id', sort=True):
        values = scaler.transform(g[features]).astype(np.float32)
        length = len(g)
        if mode == 'train':
            ends = range(window, length + 1)
        elif mode in ('validation', 'calibration'):
            count = 3 if mode == 'validation' else 1
            # Sample plausible residual horizons without using official test data.
            low = max(window, length - 150)
            ends = rng.choice(np.arange(low, length + 1), size=count, replace=False)
        elif mode == 'test':
            ends = [length]
        else:
            raise ValueError(mode)
        for end in ends:
            x = values[max(0, end-window):end]
            pad = window - len(x)
            if pad:
                x = np.concatenate([np.repeat(x[:1], pad, axis=0), x])
            xs.append(x)
            ys.append(length - end)
            meta.append(dict(unit_id=int(unit), endpoint=int(end), padded=bool(pad)))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=float), pd.DataFrame(meta)


def engineer(x, variant='history', indices=None):
    if indices is not None:
        x = x[:, :, indices]
    if variant == 'instant':
        return x[:, -1, :]
    parts = [x[:, -1, :], x.mean(axis=1), x.std(axis=1)]
    if variant == 'history':
        t = np.arange(x.shape[1], dtype=np.float32)
        t -= t.mean()
        parts.append(np.einsum('ntf,t->nf', x, t) / (t @ t))
    elif variant != 'mean_std':
        raise ValueError(variant)
    return np.concatenate(parts, axis=1)


class RULPredictor(nn.Module):
    """Original two-convolution structure with a dimension-safe dense layer."""
    def __init__(self, n_features=16, width=32, window=30):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(n_features, width, 3, padding=1), nn.ReLU(),
                                 nn.Conv1d(width, width*2, 3, padding=1), nn.ReLU(),
                                 nn.MaxPool1d(2), nn.Flatten(),
                                 nn.Linear(width*2*(window//2), 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x.transpose(1, 2)).squeeze(-1)


class LSTMPredictor(nn.Module):
    def __init__(self, n_features=16, width=32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, width, batch_first=True)
        self.head = nn.Sequential(nn.Linear(width, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        z, _ = self.lstm(x)
        return self.head(z[:, -1]).squeeze(-1)


class AsymmetricMSELoss(nn.Module):
    def __init__(self, penalty_ratio=1.):
        super().__init__()
        self.penalty_ratio = penalty_ratio

    def forward(self, y_pred, y_true):
        if y_pred.shape != y_true.shape:
            raise ValueError('Prediction/target shapes differ')
        d = y_pred - y_true
        return torch.where(d > 0, self.penalty_ratio*d.square(), d.square()).mean()


def nn_predict(model, x):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(torch.from_numpy(b)).numpy() for b in np.array_split(x, max(1, int(np.ceil(len(x)/512))))]) * CAP


def train_neural(kind, cfg, x, y, xv, yv, seed, max_epochs=35):
    seed_all(seed)
    model = RULPredictor(x.shape[-1], cfg['width'], x.shape[1]) if kind == 'cnn' else LSTMPredictor(x.shape[-1], cfg['width'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-5)
    loss_fn = AsymmetricMSELoss(cfg.get('ratio', 1.))
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.tensor(y/CAP, dtype=torch.float32)),
                        batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed))
    history, best, state, best_epoch, stale = [], float('inf'), None, 0, 0
    t0 = time.perf_counter()
    for epoch in range(1, max_epochs+1):
        model.train()
        total = 0.
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += loss.item()*len(yb)
        pred = np.clip(nn_predict(model, xv), 0, CAP)
        m = metrics(yv, pred)
        criterion = m['phm'] if cfg.get('ratio', 1) > 1 else m['rmse']
        history.append(dict(epoch=epoch, loss=total/len(x), **m))
        if criterion < best - 1e-6:
            best, state, best_epoch, stale = criterion, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if epoch % 10 == 0:
            print(f'  {kind} {cfg} epoch={epoch} val_rmse={m["rmse"]:.3f}', flush=True)
        if stale >= 8:
            break
    model.load_state_dict(state)
    return model, history, best_epoch, time.perf_counter()-t0


def conformal_radius(residuals, alpha=.1):
    """Finite-sample split conformal quantile; independent engines are the units."""
    residuals = np.asarray(residuals)
    if residuals.ndim != 1 or not len(residuals) or not 0 < alpha < 1:
        raise ValueError('Invalid calibration inputs')
    k = int(np.ceil((len(residuals)+1)*(1-alpha)))
    return float(np.sort(residuals)[k-1]) if k <= len(residuals) else float('inf')


def predict_bundle(bundle, x):
    if bundle['kind'] in ('cnn', 'lstm'):
        raw = nn_predict(bundle['model'], x)
    elif bundle['kind'] == 'constant':
        raw = np.full(len(x), bundle['value'])
    else:
        raw = bundle['model'].predict(engineer(x, bundle['variant'], bundle.get('indices')))
    return np.clip(raw, 0, CAP)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding='utf-8')


def fit_development(root=ROOT, seeds=(42, 7, 21), max_epochs=35):
    """Finish and serialize every modeling choice before opening official targets."""
    root = Path(root)
    for folder in ('artifacts', 'models', 'figures'):
        (root/folder).mkdir(exist_ok=True)
    train = load_frame(root/'train_FD001.txt')
    train.describe().T.to_csv(root/'artifacts/data_summary.csv')
    pd.DataFrame({'feature': COLUMNS[2:], 'unique': train[COLUMNS[2:]].nunique().values,
                  'std': train[COLUMNS[2:]].std().values, 'retained': [c in FEATURES for c in COLUMNS[2:]]}).to_csv(root/'artifacts/feature_audit.csv', index=False)
    all_tuning, all_sensitivity, splits, selected, histories = [], [], {}, {}, {}
    started = time.perf_counter()
    for seed in seeds:
        seed_all(seed)
        print(f'=== DEVELOPMENT SPLIT {seed} ===', flush=True)
        split = engine_split(train, seed)
        splits[str(seed)] = split
        parts = {k: train[train.unit_id.isin(v)].copy() for k, v in split.items()}
        scaler = MinMaxScaler().fit(parts['train'][FEATURES])
        x, raw_y, _ = make_windows(parts['train'], scaler)
        y = np.minimum(raw_y, CAP)
        xv, yv, mv = make_windows(parts['validation'], scaler, 'validation', seed+1000)
        xc, yc, mc = make_windows(parts['calibration'], scaler, 'calibration', seed+2000)
        if seed == seeds[0]:
            mv.assign(rul=yv).to_csv(root/'artifacts/validation_endpoints.csv', index=False)
            mc.assign(rul=yc).to_csv(root/'artifacts/calibration_endpoints.csv', index=False)
            joblib.dump(scaler, root/'models/scaler.joblib')
            np.savez_compressed(root/'artifacts/development_arrays.npz', xv=xv, yv=yv, xc=xc, yc=yc)
        local = {}
        def register(name, bundle, cfg, seconds, complexity, candidate):
            pred = predict_bundle(bundle, xv)
            m = metrics(yv, pred)
            row = dict(seed=seed, model=name, candidate=candidate, config=json.dumps(cfg, sort_keys=True),
                       fit_seconds=seconds, complexity=complexity, **m)
            all_tuning.append(row)
            objective = m['phm'] if name == 'CNN asymmetric' else m['rmse']
            if name not in local or objective < local[name]['objective']:
                local[name] = dict(bundle=bundle, config=cfg, fit_seconds=seconds, complexity=complexity,
                                   objective=objective, validation=m)
            print(f'{name} candidate {candidate}: val RMSE={m["rmse"]:.2f}, PHM={m["phm"]:.1f}', flush=True)

        register('Mean baseline', dict(kind='constant', value=float(y.mean())), {}, 0., 1, 0)
        # Equivalent endpoints and features for each classical algorithm.
        variants = ['instant', 'mean_std', 'history'] if seed == seeds[0] else ['instant', 'history']
        for variant in variants:
            xx, vv = engineer(x, variant), engineer(xv, variant)
            if seed == seeds[0]:
                t = time.perf_counter()
                model = LinearRegression().fit(xx, y)
                register('Linear '+variant, dict(kind='linear', model=model, variant=variant), {}, time.perf_counter()-t, xx.shape[1]+1, 0)
            grids = [dict(n_estimators=n, max_depth=d, learning_rate=lr, min_child_weight=mcw)
                     for n, d, lr, mcw in [(250, 3, .03, 5), (400, 3, .05, 5), (250, 5, .03, 5),
                                          (400, 5, .05, 5), (250, 3, .05, 20), (400, 5, .03, 20)]]
            if seed != seeds[0]:
                grids = [selected['XGB '+variant]['config']]
            for ci, cfg in enumerate(grids):
                model = XGBRegressor(**cfg, objective='reg:squarederror', subsample=.8, colsample_bytree=.8,
                                     reg_lambda=5., tree_method='hist', n_jobs=THREADS, random_state=seed)
                t = time.perf_counter()
                model.fit(xx, y)
                elapsed = time.perf_counter()-t
                register('XGB '+variant, dict(kind='xgb', model=model, variant=variant), cfg, elapsed,
                         len(model.get_booster().trees_to_dataframe()), ci)
            if variant in ('instant', 'history'):
                rf_grid = [dict(n_estimators=150, max_depth=d, min_samples_leaf=leaf, max_features=.8)
                           for d, leaf in [(12, 5), (None, 5), (12, 15), (None, 15)]]
                if seed != seeds[0]:
                    rf_grid = [selected['RF '+variant]['config']]
                for ci, cfg in enumerate(rf_grid):
                    model = RandomForestRegressor(**cfg, n_jobs=THREADS, random_state=seed)
                    t = time.perf_counter()
                    model.fit(xx, y)
                    elapsed = time.perf_counter()-t
                    register('RF '+variant, dict(kind='rf', model=model, variant=variant), cfg, elapsed,
                             sum(e.tree_.node_count for e in model.estimators_), ci)
        if seed == seeds[0]:
            base = local['XGB history']
            importance = base['bundle']['model'].feature_importances_.reshape(4, len(FEATURES)).sum(axis=0)
            rank = pd.DataFrame({'feature': FEATURES, 'training_gain_importance': importance}).sort_values('training_gain_importance', ascending=False)
            rank.to_csv(root/'artifacts/feature_importance.csv', index=False)
            sensor_indices = [i for i, c in enumerate(FEATURES) if c.startswith('sensor')]
            top_indices = sorted(sorted(sensor_indices, key=lambda i: importance[i], reverse=True)[:6])
            for name, indices in [('XGB sensors only', sensor_indices), ('XGB top6 sensors', top_indices)]:
                cfg = base['config']
                model = XGBRegressor(**cfg, objective='reg:squarederror', subsample=.8, colsample_bytree=.8,
                                     reg_lambda=5., tree_method='hist', n_jobs=THREADS, random_state=seed)
                t = time.perf_counter()
                model.fit(engineer(x, 'history', indices), y)
                elapsed = time.perf_counter()-t
                register(name, dict(kind='xgb', model=model, variant='history', indices=indices),
                         dict(**cfg, features=[FEATURES[i] for i in indices]), elapsed, len(model.get_booster().trees_to_dataframe()), 0)
        for name, kind, ratios in [('CNN MSE', 'cnn', [1.]), ('CNN asymmetric', 'cnn', [2., 5.]), ('LSTM MSE', 'lstm', [1.])]:
            grids = [dict(width=w, lr=lr, ratio=r) for w, lr in [(32, .001), (64, .0005)] for r in ratios]
            if seed != seeds[0]:
                grids = [{k: v for k, v in selected[name]['config'].items() if k != 'best_epoch'}]
            for ci, cfg in enumerate(grids):
                model, hist, epoch, seconds = train_neural(kind, cfg, x, y, xv, yv, seed, max_epochs)
                if seed == seeds[0]:
                    histories[f'{name}_{ci}'] = hist
                register(name, dict(kind=kind, model=model), dict(**cfg, best_epoch=epoch), seconds,
                         sum(p.numel() for p in model.parameters()), ci)
        for name, info in local.items():
            all_sensitivity.append(dict(seed=seed, model=name, **info['validation']))
        if seed == seeds[0]:
            selected = local
            for name, info in selected.items():
                bundle = info['bundle']
                pred = predict_bundle(bundle, xc)
                info['radius90'] = conformal_radius(np.abs(pred-yc))
                info['calibration_residuals'] = (yc-pred).tolist()
                filename = name.lower().replace(' ', '_')
                if bundle['kind'] in ('cnn', 'lstm'):
                    torch.save(dict(state_dict=bundle['model'].state_dict(), kind=bundle['kind'],
                                    config=info['config'], features=FEATURES, window=WINDOW, cap=CAP), root/f'models/{filename}.pt')
                elif bundle['kind'] != 'constant':
                    joblib.dump(bundle, root/f'models/{filename}.joblib')
                    if bundle['kind'] == 'xgb':
                        bundle['model'].save_model(root/f'models/{filename}.ubj')
            # Persist first-split fitted models before any test access.
            joblib.dump(selected, root/'models/development_models.joblib', compress=3)
        pd.DataFrame(all_tuning).to_csv(root/'artifacts/tuning_results.csv', index=False)
        pd.DataFrame(all_sensitivity).to_csv(root/'artifacts/split_sensitivity.csv', index=False)
        write_json(root/'artifacts/splits.json', splits)
        write_json(root/'artifacts/training_history.json', histories)
    best_name = min(selected, key=lambda k: selected[k]['validation']['rmse'])
    manifest = dict(primary_seed=seeds[0], seeds=list(seeds), features=FEATURES, window=WINDOW, cap=CAP,
                    train_rows=len(train), train_engines=int(train.unit_id.nunique()),
                    selected_model=best_name, selection='minimum validation raw RMSE across frozen model variants',
                    validation_protocol='3 random prefix endpoints per validation engine; remaining horizon 0..150 subject to 30 observed cycles',
                    calibration_protocol='one independent random prefix per calibration engine',
                    selected={name: {k: v for k, v in info.items() if k != 'bundle'} for name, info in selected.items()},
                    development_seconds=time.perf_counter()-started, python=platform.python_version(),
                    processor=platform.processor(), platform=platform.platform(), torch=torch.__version__, threads=THREADS,
                    max_epochs=max_epochs,
                    package_versions={p: importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','torch','xgboost']},
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    training_sha256=hashlib.sha256((root/'train_FD001.txt').read_bytes()).hexdigest())
    write_json(root/'artifacts/frozen_development.json', manifest)
    print(f'Frozen model selection: {best_name}; no official test labels used.', flush=True)
    return manifest


def evaluate_test(root=ROOT):
    root = Path(root)
    seed_all(42)
    frozen = json.loads((root/'artifacts/frozen_development.json').read_text())
    selected = joblib.load(root/'models/development_models.joblib')
    scaler = joblib.load(root/'models/scaler.joblib')
    frame = load_frame(root/'test_FD001.txt')
    x, _, meta = make_windows(frame, scaler, 'test')
    labels = np.loadtxt(root/'RUL_FD001.txt').reshape(-1)
    if not np.array_equal(meta.unit_id, np.arange(1, len(labels)+1)):
        raise ValueError('Official RUL row alignment requires engine IDs 1..N')
    y = labels[meta.unit_id.to_numpy()-1]
    preds, results, groups = meta.assign(true_rul=y), [], []
    for name, info in selected.items():
        bundle = info['bundle']
        p = predict_bundle(bundle, x)
        preds[name] = p
        for _ in range(3):
            predict_bundle(bundle, x)
        ts = []
        for _ in range(15):
            t = time.perf_counter()
            predict_bundle(bundle, x)
            ts.append((time.perf_counter()-t)*1000)
        q = info['radius90']
        lo, hi = np.maximum(0, p-q), p+q
        preds[name+'__lower90'], preds[name+'__upper90'] = lo, hi
        results.append(dict(model=name, **metrics(y, p), capped_rmse=metrics(np.minimum(y, CAP), p)['rmse'],
                            validation_rmse=info['validation']['rmse'], fit_seconds=info['fit_seconds'],
                            latency_ms_100=float(np.median(ts)), complexity=info['complexity'],
                            coverage90=float(np.mean((y >= lo)&(y <= hi))), mean_width90=float(np.mean(hi-lo))))
        for label, mask in [('0-30', y <= 30), ('31-80', (y > 30)&(y <= 80)), ('81-125', (y > 80)&(y <= 125)), ('>125', y > 125)]:
            if mask.any():
                groups.append(dict(model=name, stage=label, **metrics(y[mask], p[mask]), coverage90=float(np.mean((y[mask]>=lo[mask])&(y[mask]<=hi[mask])))))
    pd.DataFrame(results).to_csv(root/'artifacts/test_metrics.csv', index=False)
    pd.DataFrame(groups).to_csv(root/'artifacts/stage_metrics.csv', index=False)
    preds.to_csv(root/'artifacts/test_predictions.csv', index=False)
    # Paired engine bootstrap describes this test sample, not training uncertainty.
    chosen = frozen['selected_model']
    rng = np.random.default_rng(2026)
    idx = rng.integers(0, len(y), (5000, len(y)))
    boot = []
    for name in selected:
        err = preds[name].to_numpy()-y
        reference = preds[chosen].to_numpy()-y
        rmse = np.sqrt(np.mean(err[idx]**2, axis=1))
        delta = rmse-np.sqrt(np.mean(reference[idx]**2, axis=1))
        boot.append(dict(model=name, rmse_lo=np.quantile(rmse,.025), rmse_hi=np.quantile(rmse,.975),
                         delta_vs_selected_lo=np.quantile(delta,.025), delta_vs_selected_hi=np.quantile(delta,.975)))
    pd.DataFrame(boot).to_csv(root/'artifacts/bootstrap_intervals.csv', index=False)
    summary = dict(test_rows=len(frame), test_engines=len(y), min_history=int(frame.groupby('unit_id').size().min()),
                   padded_engines=meta.loc[meta.padded, 'unit_id'].tolist(), true_rul_min=float(y.min()), true_rul_max=float(y.max()),
                   above_cap=int((y>CAP).sum()), cap_only_rmse_floor=float(np.sqrt(np.mean(np.maximum(y-CAP,0)**2))),
                   selected_model=chosen, test_sha256=hashlib.sha256((root/'test_FD001.txt').read_bytes()).hexdigest(),
                   labels_sha256=hashlib.sha256((root/'RUL_FD001.txt').read_bytes()).hexdigest())
    write_json(root/'artifacts/test_summary.json', summary)
    print(pd.DataFrame(results)[['model','rmse','mae','phm','coverage90']].to_string(index=False), flush=True)
    return pd.DataFrame(results)


if __name__ == '__main__':
    # Import under a stable module name so checkpoints also load from the notebook/CLI.
    from rul_pipeline import fit_development, evaluate_test
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['develop','evaluate','all'], default='all')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42,7,21])
    parser.add_argument('--epochs', type=int, default=35)
    args = parser.parse_args()
    if args.stage in ('develop','all'):
        fit_development(seeds=tuple(args.seeds), max_epochs=args.epochs)
    if args.stage in ('evaluate','all'):
        evaluate_test()
