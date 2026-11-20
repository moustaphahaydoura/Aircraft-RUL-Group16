"""Infer final-cycle RUL using the frozen validation-selected model."""
import argparse
import json
import sys
from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from rul_pipeline import load_frame, make_windows, predict_bundle, seed_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path, help='26-column C-MAPSS histories; no labels needed')
    parser.add_argument('--output', type=Path, default=ROOT/'predictions.csv')
    args = parser.parse_args()
    seed_all(42)
    manifest = json.loads((ROOT/'artifacts/frozen_development.json').read_text())
    models = joblib.load(ROOT/'models/development_models.joblib')
    chosen = models[manifest['selected_model']]
    scaler = joblib.load(ROOT/'models/scaler.joblib')
    x, _, meta = make_windows(load_frame(args.input), scaler, 'test')
    p = predict_bundle(chosen['bundle'], x)
    q = chosen['radius90']
    meta.assign(predicted_rul=p, lower90=np.maximum(0,p-q), upper90=p+q,
                model=manifest['selected_model']).to_csv(args.output,index=False)
    print(f'Saved {len(meta)} engine predictions to {args.output}')
