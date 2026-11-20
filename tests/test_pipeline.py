import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from rul_pipeline import (AsymmetricMSELoss, RULPredictor, conformal_radius, engineer,
                          engine_split, make_windows, metrics, phm_terms)


def frame():
    return pd.DataFrame({'unit_id': [1]*4+[2]*2, 'time_cycles': [1,2,3,4,1,2],
                         'sensor': [10.,11.,12.,13.,100.,101.]})


def test_phm_overestimation_is_more_expensive():
    terms = phm_terms(np.array([50.,50.]), np.array([40.,60.]))
    assert terms[0] == pytest.approx(np.expm1(10/13))
    assert terms[1] == pytest.approx(np.expm1(1))
    assert terms[1] > terms[0]
    assert metrics(np.array([1.,2.]), np.array([1.,2.]))['rmse'] == 0


def test_metric_alignment_is_not_silently_truncated():
    with pytest.raises(ValueError):
        phm_terms(np.array([1.,2.]), np.array([1.]))
    with pytest.raises(ValueError):
        metrics(np.array([1.]), np.array([np.nan]))


def test_windows_do_not_cross_engines_and_targets_match_endpoint():
    df = frame()
    scaler = MinMaxScaler().fit(df[['sensor']])
    x, y, meta = make_windows(df, scaler, features=['sensor'], window=3)
    assert x.shape == (2,3,1)
    np.testing.assert_array_equal(y, [1,0])
    assert meta.unit_id.tolist() == [1,1]
    np.testing.assert_allclose(x[-1,:,0], scaler.transform(df[['sensor']].iloc[1:4])[:,0])


def test_short_test_engine_is_padded_not_dropped():
    df = frame()
    scaler = MinMaxScaler().fit(df[['sensor']])
    x, _, meta = make_windows(df, scaler, 'test', features=['sensor'], window=3)
    assert meta.unit_id.tolist() == [1,2]
    assert meta.padded.tolist() == [False,True]
    assert x[1,0,0] == x[1,1,0]
    assert x[1,2,0] > x[1,1,0]


def test_preprocessing_uses_only_training_range():
    df = frame()
    scaler = MinMaxScaler().fit(df.loc[df.unit_id == 1, ['sensor']])
    x, _, _ = make_windows(df[df.unit_id == 2], scaler, 'test', features=['sensor'], window=3)
    assert x.min() > 1  # transformed unseen observations are not clipped or refit
    assert scaler.data_max_[0] == 13


def test_engine_split_disjoint_and_reproducible():
    df = pd.DataFrame({'unit_id': np.arange(1,101)})
    split = engine_split(df)
    assert split == engine_split(df)
    assert [len(split[k]) for k in ['train','validation','calibration']] == [60,20,20]
    assert not set(split['train']) & set(split['validation'])
    assert not set(split['train']) & set(split['calibration'])
    assert not set(split['validation']) & set(split['calibration'])


def test_history_features_are_causal_and_slope_is_in_cycles():
    x = np.arange(30, dtype=np.float32).reshape(1,30,1)*2+3
    z = engineer(x)
    np.testing.assert_allclose(z[0], [61,32,np.std(x),2], rtol=1e-6)
    assert engineer(x, 'instant').shape == (1,1)
    assert engineer(x, 'mean_std').shape == (1,3)


def test_loss_direction_and_no_broadcasting():
    loss = AsymmetricMSELoss(5.)
    assert loss(torch.tensor([2.]), torch.tensor([1.])).item() == 5.
    assert loss(torch.tensor([0.]), torch.tensor([1.])).item() == 1.
    with pytest.raises(ValueError):
        loss(torch.zeros(2,1), torch.zeros(2))


def test_cnn_window_shape_is_not_hardcoded():
    for length in [20,30,40]:
        model = RULPredictor(n_features=3, window=length)
        assert model(torch.zeros(2,length,3)).shape == (2,)


def test_conformal_finite_sample_quantile():
    assert conformal_radius(np.arange(1,21)) == 19
    assert np.isinf(conformal_radius(np.array([1.,2.]), alpha=.1))
