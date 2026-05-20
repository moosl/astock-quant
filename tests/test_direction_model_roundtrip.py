"""DirectionModel.save/load 公开 API roundtrip 测试 —— P5 reviewer H1 修复守门.

H1 修复前 `load()` 戳了 `LGBMClassifier._Booster / _n_features / _classes` 私有属性，
LightGBM 升级会碎。修复后改用 `Booster.save_model` + sidecar JSON，全公开 API。

本测试守住「save → load → 同 X 出同 prediction」的 bit-exact 不变量，
任何回退到老路径都会直接挂在这。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from astock_quant.contracts import Prediction
from astock_quant.models.direction import DirectionModel


@pytest.fixture
def fitted_model() -> tuple[DirectionModel, pd.DataFrame]:
    """构造一个训练好的 DirectionModel 和对应的 X."""
    np.random.seed(0)
    n = 500
    X = pd.DataFrame(np.random.randn(n, 5), columns=[f"f{i}" for i in range(5)])
    # 简单可分的合成 label：第 0 列 + 0.5 × 第 1 列 > 0 时标 1
    y = pd.Series((X.iloc[:, 0] + 0.5 * X.iloc[:, 1] > 0).astype(int))
    # 构造 MultiIndex=(date, ticker)
    dates = pd.date_range("2025-01-01", periods=n // 5).repeat(5)
    tickers = ["A", "B", "C", "D", "E"] * (n // 5)
    X.index = pd.MultiIndex.from_arrays([dates, tickers], names=["date", "ticker"])
    y.index = X.index

    model = DirectionModel(n_estimators=50, verbose=-1)
    model.fit(X, y)
    return model, X


def test_save_load_bit_exact_roundtrip(fitted_model):
    """save → load → 同 X 出同 score（bit-exact）."""
    m1, X = fitted_model
    preds1 = m1.predict(X)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "model.txt"
        m1.save(path)
        # 加载到新实例
        m2 = DirectionModel().load(path)
        preds2 = m2.predict(X)

    assert len(preds1) == len(preds2) == len(X)
    for a, b in zip(preds1, preds2, strict=True):
        assert isinstance(a, Prediction) and isinstance(b, Prediction)
        assert a.ticker == b.ticker and a.date == b.date
        assert abs(a.score - b.score) < 1e-9, f"score mismatch: {a.score} vs {b.score}"
        assert a.value == b.value


def test_save_creates_sidecar_with_feature_names(fitted_model):
    """save 必须产出 sidecar JSON，内容 = feature_names_."""
    m, _ = fitted_model
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "model.txt"
        m.save(path)

        sidecar = path.with_suffix(path.suffix + ".feature_names.json")
        assert sidecar.exists(), f"sidecar 文件应存在：{sidecar}"
        with sidecar.open("r", encoding="utf-8") as f:
            names = json.load(f)
        assert names == m.feature_names_


def test_load_does_not_touch_private_attrs(fitted_model):
    """关键不变量：load 后不应该挂 _clf._Booster 这种私有属性（H1 修复点）.

    老实现的痕迹是 self._clf._Booster = booster；新实现应该是 self._clf = None
    或不依赖 _clf 的任何私有属性来做预测。
    """
    m, X = fitted_model
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "model.txt"
        m.save(path)
        m2 = DirectionModel().load(path)

    # 修复后 load 把 _clf 置 None，predict 走 _booster
    assert m2._clf is None, "load 后 _clf 应为 None（H1 修复：不再依赖 LGBMClassifier 私有属性）"
    assert m2._booster is not None, "load 后 _booster 应被设置"

    # 即使 _clf 为 None，predict 仍然可用
    preds = m2.predict(X)
    assert len(preds) == len(X)


def test_load_falls_back_when_sidecar_missing(fitted_model):
    """sidecar 缺失时回退用 booster.feature_name()（兼容老存档，但应 warn）."""
    m, X = fitted_model
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "model.txt"
        m.save(path)
        # 删掉 sidecar
        sidecar = path.with_suffix(path.suffix + ".feature_names.json")
        sidecar.unlink()
        assert not sidecar.exists()

        m2 = DirectionModel().load(path)
        # 即使没 sidecar，predict 仍然可用，feature_names_ 从 booster 取回
        assert m2.feature_names_ == list(m._booster.feature_name())
        preds = m2.predict(X)
        assert len(preds) == len(X)


def test_predict_and_predict_score_frame_consistent(fitted_model):
    """两条 predict 路径（list[Prediction] vs DataFrame）输出 score 一致."""
    m, X = fitted_model
    preds = m.predict(X)
    frame = m.predict_score_frame(X)
    pred_scores = np.array([p.score for p in preds])
    frame_scores = frame["score"].values
    assert np.allclose(pred_scores, frame_scores, atol=1e-12)


def test_feature_importance_uses_booster(fitted_model):
    """feature_importance 应该走 booster.feature_importance —— save → load → 仍可用.

    注意 feature_importance 内部按值降序排，所以 imp.index 顺序与 feature_names_ 不一定相同。
    我们只断言：覆盖所有 feature 且 save/load 前后值完全一致。
    """
    m, _ = fitted_model
    imp = m.feature_importance()
    assert len(imp) == 5
    assert set(imp.index) == set(m.feature_names_)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "model.txt"
        m.save(path)
        m2 = DirectionModel().load(path)
        imp2 = m2.feature_importance()
        # gain importance 在同 booster 上必然一致 —— 按 feature name 对齐后值相等
        for name in m.feature_names_:
            assert imp[name] == imp2[name], (
                f"feature_importance for {name} 不一致：save 前 {imp[name]} vs load 后 {imp2[name]}"
            )
