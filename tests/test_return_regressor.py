"""P9 ② 收益率回归 —— ReturnRegressor 单元测试.

复用 DirectionModel roundtrip 测试的命门设计：
- bit-exact save/load roundtrip
- save 产出 sidecar JSON
- load 后不触碰 LGBMRegressor 私有 attr（H1 同款守门）
- predict / predict_score_frame 两路一致
- feature_importance save/load 后值相等

回归特化测试：
- predict 输出 float 而非 0/1
- Prediction.value == Prediction.score、proba is None（回归契约）
- fit 校验 X / y 形状 + NaN
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from astock_quant.contracts import Prediction
from astock_quant.models.ret_regression import ReturnRegressor


@pytest.fixture
def fitted_model() -> tuple[ReturnRegressor, pd.DataFrame]:
    """构造训练好的 ReturnRegressor 和对应的 X."""
    np.random.seed(0)
    n = 500
    X = pd.DataFrame(np.random.randn(n, 5), columns=[f"f{i}" for i in range(5)])
    # 合成 label：x0*0.02 + x1*0.005 + 噪声 → 连续 float 收益率
    y = pd.Series(X.iloc[:, 0] * 0.02 + X.iloc[:, 1] * 0.005 + np.random.normal(0, 0.005, n))
    dates = pd.date_range("2025-01-01", periods=n // 5).repeat(5)
    tickers = ["A", "B", "C", "D", "E"] * (n // 5)
    X.index = pd.MultiIndex.from_arrays([dates, tickers], names=["date", "ticker"])
    y.index = X.index

    model = ReturnRegressor(n_estimators=50, verbose=-1)
    model.fit(X, y)
    return model, X


# ===========================================================================
# 回归契约：值域 + Prediction 形状
# ===========================================================================


def test_predict_outputs_float_not_binary(fitted_model):
    """predict 输出 float 收益率，不是 0/1（区别 DirectionModel 的核心）."""
    m, X = fitted_model
    preds = m.predict(X)
    assert len(preds) == len(X)
    assert all(isinstance(p, Prediction) for p in preds)
    vals = np.array([p.value for p in preds])
    # 不应是二值
    assert not set(np.round(vals, 2)).issubset({0.0, 1.0})
    # 应有合理的连续分布
    assert vals.std() > 1e-6


def test_prediction_value_equals_score_proba_none(fitted_model):
    """回归任务的契约：value == score，proba is None."""
    m, X = fitted_model
    preds = m.predict(X)
    for p in preds:
        assert p.score == p.value, f"回归任务 score 应等于 value: {p.score} vs {p.value}"
        assert p.proba is None, f"回归任务 proba 应为 None，实际：{p.proba}"
        assert p.target_type == "return"


def test_predict_and_predict_score_frame_consistent(fitted_model):
    """两条 predict 路径 score 完全一致（共享 booster.predict 结果）."""
    m, X = fitted_model
    preds = m.predict(X)
    frame = m.predict_score_frame(X)
    pred_scores = np.array([p.score for p in preds])
    frame_scores = frame["score"].values
    assert np.allclose(pred_scores, frame_scores, atol=1e-12)
    assert np.allclose(frame["value"].values, frame_scores, atol=1e-12)


# ===========================================================================
# Save / Load roundtrip —— 与 DirectionModel H1 同款命门
# ===========================================================================


def test_save_load_bit_exact_roundtrip(fitted_model):
    """save → load → 同 X 出同 value（精度 1e-9）."""
    m1, X = fitted_model
    preds1 = m1.predict(X)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ret.txt"
        m1.save(path)
        m2 = ReturnRegressor().load(path)
        preds2 = m2.predict(X)

    assert len(preds1) == len(preds2) == len(X)
    for a, b in zip(preds1, preds2, strict=True):
        assert a.ticker == b.ticker and a.date == b.date
        assert abs(a.value - b.value) < 1e-9
        assert a.target_type == b.target_type == "return"


def test_save_creates_sidecar_with_feature_names(fitted_model):
    """save 必须产出 sidecar JSON，内容 = feature_names_."""
    m, _ = fitted_model
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ret.txt"
        m.save(path)

        sidecar = path.with_suffix(path.suffix + ".feature_names.json")
        assert sidecar.exists(), f"sidecar 文件应存在：{sidecar}"
        with sidecar.open("r", encoding="utf-8") as f:
            names = json.load(f)
        assert names == m.feature_names_


def test_load_does_not_touch_private_attrs(fitted_model):
    """命门：load 后 `_reg is None`、`_booster is not None`（与 DirectionModel H1 同款守门）.

    老版本若戳 _reg._Booster / _n_features / _classes 这种私有属性，LightGBM 升级会碎。
    本测试断言 load 路径绕开 sklearn wrapper，predict 直接走 booster.predict（公开 API）。
    """
    m, X = fitted_model
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ret.txt"
        m.save(path)
        m2 = ReturnRegressor().load(path)

    assert m2._reg is None, "load 后 _reg 应为 None（不再依赖 LGBMRegressor 私有属性）"
    assert m2._booster is not None, "load 后 _booster 应被设置"

    # 即使 _reg 为 None，predict 仍然可用
    preds = m2.predict(X)
    assert len(preds) == len(X)


def test_load_falls_back_when_sidecar_missing(fitted_model):
    """sidecar 缺失时回退用 booster.feature_name()，predict 仍可用."""
    m, X = fitted_model
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ret.txt"
        m.save(path)
        sidecar = path.with_suffix(path.suffix + ".feature_names.json")
        sidecar.unlink()
        assert not sidecar.exists()

        m2 = ReturnRegressor().load(path)
        assert m2.feature_names_ == list(m._booster.feature_name())
        preds = m2.predict(X)
        assert len(preds) == len(X)


def test_feature_importance_uses_booster(fitted_model):
    """feature_importance save / load 前后按 name 对齐相等."""
    m, _ = fitted_model
    imp = m.feature_importance()
    assert len(imp) == 5
    assert set(imp.index) == set(m.feature_names_)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ret.txt"
        m.save(path)
        m2 = ReturnRegressor().load(path)
        imp2 = m2.feature_importance()
        for name in m.feature_names_:
            assert imp[name] == imp2[name], f"importance for {name} 不一致"


# ===========================================================================
# fit 校验
# ===========================================================================


def test_fit_rejects_empty_X():
    """空 X 应该抛 ValueError."""
    m = ReturnRegressor(n_estimators=10, verbose=-1)
    X = pd.DataFrame(columns=["f0"])
    y = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="训练集为空"):
        m.fit(X, y)


def test_fit_rejects_xy_length_mismatch():
    m = ReturnRegressor(n_estimators=10, verbose=-1)
    X = pd.DataFrame({"f0": [1.0, 2.0, 3.0]})
    y = pd.Series([0.01, 0.02])  # 长度不一致
    with pytest.raises(ValueError, match="行数不一致"):
        m.fit(X, y)


def test_fit_rejects_nan_in_y():
    """y 含 NaN 必须报错（labels.align_xy 应该上游清掉）."""
    m = ReturnRegressor(n_estimators=10, verbose=-1)
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2025-01-01", periods=3), ["A"]], names=["date", "ticker"]
    )
    X = pd.DataFrame({"f0": [1.0, 2.0, 3.0]}, index=idx)
    y = pd.Series([0.01, np.nan, 0.02], index=idx)
    with pytest.raises(ValueError, match="NaN"):
        m.fit(X, y)


def test_predict_rejects_unfitted_model():
    """未训练的 model.predict 应该抛 RuntimeError."""
    m = ReturnRegressor(n_estimators=10, verbose=-1)
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2025-01-01", periods=2), ["A"]], names=["date", "ticker"]
    )
    X = pd.DataFrame({"f0": [1.0, 2.0]}, index=idx)
    with pytest.raises(RuntimeError, match="模型未训练"):
        m.predict(X)


def test_predict_rejects_missing_feature_columns(fitted_model):
    """predict 时 X 缺少训练时的因子列应该报错."""
    m, X = fitted_model
    X_missing = X.drop(columns=["f0"])  # 故意丢掉一列
    with pytest.raises(ValueError, match="X 缺少训练时的因子列"):
        m.predict(X_missing)


def test_predict_score_frame_empty_X():
    """空 X 时 predict_score_frame 返回空 DataFrame."""
    m = ReturnRegressor(n_estimators=10, verbose=-1)
    # 不需要 fit，预测空就直接返回
    out = m.predict_score_frame(pd.DataFrame())
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 0
