"""P11 ④ 买卖信号 —— TradeSignalModel 单元 + 集成测试.

覆盖：
- predict 输出 value ∈ {-1.0, 0.0, 1.0}（严格枚举）
- score ∈ [0, 1]（max_proba）
- proba is None（contracts 不支持 3 元组）
- predict_score_frame 5 列 + proba 行和 ≈ 1.0
- bit-exact save/load roundtrip（H1 同款命门）
- load 后 _clf is None + _booster is not None
- LabelEncoder 不漂移：{-1,0,1} → {0,1,2} → {-1,0,1}
- fit 校验：空 X / y 含 NaN / y 含非法值 / 长度不一致
- 信号分支：value=+1→buy / value=-1→sell / value=0→hold / strength=score
- signal notes 含 n_buy / n_sell / n_hold

不依赖真实缓存，全部合成数据。
"""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from astock_quant.contracts import Prediction
from astock_quant.models.trade_signal import TradeSignalModel
from astock_quant.signals.generator import SignalGenerator


# ===========================================================================
# helpers
# ===========================================================================

def _make_panel_index(n_dates: int = 80, n_tickers: int = 8) -> pd.MultiIndex:
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    return pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])


def _make_xy(
    n_dates: int = 80,
    n_tickers: int = 8,
    n_features: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """构造 (X, y)，y ∈ {-1.0, 0.0, 1.0}（模拟 trade_signal_label 输出）."""
    idx = _make_panel_index(n_dates=n_dates, n_tickers=n_tickers)
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((len(idx), n_features)),
        index=idx,
        columns=[f"f{i}" for i in range(n_features)],
    )
    # 三类均匀分布
    raw = rng.integers(0, 3, size=len(idx))
    label_map = {0: -1.0, 1: 0.0, 2: 1.0}
    y = pd.Series([label_map[v] for v in raw], index=idx, name="trade_signal_label")
    return X, y


def _train_test_split(X, y, train_frac=0.7):
    all_dates = X.index.get_level_values("date").unique().sort_values()
    n_tr = int(len(all_dates) * train_frac)
    cutoff = all_dates[n_tr - 1]
    tr = X.index.get_level_values("date") <= cutoff
    va = X.index.get_level_values("date") > cutoff
    return X[tr], y[tr], X[va], y[va]


# ===========================================================================
# 基础训练/预测
# ===========================================================================

def test_trade_signal_model_predict_value_domain():
    """predict 输出 value 严格 ∈ {-1.0, 0.0, 1.0}."""
    X, y = _make_xy()
    X_tr, y_tr, X_va, _ = _train_test_split(X, y)
    model = TradeSignalModel()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va)
    assert len(preds) > 0
    allowed = {-1.0, 0.0, 1.0}
    for p in preds:
        assert p.value in allowed, f"value {p.value} 不在 {{-1,0,1}}"


def test_trade_signal_model_predict_score_range():
    """score = max_proba ∈ [0, 1.0]（3 分类置信度）."""
    X, y = _make_xy()
    X_tr, y_tr, X_va, _ = _train_test_split(X, y)
    model = TradeSignalModel()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va)
    for p in preds:
        assert 0.0 <= p.score <= 1.0, f"score {p.score} 超出 [0,1]"


def test_trade_signal_model_predict_proba_is_none():
    """proba is None（contracts 不支持 3 元组，完整概率走 predict_score_frame）."""
    X, y = _make_xy()
    X_tr, y_tr, X_va, _ = _train_test_split(X, y)
    model = TradeSignalModel()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va)
    assert all(p.proba is None for p in preds)


def test_trade_signal_model_predict_target_type():
    """target_type 必须是 'trade_signal'."""
    X, y = _make_xy()
    X_tr, y_tr, X_va, _ = _train_test_split(X, y)
    model = TradeSignalModel()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va)
    assert all(p.target_type == "trade_signal" for p in preds)


# ===========================================================================
# predict_score_frame：5 列 + proba 行和
# ===========================================================================

def test_trade_signal_model_predict_score_frame_columns():
    """predict_score_frame 输出 5 列：value / score / proba_sl / proba_hold / proba_tp."""
    X, y = _make_xy()
    X_tr, y_tr, X_va, _ = _train_test_split(X, y)
    model = TradeSignalModel()
    model.fit(X_tr, y_tr)
    frame = model.predict_score_frame(X_va)
    assert set(frame.columns) == {"value", "score", "proba_sl", "proba_hold", "proba_tp"}
    assert len(frame) == len(X_va)


def test_trade_signal_model_proba_rows_sum_to_one():
    """每行 proba_sl + proba_hold + proba_tp ≈ 1.0."""
    X, y = _make_xy()
    X_tr, y_tr, X_va, _ = _train_test_split(X, y)
    model = TradeSignalModel()
    model.fit(X_tr, y_tr)
    frame = model.predict_score_frame(X_va)
    row_sums = frame[["proba_sl", "proba_hold", "proba_tp"]].sum(axis=1)
    assert (row_sums - 1.0).abs().max() < 1e-6, f"proba 行和偏离 1.0：{(row_sums - 1.0).abs().max()}"


def test_trade_signal_model_score_equals_max_proba():
    """score 列 == max(proba_sl, proba_hold, proba_tp) 每行一致."""
    X, y = _make_xy()
    X_tr, y_tr, X_va, _ = _train_test_split(X, y)
    model = TradeSignalModel()
    model.fit(X_tr, y_tr)
    frame = model.predict_score_frame(X_va)
    expected_score = frame[["proba_sl", "proba_hold", "proba_tp"]].max(axis=1)
    diff = (frame["score"] - expected_score).abs().max()
    assert diff < 1e-9, f"score 不等于 max_proba，最大差 {diff}"


# ===========================================================================
# LabelEncoder 不漂移
# ===========================================================================

def test_trade_signal_model_label_encoder_roundtrip():
    """fit 用 {-1,0,1}，predict 必须反向映回 {-1,0,1}，不能出现 {0,1,2}."""
    X, y = _make_xy(n_dates=60, seed=1)
    X_tr, y_tr, X_va, _ = _train_test_split(X, y)
    model = TradeSignalModel()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va)
    values = {p.value for p in preds}
    # 不能出现 LightGBM 内部标签 {0,1,2}（只允许 {-1,0,1}）
    assert not values.intersection({2.0}), f"发现 LightGBM 内部标签泄漏到 predict 输出：{values}"
    assert values.issubset({-1.0, 0.0, 1.0}), f"predict 输出含非法值：{values}"


def test_trade_signal_model_label_encoder_all_three_classes():
    """训练集含三类，predict 输出也应能覆盖三类（encoder 正常工作）."""
    X, y = _make_xy(n_dates=100, n_tickers=10, seed=3)
    X_tr, y_tr, X_va, _ = _train_test_split(X, y)
    # 确保训练集含三类
    assert set(y_tr.unique()) == {-1.0, 0.0, 1.0}, "训练集应包含三类"
    model = TradeSignalModel()
    model.fit(X_tr, y_tr)
    frame = model.predict_score_frame(X_va)
    # 三列 proba 都有有效值
    for col in ["proba_sl", "proba_hold", "proba_tp"]:
        assert frame[col].notna().all(), f"{col} 含 NaN"
        assert (frame[col] >= 0).all() and (frame[col] <= 1).all(), f"{col} 超出 [0,1]"


# ===========================================================================
# feature_names_ 对齐
# ===========================================================================

def test_trade_signal_model_feature_names_alignment():
    """fit 后 feature_names_ 与 X 列名一致；predict 时缺列应抛错."""
    X, y = _make_xy()
    X_tr, y_tr, X_va, _ = _train_test_split(X, y)
    model = TradeSignalModel()
    model.fit(X_tr, y_tr)
    assert model.feature_names_ == list(X_tr.columns)

    X_missing = X_va.drop(columns=["f0"])
    with pytest.raises(ValueError, match="缺少"):
        model.predict(X_missing)


# ===========================================================================
# save/load bit-exact 命门（H1 同款）
# ===========================================================================

def test_trade_signal_model_save_load_bit_exact():
    """save/load 后 predict 结果 bit-exact."""
    X, y = _make_xy(n_dates=80, n_tickers=10)
    X_tr, y_tr, X_va, _ = _train_test_split(X, y)
    model = TradeSignalModel()
    model.fit(X_tr, y_tr)
    preds_before = model.predict(X_va)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "trade_signal_lgbm.txt"
        model.save(str(path))

        sidecar = path.with_suffix(path.suffix + ".feature_names.json")
        assert sidecar.exists(), "sidecar feature_names.json 应被 save() 创建"
        with sidecar.open() as f:
            saved_names = json.load(f)
        assert saved_names == model.feature_names_

        loaded = TradeSignalModel().load(str(path))
        assert loaded.feature_names_ == model.feature_names_
        preds_after = loaded.predict(X_va)

    assert len(preds_before) == len(preds_after)
    for pb, pa in zip(preds_before, preds_after):
        assert pb.value == pa.value, f"load 后 value 不一致：{pb.value} vs {pa.value}"
        assert abs(pb.score - pa.score) < 1e-9, f"load 后 score 不一致：{pb.score} vs {pa.score}"


def test_trade_signal_model_load_clears_clf():
    """load 后 _clf is None + _booster is not None（H1 私有属性命门）."""
    X, y = _make_xy()
    X_tr, y_tr, X_va, _ = _train_test_split(X, y)
    model = TradeSignalModel()
    model.fit(X_tr, y_tr)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ts_model.txt"
        model.save(str(path))
        loaded = TradeSignalModel().load(str(path))

    assert loaded._clf is None, "_clf 应在 load 后设为 None"
    assert loaded._booster is not None, "_booster 应在 load 后非 None"


def test_trade_signal_model_load_nonexistent_raises():
    """load 不存在的文件 → 抛 FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        TradeSignalModel().load("/nonexistent/path/model.txt")


# ===========================================================================
# fit 校验
# ===========================================================================

def test_trade_signal_model_fit_empty_raises():
    """fit 空 X → 抛 ValueError."""
    X, y = _make_xy(n_dates=20)
    with pytest.raises(ValueError):
        TradeSignalModel().fit(X.iloc[0:0], y.iloc[0:0])


def test_trade_signal_model_fit_y_with_nan_raises():
    """fit y 含 NaN → 抛 ValueError."""
    X, y = _make_xy(n_dates=20)
    y_nan = y.copy().astype(float)
    y_nan.iloc[0] = np.nan
    with pytest.raises(ValueError):
        TradeSignalModel().fit(X, y_nan)


def test_trade_signal_model_fit_y_illegal_value_raises():
    """fit y 含非法值（不在 {-1,0,1}）→ 抛 ValueError."""
    X, y = _make_xy(n_dates=20)
    y_bad = y.copy()
    y_bad.iloc[0] = 2.0  # LightGBM 内部标签泄漏到用户层
    with pytest.raises(ValueError, match="非法值"):
        TradeSignalModel().fit(X, y_bad)


def test_trade_signal_model_fit_xy_length_mismatch_raises():
    """X / y 行数不一致 → 抛 ValueError."""
    X, y = _make_xy(n_dates=20)
    with pytest.raises(ValueError):
        TradeSignalModel().fit(X, y.iloc[:-1])


def test_trade_signal_model_predict_before_fit_raises():
    """未 fit 直接 predict → 抛 RuntimeError."""
    X, _ = _make_xy(n_dates=20)
    with pytest.raises(RuntimeError):
        TradeSignalModel().predict(X)


# ===========================================================================
# 空输入
# ===========================================================================

def test_trade_signal_model_predict_empty_returns_empty():
    """predict 空 DataFrame → 返回空 list."""
    X, y = _make_xy()
    X_tr, y_tr, X_va, _ = _train_test_split(X, y)
    model = TradeSignalModel()
    model.fit(X_tr, y_tr)
    assert model.predict(X_va.iloc[0:0]) == []


# ===========================================================================
# 信号分支测试
# ===========================================================================

def test_signal_trade_signal_buy_sell_hold_mapping():
    """value=+1 → buy / value=-1 → sell / value=0 → hold."""
    preds = [
        Prediction(ticker="A", date=date(2025, 1, 2), target_type="trade_signal", value=1.0, score=0.8),
        Prediction(ticker="B", date=date(2025, 1, 2), target_type="trade_signal", value=-1.0, score=0.7),
        Prediction(ticker="C", date=date(2025, 1, 2), target_type="trade_signal", value=0.0, score=0.5),
    ]
    rep = SignalGenerator().generate(preds)
    actions = {it.ticker: it.action for it in rep.items}
    assert actions["A"] == "buy"
    assert actions["B"] == "sell"
    assert actions["C"] == "hold"


def test_signal_trade_signal_strength_equals_score():
    """strength == score（max_proba），不是硬编码 1.0/0.0。"""
    preds = [
        Prediction(ticker="A", date=date(2025, 1, 2), target_type="trade_signal", value=1.0, score=0.75),
        Prediction(ticker="B", date=date(2025, 1, 2), target_type="trade_signal", value=-1.0, score=0.60),
        Prediction(ticker="C", date=date(2025, 1, 2), target_type="trade_signal", value=0.0, score=0.45),
    ]
    rep = SignalGenerator().generate(preds)
    strengths = {it.ticker: it.strength for it in rep.items}
    assert abs(strengths["A"] - 0.75) < 1e-9, f"A strength 应 0.75，实际 {strengths['A']}"
    assert abs(strengths["B"] - 0.60) < 1e-9, f"B strength 应 0.60，实际 {strengths['B']}"
    assert abs(strengths["C"] - 0.45) < 1e-9, f"C strength 应 0.45，实际 {strengths['C']}"


def test_signal_trade_signal_reason_contains_class_info():
    """reason 含 TP/SL/HOLD 字样 + 置信度数字。"""
    preds = [
        Prediction(ticker="A", date=date(2025, 1, 2), target_type="trade_signal", value=1.0, score=0.80),
        Prediction(ticker="B", date=date(2025, 1, 2), target_type="trade_signal", value=-1.0, score=0.65),
        Prediction(ticker="C", date=date(2025, 1, 2), target_type="trade_signal", value=0.0, score=0.40),
    ]
    rep = SignalGenerator().generate(preds)
    reasons = {it.ticker: it.reason for it in rep.items}
    assert "TP" in reasons["A"], f"buy reason 应含 TP，实际：{reasons['A']}"
    assert "SL" in reasons["B"], f"sell reason 应含 SL，实际：{reasons['B']}"
    assert "HOLD" in reasons["C"], f"hold reason 应含 HOLD，实际：{reasons['C']}"
    # 含置信度数字（格式如 "0.800"）
    assert "0.800" in reasons["A"] or "0.80" in reasons["A"], f"reason 应含置信度，实际：{reasons['A']}"


def test_signal_trade_signal_notes_format():
    """notes 含 target=trade_signal + n_buy / n_sell / n_hold。"""
    preds = [
        Prediction(ticker="A", date=date(2025, 1, 2), target_type="trade_signal", value=1.0, score=0.8),
        Prediction(ticker="B", date=date(2025, 1, 2), target_type="trade_signal", value=-1.0, score=0.7),
        Prediction(ticker="C", date=date(2025, 1, 2), target_type="trade_signal", value=0.0, score=0.5),
        Prediction(ticker="D", date=date(2025, 1, 2), target_type="trade_signal", value=0.0, score=0.4),
    ]
    rep = SignalGenerator().generate(preds)
    assert "target=trade_signal" in rep.notes
    assert "n_buy=1" in rep.notes
    assert "n_sell=1" in rep.notes
    assert "n_hold=2" in rep.notes


def test_signal_trade_signal_target_type():
    """SignalReport.target_type 必须是 'trade_signal'."""
    preds = [
        Prediction(ticker="A", date=date(2025, 1, 2), target_type="trade_signal", value=1.0, score=0.8),
    ]
    rep = SignalGenerator().generate(preds)
    assert rep.target_type == "trade_signal"
