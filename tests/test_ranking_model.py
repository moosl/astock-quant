"""P10 ③ 横截面排序 —— RankingModel 单元 + 集成测试.

覆盖：
- spearman_corr / NDCG@5 / hit-rate / top5-bottom5 计算正确性
- save/load bit-exact（同 DirectionModel H1 命门）
- feature_names_ 对齐防漏接
- predict 输出为 float（连续分数，不是 0/1）
- 空 X、未训练时的错误处理

不依赖真实缓存，全部合成数据。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from astock_quant.models.ranking import RankingModel
from astock_quant.contracts import Prediction


# ===========================================================================
# helpers
# ===========================================================================

def _make_panel_index(n_dates: int = 60, n_tickers: int = 8, seed: int = 0) -> pd.MultiIndex:
    """构造 MultiIndex(date, ticker)."""
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    return pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])


def _make_xy(
    n_dates: int = 60,
    n_tickers: int = 8,
    n_features: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """构造 (X, y)，y 是横截面分位数 label (0~1 float)."""
    idx = _make_panel_index(n_dates=n_dates, n_tickers=n_tickers, seed=seed)
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((len(idx), n_features)),
        index=idx,
        columns=[f"f{i}" for i in range(n_features)],
    )
    # y：每日横截面 pct rank（0~1），模拟 ranking_label 输出
    y_raw = pd.Series(rng.standard_normal(len(idx)), index=idx)
    y = y_raw.groupby(level="date").rank(pct=True)
    y.name = "ranking_label"
    return X, y


def _train_test_split_by_date(
    X: pd.DataFrame, y: pd.Series, train_frac: float = 0.7
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """按日期分割 train/test（时序安全，不随机打乱）."""
    all_dates = X.index.get_level_values("date").unique().sort_values()
    n_train = int(len(all_dates) * train_frac)
    cutoff = all_dates[n_train - 1]
    train_mask = X.index.get_level_values("date") <= cutoff
    test_mask = X.index.get_level_values("date") > cutoff
    return X[train_mask], y[train_mask], X[test_mask], y[test_mask]


# ===========================================================================
# 基础训练/预测
# ===========================================================================

def test_ranking_model_fit_predict_returns_float():
    """predict 输出 list[Prediction]，value 是 float（连续分数，非 0/1）."""
    X, y = _make_xy(n_dates=60, n_tickers=8)
    X_tr, y_tr, X_va, _ = _train_test_split_by_date(X, y)
    model = RankingModel()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va)
    assert len(preds) > 0
    assert all(isinstance(p, Prediction) for p in preds)
    values = [p.value for p in preds]
    # 连续 float，不是二值
    unique_vals = set(values)
    assert len(unique_vals) > 2, "predict 输出应是连续 float，不是二值"
    # target_type 是 "ranking"
    assert all(p.target_type == "ranking" for p in preds)
    # proba 为 None（回归任务）
    assert all(p.proba is None for p in preds)


def test_ranking_model_feature_names_alignment():
    """fit 后 feature_names_ 与 X 列名一致；predict 时缺列应抛错."""
    X, y = _make_xy(n_dates=60)
    X_tr, y_tr, X_va, _ = _train_test_split_by_date(X, y)
    model = RankingModel()
    model.fit(X_tr, y_tr)
    assert model.feature_names_ == list(X_tr.columns)

    # 缺一列 → 抛 ValueError
    X_missing = X_va.drop(columns=["f0"])
    with pytest.raises(ValueError, match="缺少"):
        model.predict(X_missing)


# ===========================================================================
# save/load bit-exact（命门）
# ===========================================================================

def test_ranking_model_save_load_bit_exact():
    """save/load 后 predict 结果 bit-exact（同 DirectionModel H1 命门）.

    这是持久化路径的最基本命门：模型序列化 → 反序列化后，
    对相同输入必须给出完全相同的预测分数。
    如果 load 后有任何数值漂移，说明序列化实现有 bug。
    """
    X, y = _make_xy(n_dates=80, n_tickers=10)
    X_tr, y_tr, X_va, _ = _train_test_split_by_date(X, y)
    model = RankingModel()
    model.fit(X_tr, y_tr)
    preds_before = model.predict(X_va)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ranking_model.txt"
        model.save(str(path))

        # 验证 sidecar JSON 存在
        sidecar = path.with_suffix(path.suffix + ".feature_names.json")
        assert sidecar.exists(), "sidecar feature_names.json 应该被 save() 创建"
        with sidecar.open() as f:
            saved_names = json.load(f)
        assert saved_names == model.feature_names_

        # load
        loaded = RankingModel().load(str(path))
        assert loaded.feature_names_ == model.feature_names_

        preds_after = loaded.predict(X_va)

    # bit-exact
    assert len(preds_before) == len(preds_after)
    for pb, pa in zip(preds_before, preds_after):
        assert pb.ticker == pa.ticker
        assert pb.date == pa.date
        assert pb.value == pa.value, (
            f"load 后 ({pb.ticker}, {pb.date}) value 不一致: {pb.value} vs {pa.value}"
        )


def test_ranking_model_load_nonexistent_raises():
    """load 不存在的文件 → 抛 FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        RankingModel().load("/nonexistent/path/model.txt")


def test_ranking_model_predict_before_fit_raises():
    """未 fit 直接 predict → 抛 RuntimeError."""
    X, _ = _make_xy(n_dates=20)
    with pytest.raises(RuntimeError):
        RankingModel().predict(X)


# ===========================================================================
# 评估指标正确性
# ===========================================================================

def test_ranking_model_spearman_corr():
    """Spearman IC（rank correlation）在合成数据上可以被正确计算.

    训练好模型后，验证集的预测分数与真实 y 的 Spearman rank correlation
    应该是有限的（不是 NaN）且在 [-1, 1] 范围内。
    这个测试不要求模型效果好，只验证 spearmanr 能正常调用 predict 结果。
    """
    X, y = _make_xy(n_dates=80, n_tickers=10)
    X_tr, y_tr, X_va, y_va = _train_test_split_by_date(X, y)
    model = RankingModel()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va)

    pred_values = pd.Series(
        {(p.date, p.ticker): p.value for p in preds},
    )
    pred_values.index = pd.MultiIndex.from_tuples(pred_values.index, names=["date", "ticker"])

    common = pred_values.index.intersection(y_va.index)
    assert len(common) > 0
    rho, pval = spearmanr(pred_values.loc[common].values, y_va.loc[common].values)
    assert not np.isnan(rho), "Spearman IC 是 NaN，说明预测或真值有问题"
    assert -1.0 <= rho <= 1.0


def _compute_ndcg_at_k(y_true_rank: np.ndarray, y_pred_score: np.ndarray, k: int = 5) -> float:
    """计算 NDCG@K（排序指标）.

    y_true_rank：真实横截面 rank（分位值，越高越好）
    y_pred_score：预测分数（越高越好）
    """
    n = len(y_true_rank)
    k = min(k, n)
    # 按预测分数降序排列
    sorted_idx = np.argsort(y_pred_score)[::-1]
    # DCG
    dcg = sum(
        (2 ** y_true_rank[sorted_idx[i]] - 1) / np.log2(i + 2)
        for i in range(k)
    )
    # Ideal DCG（真实 rank 降序）
    ideal_idx = np.argsort(y_true_rank)[::-1]
    idcg = sum(
        (2 ** y_true_rank[ideal_idx[i]] - 1) / np.log2(i + 2)
        for i in range(k)
    )
    if idcg < 1e-12:
        return 0.0
    return dcg / idcg


def test_ranking_model_ndcg5_is_finite():
    """NDCG@5 在合成数据上可以被正确计算且是有限值."""
    X, y = _make_xy(n_dates=80, n_tickers=10)
    X_tr, y_tr, X_va, y_va = _train_test_split_by_date(X, y)
    model = RankingModel()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va)

    pred_map = {(pd.Timestamp(p.date), p.ticker): p.value for p in preds}

    all_dates = y_va.index.get_level_values("date").unique()
    ndcg_scores = []
    for d in all_dates:
        day_true = y_va.xs(d, level="date")
        day_pred = pd.Series({tk: pred_map.get((pd.Timestamp(d), tk), np.nan) for tk in day_true.index})
        valid = day_true.notna() & day_pred.notna()
        if valid.sum() < 2:
            continue
        ndcg = _compute_ndcg_at_k(
            day_true[valid].values, day_pred[valid].values, k=5
        )
        ndcg_scores.append(ndcg)

    assert len(ndcg_scores) > 0
    mean_ndcg = np.mean(ndcg_scores)
    assert 0.0 <= mean_ndcg <= 1.0, f"NDCG@5 超出 [0,1]：{mean_ndcg}"
    assert not np.isnan(mean_ndcg)


def test_ranking_model_hit_rate():
    """hit-rate：预测 Top N 里真实排名靠前（top 50%）的比例.

    hit-rate = |{预测 TopK ∩ 真实 TopK}| / K
    这是横截面选股模型最直观的评估指标之一。
    本测试只验证 hit-rate 可以被计算且在 [0, 1] 内（不要求效果好）。
    """
    X, y = _make_xy(n_dates=80, n_tickers=10)
    X_tr, y_tr, X_va, y_va = _train_test_split_by_date(X, y)
    model = RankingModel()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va)

    pred_map = {(pd.Timestamp(p.date), p.ticker): p.value for p in preds}
    all_dates = y_va.index.get_level_values("date").unique()
    hit_rates = []
    k = 5
    for d in all_dates:
        day_true = y_va.xs(d, level="date").dropna()
        day_pred = pd.Series({
            tk: pred_map.get((pd.Timestamp(d), tk), np.nan) for tk in day_true.index
        }).dropna()
        common = day_true.index.intersection(day_pred.index)
        if len(common) < k:
            continue
        top_k_pred = set(day_pred.loc[common].nlargest(k).index)
        top_k_true = set(day_true.loc[common].nlargest(k).index)
        hit = len(top_k_pred & top_k_true) / k
        hit_rates.append(hit)

    assert len(hit_rates) > 0
    mean_hr = np.mean(hit_rates)
    assert 0.0 <= mean_hr <= 1.0, f"hit-rate 超出 [0,1]：{mean_hr}"


def test_ranking_model_top5_bottom5():
    """top5-bottom5：预测 Top5 的平均真实 rank 应高于 Bottom5.

    这是「方向正确性」的最基本检验。即使模型效果弱，
    在充分数量的测试日期上，Top5 的真实 rank 均值也应该高于 Bottom5 均值。
    （合成随机数据可能不满足，本测试只验证计算逻辑正确，不做效果断言）
    """
    X, y = _make_xy(n_dates=80, n_tickers=10, seed=7)
    X_tr, y_tr, X_va, y_va = _train_test_split_by_date(X, y)
    model = RankingModel()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va)

    pred_map = {(pd.Timestamp(p.date), p.ticker): p.value for p in preds}
    all_dates = y_va.index.get_level_values("date").unique()

    top5_ranks, bottom5_ranks = [], []
    k = 3  # 每日只有 10 只票，用 top3/bottom3
    for d in all_dates:
        day_true = y_va.xs(d, level="date").dropna()
        day_pred = pd.Series({
            tk: pred_map.get((pd.Timestamp(d), tk), np.nan) for tk in day_true.index
        }).dropna()
        common = day_true.index.intersection(day_pred.index)
        if len(common) < 2 * k:
            continue
        top_k = day_pred.loc[common].nlargest(k).index
        bot_k = day_pred.loc[common].nsmallest(k).index
        top5_ranks.append(day_true.loc[top_k].mean())
        bottom5_ranks.append(day_true.loc[bot_k].mean())

    assert len(top5_ranks) > 0, "没有足够的测试日期"
    # 验证计算结果是有效的 float（不是 NaN），不对效果做强断言（合成随机数据）
    assert all(not np.isnan(v) for v in top5_ranks)
    assert all(not np.isnan(v) for v in bottom5_ranks)


# ===========================================================================
# 空输入
# ===========================================================================

def test_ranking_model_predict_empty_x_returns_empty():
    """predict 空 DataFrame → 返回空 list."""
    X, y = _make_xy(n_dates=60)
    X_tr, y_tr, X_va, _ = _train_test_split_by_date(X, y)
    model = RankingModel()
    model.fit(X_tr, y_tr)

    empty_X = X_va.iloc[0:0]
    preds = model.predict(empty_X)
    assert preds == []


def test_ranking_model_fit_empty_raises():
    """fit 空 X → 抛 ValueError."""
    X, y = _make_xy(n_dates=20)
    empty_X = X.iloc[0:0]
    empty_y = y.iloc[0:0]
    with pytest.raises(ValueError):
        RankingModel().fit(empty_X, empty_y)


def test_ranking_model_fit_y_with_nan_raises():
    """fit 时 y 含 NaN → 抛 ValueError（应先用 align_xy drop）."""
    X, y = _make_xy(n_dates=20)
    y_with_nan = y.copy()
    y_with_nan.iloc[0] = np.nan
    with pytest.raises(ValueError):
        RankingModel().fit(X, y_with_nan)
