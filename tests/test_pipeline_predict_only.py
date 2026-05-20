"""P12 predict_only 模式命门测试.

守住三条核心不变量：
1. predict_only=True 时 model.fit 调用次数 = 0（不重训）
2. 模型文件缺失时抛 FileNotFoundError，错误信息含「先跑」引导
3. 显式 predict_model_path 覆盖 date-based 路径解析

注意：这些测试 mock 掉 prepare_stage1_data + compute_factor_frame，避免真实拉数据。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from astock_quant.contracts import Prediction


# ---------------------------------------------------------------------------
# 共享 fixtures
# ---------------------------------------------------------------------------

def _make_multiindex(n_dates: int = 5, tickers: list[str] | None = None) -> pd.MultiIndex:
    tickers = tickers or ["000001", "000002", "600519"]
    dates = pd.date_range("2025-01-02", periods=n_dates, freq="B")
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    return idx


def _make_factor_frame(n_dates: int = 5, tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or ["000001", "000002", "600519"]
    idx = _make_multiindex(n_dates, tickers)
    rng = np.random.default_rng(42)
    data = rng.standard_normal((len(idx), 3))
    df = pd.DataFrame(data, index=idx, columns=["ma5_close", "rsi_14", "volume_ratio"])
    return df


def _make_mock_ff(df: pd.DataFrame):
    ff = MagicMock()
    ff.data = df
    ff.shape = df.shape
    ff.factor_names = list(df.columns)
    return ff


def _make_predictions(idx: pd.MultiIndex, values: list[float] | None = None) -> list[Prediction]:
    preds = []
    for i, (dt, tk) in enumerate(idx):
        v = (values[i] if values else 1.0) if i < (len(values) if values else 0) else 0.0
        preds.append(
            Prediction(
                ticker=str(tk),
                date=pd.Timestamp(dt).date(),
                target_type="direction",
                value=float(v),
                score=0.6,
                proba=(0.4, 0.6),
            )
        )
    return preds


def _save_dummy_lgb_model(path: Path, feature_names: list[str]) -> None:
    """실제 LightGBM 모델을 저장 (테스트용 더미 데이터로 학습)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, len(feature_names)))
    y = (rng.standard_normal(100) > 0).astype(int)
    train_data = lgb.Dataset(X, label=y)
    params = {"objective": "binary", "num_leaves": 4, "n_estimators": 5,
              "verbose": -1, "num_iterations": 5}
    booster = lgb.train(params, train_data, num_boost_round=5)
    booster.save_model(str(path))
    sidecar = Path(str(path) + ".feature_names.json")
    sidecar.write_text(json.dumps(feature_names), encoding="utf-8")


def _save_dummy_lgb_model_regression(path: Path, feature_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, len(feature_names)))
    y = rng.standard_normal(100)
    train_data = lgb.Dataset(X, label=y)
    params = {"objective": "regression", "num_leaves": 4, "verbose": -1, "num_iterations": 5}
    booster = lgb.train(params, train_data, num_boost_round=5)
    booster.save_model(str(path))
    sidecar = Path(str(path) + ".feature_names.json")
    sidecar.write_text(json.dumps(feature_names), encoding="utf-8")


def _save_dummy_lgb_model_multiclass(path: Path, feature_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, len(feature_names)))
    y = rng.integers(0, 3, size=120)
    train_data = lgb.Dataset(X, label=y)
    params = {"objective": "multiclass", "num_class": 3, "num_leaves": 4,
              "verbose": -1, "num_iterations": 5}
    booster = lgb.train(params, train_data, num_boost_round=5)
    booster.save_model(str(path))
    sidecar = Path(str(path) + ".feature_names.json")
    sidecar.write_text(json.dumps(feature_names), encoding="utf-8")


# ---------------------------------------------------------------------------
# 命门 1：predict_only=True 不调 fit（run_direction）
# ---------------------------------------------------------------------------

class TestPredictOnlySkipsFit:

    def test_run_direction_predict_only_does_not_call_fit(self, tmp_path):
        """命门：predict_only=True 路径下 DirectionModel.fit 调用次数必须 = 0."""
        feature_names = ["ma5_close", "rsi_14", "volume_ratio"]
        model_path = tmp_path / "models" / "direction_2026-05-16.lgb"
        _save_dummy_lgb_model(model_path, feature_names)

        factor_df = _make_factor_frame()
        mock_ff = _make_mock_ff(factor_df)

        mock_data = {
            "prices": pd.DataFrame(
                {"close": [10.0] * 15},
                index=_make_multiindex(),
            ),
            "moneyflow": None,
            "financials": {},
            "source": MagicMock(get_news=MagicMock(return_value=[])),
        }

        fit_spy = MagicMock(side_effect=AssertionError("predict_only 路径不允许调 fit"))

        with (
            patch("astock_quant.pipeline.run_direction.prepare_stage1_data", return_value=mock_data),
            patch("astock_quant.pipeline.run_direction.compute_factor_frame", return_value=mock_ff),
            patch("astock_quant.models.direction.DirectionModel.fit", fit_spy),
            patch("astock_quant.pipeline.run_direction.DEFAULT_MODELS_DIR", tmp_path / "models"),
        ):
            from astock_quant.pipeline.run_direction import run_direction
            result = run_direction(
                predict_only=True,
                predict_date="2026-05-16",
                verbose=False,
            )

        assert fit_spy.call_count == 0, "predict_only 路径调了 fit，命门失败"
        assert result["metrics"]["mode"] == "predict_only"

    def test_run_return_predict_only_does_not_call_fit(self, tmp_path):
        """run_return predict_only 不调 fit."""
        feature_names = ["ma5_close", "rsi_14", "volume_ratio"]
        model_path = tmp_path / "models" / "return_2026-05-16.lgb"
        _save_dummy_lgb_model_regression(model_path, feature_names)

        factor_df = _make_factor_frame()
        mock_ff = _make_mock_ff(factor_df)
        mock_data = {
            "prices": pd.DataFrame({"close": [10.0] * 15}, index=_make_multiindex()),
            "moneyflow": None,
            "financials": {},
            "source": MagicMock(get_news=MagicMock(return_value=[])),
        }

        fit_spy = MagicMock(side_effect=AssertionError("predict_only 路径不允许调 fit"))

        with (
            patch("astock_quant.pipeline.run_return.prepare_stage1_data", return_value=mock_data),
            patch("astock_quant.pipeline.run_return.compute_factor_frame", return_value=mock_ff),
            patch("astock_quant.models.ret_regression.ReturnRegressor.fit", fit_spy),
            patch("astock_quant.pipeline.run_return.DEFAULT_MODELS_DIR", tmp_path / "models"),
        ):
            from astock_quant.pipeline.run_return import run_return
            result = run_return(
                predict_only=True,
                predict_date="2026-05-16",
                verbose=False,
            )

        assert fit_spy.call_count == 0
        assert result["metrics"]["mode"] == "predict_only"

    def test_run_ranking_predict_only_does_not_call_fit(self, tmp_path):
        """run_ranking predict_only 不调 fit."""
        feature_names = ["ma5_close", "rsi_14", "volume_ratio"]
        model_path = tmp_path / "models" / "ranking_2026-05-16.lgb"
        _save_dummy_lgb_model_regression(model_path, feature_names)

        factor_df = _make_factor_frame()
        mock_ff = _make_mock_ff(factor_df)
        mock_data = {
            "prices": pd.DataFrame({"close": [10.0] * 15}, index=_make_multiindex()),
            "moneyflow": None,
            "financials": {},
            "source": MagicMock(get_news=MagicMock(return_value=[])),
        }

        fit_spy = MagicMock(side_effect=AssertionError("predict_only 路径不允许调 fit"))

        with (
            patch("astock_quant.pipeline.run_ranking.prepare_stage1_data", return_value=mock_data),
            patch("astock_quant.pipeline.run_ranking.compute_factor_frame", return_value=mock_ff),
            patch("astock_quant.models.ranking.RankingModel.fit", fit_spy),
            patch("astock_quant.pipeline.run_ranking.DEFAULT_MODELS_DIR", tmp_path / "models"),
        ):
            from astock_quant.pipeline.run_ranking import run_ranking
            result = run_ranking(
                predict_only=True,
                predict_date="2026-05-16",
                verbose=False,
            )

        assert fit_spy.call_count == 0
        assert result["metrics"]["mode"] == "predict_only"

    def test_run_trade_signal_predict_only_does_not_call_fit(self, tmp_path):
        """run_trade_signal predict_only 不调 fit."""
        feature_names = ["ma5_close", "rsi_14", "volume_ratio"]
        model_path = tmp_path / "models" / "trade_signal_2026-05-16.lgb"
        _save_dummy_lgb_model_multiclass(model_path, feature_names)

        factor_df = _make_factor_frame()
        mock_ff = _make_mock_ff(factor_df)
        mock_data = {
            "prices": pd.DataFrame({"close": [10.0] * 15}, index=_make_multiindex()),
            "moneyflow": None,
            "financials": {},
            "source": MagicMock(get_news=MagicMock(return_value=[])),
        }

        fit_spy = MagicMock(side_effect=AssertionError("predict_only 路径不允许调 fit"))

        with (
            patch("astock_quant.pipeline.run_trade_signal.prepare_stage1_data", return_value=mock_data),
            patch("astock_quant.pipeline.run_trade_signal.compute_factor_frame", return_value=mock_ff),
            patch("astock_quant.models.trade_signal.TradeSignalModel.fit", fit_spy),
            patch("astock_quant.pipeline.run_trade_signal.DEFAULT_MODELS_DIR", tmp_path / "models"),
        ):
            from astock_quant.pipeline.run_trade_signal import run_trade_signal
            result = run_trade_signal(
                predict_only=True,
                predict_date="2026-05-16",
                verbose=False,
            )

        assert fit_spy.call_count == 0
        assert result["metrics"]["mode"] == "predict_only"


# ---------------------------------------------------------------------------
# 命门 2：模型文件缺失时抛 FileNotFoundError，含引导信息
# ---------------------------------------------------------------------------

class TestPredictOnlyMissingModel:

    def _assert_missing_model_error(self, run_fn, predict_date: str = "2099-01-01"):
        with pytest.raises(FileNotFoundError) as exc_info:
            run_fn(predict_only=True, predict_date=predict_date, verbose=False)
        msg = str(exc_info.value)
        assert "predict_only" in msg or "模型文件" in msg or "找不到" in msg, \
            f"错误信息没有说明是 predict_only 模式问题: {msg}"
        assert "先跑" in msg or "训练" in msg or "save" in msg, \
            f"错误信息没有给出「先跑训练」引导: {msg}"

    def test_run_direction_missing_model_raises(self):
        from astock_quant.pipeline.run_direction import run_direction
        self._assert_missing_model_error(run_direction)

    def test_run_return_missing_model_raises(self):
        from astock_quant.pipeline.run_return import run_return
        self._assert_missing_model_error(run_return)

    def test_run_ranking_missing_model_raises(self):
        from astock_quant.pipeline.run_ranking import run_ranking
        self._assert_missing_model_error(run_ranking)

    def test_run_trade_signal_missing_model_raises(self):
        from astock_quant.pipeline.run_trade_signal import run_trade_signal
        self._assert_missing_model_error(run_trade_signal)


# ---------------------------------------------------------------------------
# 命门 3：显式 predict_model_path 覆盖 date-based 解析（run_direction 代表）
# ---------------------------------------------------------------------------

class TestPredictOnlyExplicitModelPath:

    def test_explicit_predict_model_path_overrides_date(self, tmp_path):
        """显式 predict_model_path 优先，不按 date 解析."""
        feature_names = ["ma5_close", "rsi_14", "volume_ratio"]
        # 把模型存到非标准路径
        custom_path = tmp_path / "my_custom_model.lgb"
        _save_dummy_lgb_model(custom_path, feature_names)

        factor_df = _make_factor_frame()
        mock_ff = _make_mock_ff(factor_df)
        mock_data = {
            "prices": pd.DataFrame({"close": [10.0] * 15}, index=_make_multiindex()),
            "moneyflow": None,
            "financials": {},
            "source": MagicMock(get_news=MagicMock(return_value=[])),
        }

        with (
            patch("astock_quant.pipeline.run_direction.prepare_stage1_data", return_value=mock_data),
            patch("astock_quant.pipeline.run_direction.compute_factor_frame", return_value=mock_ff),
        ):
            from astock_quant.pipeline.run_direction import run_direction
            result = run_direction(
                predict_only=True,
                predict_model_path=str(custom_path),
                predict_date="2099-01-01",  # date-based 解析会找不到，但 explicit path 优先
                verbose=False,
            )

        assert str(custom_path) in result["predict_model_path"]
        assert result["metrics"]["mode"] == "predict_only"


# ---------------------------------------------------------------------------
# predict_only 模式返回 dict schema 验证
# ---------------------------------------------------------------------------

class TestPredictOnlyReturnSchema:

    def test_run_direction_predict_only_return_schema(self, tmp_path):
        """predict_only 返回 dict 含必需 keys，metrics["mode"]=="predict_only"."""
        feature_names = ["ma5_close", "rsi_14", "volume_ratio"]
        model_path = tmp_path / "models" / "direction_2026-05-16.lgb"
        _save_dummy_lgb_model(model_path, feature_names)

        factor_df = _make_factor_frame()
        mock_ff = _make_mock_ff(factor_df)
        mock_data = {
            "prices": pd.DataFrame({"close": [10.0] * 15}, index=_make_multiindex()),
            "moneyflow": None,
            "financials": {},
            "source": MagicMock(get_news=MagicMock(return_value=[])),
        }

        with (
            patch("astock_quant.pipeline.run_direction.prepare_stage1_data", return_value=mock_data),
            patch("astock_quant.pipeline.run_direction.compute_factor_frame", return_value=mock_ff),
            patch("astock_quant.pipeline.run_direction.DEFAULT_MODELS_DIR", tmp_path / "models"),
        ):
            from astock_quant.pipeline.run_direction import run_direction
            result = run_direction(
                predict_only=True,
                predict_date="2026-05-16",
                verbose=False,
            )

        for key in ["predictions", "score_frame", "model", "predict_model_path",
                    "metrics", "factor_names"]:
            assert key in result, f"predict_only 返回 dict 缺少 key: {key}"
        assert result["metrics"]["mode"] == "predict_only"
        assert isinstance(result["predictions"], list)
        assert isinstance(result["factor_names"], list)

    def test_run_direction_predict_only_no_backtest_no_signals(self, tmp_path):
        """predict_only 模式不含 backtest / backtest_metrics / signals."""
        feature_names = ["ma5_close", "rsi_14", "volume_ratio"]
        model_path = tmp_path / "models" / "direction_2026-05-16.lgb"
        _save_dummy_lgb_model(model_path, feature_names)

        factor_df = _make_factor_frame()
        mock_ff = _make_mock_ff(factor_df)
        mock_data = {
            "prices": pd.DataFrame({"close": [10.0] * 15}, index=_make_multiindex()),
            "moneyflow": None,
            "financials": {},
            "source": MagicMock(get_news=MagicMock(return_value=[])),
        }

        with (
            patch("astock_quant.pipeline.run_direction.prepare_stage1_data", return_value=mock_data),
            patch("astock_quant.pipeline.run_direction.compute_factor_frame", return_value=mock_ff),
            patch("astock_quant.pipeline.run_direction.DEFAULT_MODELS_DIR", tmp_path / "models"),
        ):
            from astock_quant.pipeline.run_direction import run_direction
            result = run_direction(
                predict_only=True,
                predict_date="2026-05-16",
                verbose=False,
            )

        for forbidden_key in ["backtest", "backtest_metrics", "signals"]:
            assert forbidden_key not in result, \
                f"predict_only 不应含 {forbidden_key}，但返回了"

    def test_run_trade_signal_predict_only_has_buy_predictions(self, tmp_path):
        """trade_signal predict_only 返回 buy_predictions（value=+1 filter）."""
        feature_names = ["ma5_close", "rsi_14", "volume_ratio"]
        model_path = tmp_path / "models" / "trade_signal_2026-05-16.lgb"
        _save_dummy_lgb_model_multiclass(model_path, feature_names)

        factor_df = _make_factor_frame()
        mock_ff = _make_mock_ff(factor_df)
        mock_data = {
            "prices": pd.DataFrame({"close": [10.0] * 15}, index=_make_multiindex()),
            "moneyflow": None,
            "financials": {},
            "source": MagicMock(get_news=MagicMock(return_value=[])),
        }

        with (
            patch("astock_quant.pipeline.run_trade_signal.prepare_stage1_data", return_value=mock_data),
            patch("astock_quant.pipeline.run_trade_signal.compute_factor_frame", return_value=mock_ff),
            patch("astock_quant.pipeline.run_trade_signal.DEFAULT_MODELS_DIR", tmp_path / "models"),
        ):
            from astock_quant.pipeline.run_trade_signal import run_trade_signal
            result = run_trade_signal(
                predict_only=True,
                predict_date="2026-05-16",
                verbose=False,
            )

        assert "buy_predictions" in result
        # buy_predictions 必须全是 value=+1
        for p in result["buy_predictions"]:
            assert p.value == 1.0, f"buy_predictions 含非 TP 预测: {p}"


# ---------------------------------------------------------------------------
# Bug 1 修复：save_model_to 时落 sidecar metadata.json
# ---------------------------------------------------------------------------

def _make_train_mock_data(tickers=None):
    tickers = tickers or ["000001", "000002", "600519"]
    idx = _make_multiindex(n_dates=5, tickers=tickers)
    mock_data = {
        "prices": pd.DataFrame({"close": [10.0] * len(idx)}, index=idx),
        "moneyflow": None,
        "financials": {},
        "source": MagicMock(get_news=MagicMock(return_value=[])),
    }
    return mock_data


class TestSaveCreatesMetadataSidecar:
    """命门：save_model_to 时必须调用 _save_train_metadata，且 sidecar 格式正确。

    直接测 _save_train_metadata 函数（单元），再 spy 验证 run_direction 在 save 时确实调了它。
    """

    @pytest.mark.parametrize("pipeline_name,model_type,whitelist_key", [
        ("run_direction", "direction", "auc"),
        ("run_return", "return", "rmse"),
        ("run_ranking", "ranking", "spearman_corr"),
        ("run_trade_signal", "trade_signal", "accuracy"),
    ])
    def test_save_train_metadata_writes_sidecar(self, tmp_path, pipeline_name, model_type, whitelist_key):
        """_save_train_metadata 直接单元测试：写入 sidecar，含白名单字段 + _saved_at + _model_type。"""
        import importlib
        import json

        save_path = tmp_path / f"{model_type}_test.lgb"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text("dummy", encoding="utf-8")  # 假 model 文件

        mod = importlib.import_module(f"astock_quant.pipeline.{pipeline_name}")
        _save_fn = getattr(mod, "_save_train_metadata")

        # 构造含白名单字段的 metrics
        metrics = {
            whitelist_key: 0.75,
            "train_size": 1000,
            "valid_size": 200,
            "_internal_obj": object(),  # 非白名单，不应写入 JSON
        }
        _save_fn(save_path, metrics)

        sidecar = Path(str(save_path) + ".metadata.json")
        assert sidecar.exists(), f"sidecar 未生成: {sidecar}"

        data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert "_saved_at" in data, "metadata 缺 _saved_at"
        assert "_model_type" in data, "metadata 缺 _model_type"
        assert data["_model_type"] == model_type
        assert whitelist_key in data, f"白名单字段 {whitelist_key} 未写入 sidecar"
        assert "train_size" in data
        assert "_internal_obj" not in data, "非白名单字段 _internal_obj 不应写入"



# ---------------------------------------------------------------------------
# Bug 2 修复：predict_only inference 只取最新一天 × universe
# ---------------------------------------------------------------------------

class TestPredictOnlyFiltersToLatestDayUniverse:
    """命门：predict_only 时 n_predictions == len(universe)，date == panel.max date。"""

    def _run_direction_predict_only_with_panel(
        self, tmp_path, n_dates: int, all_tickers: list[str], universe: list[str]
    ):
        """Helper：构造 n_dates × all_tickers panel，跑 run_direction predict_only。"""
        feature_names = ["ma5_close", "rsi_14", "volume_ratio"]
        model_path = tmp_path / "models" / "direction_2026-05-16.lgb"
        _save_dummy_lgb_model(model_path, feature_names)

        factor_df = _make_factor_frame(n_dates=n_dates, tickers=all_tickers)
        mock_ff = _make_mock_ff(factor_df)
        mock_data = _make_train_mock_data(tickers=all_tickers)

        with (
            patch("astock_quant.pipeline.run_direction.prepare_stage1_data", return_value=mock_data),
            patch("astock_quant.pipeline.run_direction.compute_factor_frame", return_value=mock_ff),
            patch("astock_quant.pipeline.run_direction.DEFAULT_MODELS_DIR", tmp_path / "models"),
        ):
            from astock_quant.pipeline.run_direction import run_direction
            result = run_direction(
                predict_only=True,
                predict_date="2026-05-16",
                universe=universe,
                verbose=False,
            )
        return result

    def test_n_predictions_equals_universe_size(self, tmp_path):
        """5 日 × 5 票 panel，universe=['000001','000002'] → n_predictions == 2。"""
        all_tickers = ["000001", "000002", "600519", "601398", "600036"]
        universe = ["000001", "000002"]
        result = self._run_direction_predict_only_with_panel(
            tmp_path, n_dates=5, all_tickers=all_tickers, universe=universe
        )
        n = result["metrics"]["n_predictions"]
        assert n == len(universe), \
            f"期望 n_predictions=={len(universe)}（universe 大小），实际 {n}"

    def test_predictions_date_is_panel_max_date(self, tmp_path):
        """所有预测的 date 应是 factor panel 里最大的交易日。"""
        all_tickers = ["000001", "000002", "600519"]
        universe = ["000001", "000002", "600519"]
        result = self._run_direction_predict_only_with_panel(
            tmp_path, n_dates=5, all_tickers=all_tickers, universe=universe
        )
        # panel 的最大日期
        expected_max = pd.date_range("2025-01-02", periods=5, freq="B")[-1].date()
        for pred in result["predictions"]:
            assert pred.date == expected_max, \
                f"预测日期 {pred.date} != panel 最大日期 {expected_max}"

    def test_no_universe_filter_uses_all_latest_day_tickers(self, tmp_path):
        """universe=None → 不过滤，最新一天所有 ticker 都出现在预测里。"""
        all_tickers = ["000001", "000002", "600519"]
        result = self._run_direction_predict_only_with_panel(
            tmp_path, n_dates=5, all_tickers=all_tickers, universe=all_tickers
        )
        pred_tickers = {p.ticker for p in result["predictions"]}
        for tk in all_tickers:
            assert tk in pred_tickers, f"ticker {tk} 应在预测里但缺失"


# ---------------------------------------------------------------------------
# Bug 1 修复：predict_only 从 sidecar metadata 透传训练 metrics 到 result["metrics"]
# ---------------------------------------------------------------------------

class TestPredictOnlyLoadsMetadataIntoMetrics:
    """命门：训练 + save → predict_only → result["metrics"] 含训练时的 auc 等字段。"""

    def test_predict_only_loads_metadata_into_metrics(self, tmp_path):
        """手写 sidecar → predict_only → metrics 中含 sidecar 字段。"""
        import json

        feature_names = ["ma5_close", "rsi_14", "volume_ratio"]
        model_path = tmp_path / "models" / "direction_2026-05-16.lgb"
        _save_dummy_lgb_model(model_path, feature_names)

        # 手写 sidecar，模拟训练时写入的 metadata
        sidecar = Path(str(model_path) + ".metadata.json")
        sidecar.write_text(
            json.dumps({
                "auc": 0.72,
                "accuracy": 0.65,
                "train_size": 5000,
                "_saved_at": "2026-05-16T10:00:00",
                "_model_type": "direction",
            }),
            encoding="utf-8",
        )

        factor_df = _make_factor_frame()
        mock_ff = _make_mock_ff(factor_df)
        mock_data = _make_train_mock_data()

        with (
            patch("astock_quant.pipeline.run_direction.prepare_stage1_data", return_value=mock_data),
            patch("astock_quant.pipeline.run_direction.compute_factor_frame", return_value=mock_ff),
            patch("astock_quant.pipeline.run_direction.DEFAULT_MODELS_DIR", tmp_path / "models"),
        ):
            from astock_quant.pipeline.run_direction import run_direction
            result = run_direction(
                predict_only=True,
                predict_date="2026-05-16",
                verbose=False,
            )

        metrics = result["metrics"]
        assert metrics.get("auc") == pytest.approx(0.72), \
            f"predict_only metrics 未透传 auc，实际 metrics={metrics}"
        assert metrics.get("accuracy") == pytest.approx(0.65)
        assert metrics.get("train_size") == 5000

    def test_predict_only_metadata_missing_does_not_raise(self, tmp_path):
        """sidecar 不存在 → predict_only 仍能跑（不抛），metrics 不含 auc 等字段。"""
        feature_names = ["ma5_close", "rsi_14", "volume_ratio"]
        model_path = tmp_path / "models" / "direction_2026-05-16.lgb"
        _save_dummy_lgb_model(model_path, feature_names)
        # 故意不写 sidecar

        factor_df = _make_factor_frame()
        mock_ff = _make_mock_ff(factor_df)
        mock_data = _make_train_mock_data()

        with (
            patch("astock_quant.pipeline.run_direction.prepare_stage1_data", return_value=mock_data),
            patch("astock_quant.pipeline.run_direction.compute_factor_frame", return_value=mock_ff),
            patch("astock_quant.pipeline.run_direction.DEFAULT_MODELS_DIR", tmp_path / "models"),
        ):
            from astock_quant.pipeline.run_direction import run_direction
            result = run_direction(
                predict_only=True,
                predict_date="2026-05-16",
                verbose=False,
            )

        # 不抛异常，但 auc 等字段不应出现在 metrics
        assert result["metrics"]["mode"] == "predict_only"
        assert "auc" not in result["metrics"], \
            "sidecar 缺失时不应有 auc 字段，但 metrics 里出现了"
