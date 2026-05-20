"""③ 横截面 Top N 选股 —— P10 实装.

复用 DirectionModel / ReturnRegressor 的设计哲学（copy-then-modify 纪律，对应
Stage 3 启动设计 §2.2 + §3.1）：
- 同一 BasePredictor 接口（fit / predict / save / load）
- 同样的 Booster.save_model + sidecar JSON 持久化（H1 修复后的全公开 API 路径）
- 同样的 feature_names_ 严格列对齐
- LightGBM 原生支持 NaN，因子层不填充直接喂

与 ReturnRegressor 的差异：
- target_type = "ranking"（不是 "return"）
- y 来自 `labels.ranking_label`：横截面 pct rank ∈ [0, 1]，仍是连续 float
- 训练用 `LGBMRegressor(objective="regression")`，**不是** LGBMRanker
  （Stage 3 §2.2 决策：ranker 需要 group 数组维护，A股 universe 每日不稳定，配置容易写错；
  回归分数排序在横截面任务上效果与 ranker 相当）
- predict 输出预测的横截面分位（float ≈ [0, 1]，但不强制 clip —— 让下游 signal/backtest
  按横截面 rank 取 Top N，绝对值不重要，相对顺序才重要）
- 评估：上层（pipeline）用 spearman_corr / NDCG@K / hit-rate-top-K / 平均收益差（top-K vs bottom-K）

为什么不抽基类（仍 copy-then-modify）：
    项目纪律是「一个 target_type 一个 class，复用骨架不抽继承」—— ranking.py 的命名
    已在 P1 骨架里，Stage 3 把 stub 填实即可。3 个 model 类有 ~80% 代码相似，但每类的
    eval 指标、predict 输出语义、未来 ranker 选型可能分叉，抽基类反而绑死。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from astock_quant.contracts import Prediction
from astock_quant.models.base import BasePredictor

logger = logging.getLogger(__name__)


class RankingModel(BasePredictor):
    """LightGBM 横截面分位回归器 —— ③ Stage 3 P10 实装.

    用法：
        model = RankingModel()
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
        preds = model.predict(X_valid)            # → list[Prediction]，value=预测分位 ≈ [0,1]
        model.save("artifacts/ranking_lgbm.txt")
        loaded = RankingModel().load("artifacts/ranking_lgbm.txt")

    与 DirectionModel / ReturnRegressor 共享：
        - Booster API 持久化（H1 同款）
        - 同样的 fit / predict / save / load 签名
        - 同样的 _booster + sidecar JSON 双文件产物
    """

    target_type = "ranking"

    DEFAULT_PARAMS: dict[str, Any] = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    def __init__(self, **params: Any) -> None:
        """构造。`params` 覆盖 DEFAULT_PARAMS（例：`RankingModel(num_leaves=63)`）。"""
        merged = {**self.DEFAULT_PARAMS, **params}
        self._reg: lgb.LGBMRegressor | None = lgb.LGBMRegressor(objective="regression", **merged)
        # _booster：load 后的预测入口（也指向 fit 完后 _reg.booster_，二者一致）
        self._booster: lgb.Booster | None = None
        self.feature_names_: list[str] = []

    # ------------------------------------------------------------------
    # BasePredictor 接口
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        eval_set: list[tuple[pd.DataFrame, pd.Series]] | None = None,
        early_stopping_rounds: int | None = 30,
        **kwargs: Any,
    ) -> "RankingModel":
        """训练.

        参数：
            X, y:                    训练数据，索引一致（labels.align_xy 已对齐）。
                                     y 来自 `labels.ranking_label`，是横截面 pct rank ∈ [0, 1]。
            eval_set:                [(X_valid, y_valid)] 用于 early stopping / 监控。
            early_stopping_rounds:   连续 N 轮 RMSE 无提升就停。eval_set 为空则忽略。
            kwargs:                  透传给 LGBMRegressor.fit（如 sample_weight）。
        """
        if X is None or len(X) == 0:
            raise ValueError("RankingModel.fit: 训练集为空（X 没有样本）")
        if len(X) != len(y):
            raise ValueError(f"X / y 行数不一致: {len(X)} vs {len(y)}")
        if y.isna().any():
            raise ValueError(
                f"训练集 y 含 {int(y.isna().sum())} 个 NaN —— 请先用 labels.align_xy(drop_label_nan=True)"
            )

        self.feature_names_ = list(X.columns)
        y_float = y.astype(float)

        callbacks = []
        fit_kwargs = dict(kwargs)
        if eval_set:
            eval_set_float = [(xv, yv.astype(float)) for xv, yv in eval_set]
            fit_kwargs["eval_set"] = eval_set_float
            # 横截面回归用 RMSE / L1 监控；spearman / NDCG 由 pipeline 自己算（更贵）
            fit_kwargs["eval_metric"] = ["rmse", "l1"]
            if early_stopping_rounds:
                callbacks.append(
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
                )
                callbacks.append(lgb.log_evaluation(period=0))
        if callbacks:
            fit_kwargs["callbacks"] = callbacks

        logger.info(
            "RankingModel.fit: n_samples=%d, n_features=%d, eval_set=%s, "
            "y stats: mean=%.4f std=%.4f min=%.4f max=%.4f",
            len(X), X.shape[1], "yes" if eval_set else "no",
            float(y_float.mean()), float(y_float.std()),
            float(y_float.min()), float(y_float.max()),
        )
        self._reg.fit(X, y_float, **fit_kwargs)
        # 与 DirectionModel / ReturnRegressor 一致：fit 完成后 hook booster
        self._booster = self._reg.booster_
        return self

    def predict(self, X: pd.DataFrame) -> list[Prediction]:
        """预测，返回 list[Prediction]，按 X.index 顺序对齐.

        每条 Prediction:
            value:  预测的横截面分位（float ≈ [0, 1]，不强制 clip）
            score:  = value（下游 SignalGenerator 按日横截面 rank 取 Top N）
            proba:  None（回归任务无概率）

        注意：模型可能预测出 ≥1 或 ≤0 的值（外推），下游用 score 做横截面排序时 **不影响**
        Top N 选择 —— 相对顺序才重要，绝对值不重要。如果非要 clip 到 [0,1]，应该在 SignalGenerator
        里做（不在模型层做，保持预测原始分布）。
        """
        if X is None or len(X) == 0:
            return []
        if self._booster is None:
            raise RuntimeError("RankingModel.predict: 模型未训练，请先 fit 或 load")

        # 列对齐 —— LightGBM 不在意列顺序，但保持稳定性
        cols = [c for c in self.feature_names_ if c in X.columns]
        if len(cols) != len(self.feature_names_):
            missing = set(self.feature_names_) - set(X.columns)
            raise ValueError(f"predict: X 缺少训练时的因子列: {missing}")
        X_aligned = X[cols]

        pred = np.asarray(self._booster.predict(X_aligned), dtype=float)

        out: list[Prediction] = []
        if isinstance(X.index, pd.MultiIndex):
            try:
                dates = X.index.get_level_values("date")
                tickers = X.index.get_level_values("ticker")
            except KeyError:
                dates = X.index.get_level_values(0)
                tickers = X.index.get_level_values(1)
        else:
            raise ValueError("predict: X.index 必须是 (date, ticker) 的 MultiIndex")

        for i, (dt, tk) in enumerate(zip(dates, tickers, strict=True)):
            v = float(pred[i])
            out.append(
                Prediction(
                    ticker=str(tk),
                    date=pd.Timestamp(dt).date(),
                    target_type="ranking",
                    value=v,
                    score=v,  # 横截面排序按 score 降序取 Top N，value 本身就连续可比
                    proba=None,
                )
            )
        return out

    def predict_score_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """便利方法：返回索引与 X 一致的 DataFrame，列 = ['value', 'score'].

        与 ReturnRegressor 同款，便于「不需要 Pydantic 对象、直接做横截面 rank 分析」的场景。
        """
        if X is None or len(X) == 0:
            return pd.DataFrame(
                index=X.index if X is not None else None,
                columns=["value", "score"],
            )
        if self._booster is None:
            raise RuntimeError("predict_score_frame: 模型未训练")
        X_aligned = X[[c for c in self.feature_names_ if c in X.columns]]
        pred = np.asarray(self._booster.predict(X_aligned), dtype=float)
        return pd.DataFrame(
            {"value": pred, "score": pred},
            index=X.index,
        )

    # ------------------------------------------------------------------
    # 持久化 —— 全公开 API（与 DirectionModel / ReturnRegressor H1 修复模式一致）
    # ------------------------------------------------------------------

    @staticmethod
    def _sidecar_path(path: Path) -> Path:
        """模型文件对应的 feature_names 边车 JSON 路径."""
        return path.with_suffix(path.suffix + ".feature_names.json")

    def save(self, path: str | Path) -> None:
        """保存 LightGBM Booster + sidecar feature_names JSON.

        产物（与 DirectionModel / ReturnRegressor 同款）：
            - `<path>`：LightGBM 文本格式 booster
            - `<path>.feature_names.json`：训练时的 feature_names_（list[str]）
        """
        if self._booster is None or not self.feature_names_:
            raise RuntimeError("save: 模型未训练，无可保存")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(path))
        with self._sidecar_path(path).open("w", encoding="utf-8") as f:
            json.dump(self.feature_names_, f, ensure_ascii=False, indent=2)

    def load(self, path: str | Path) -> "RankingModel":
        """从磁盘加载 Booster + sidecar feature_names —— 全公开 API.

        加载后状态：
            - `self._booster`：可用的 lgb.Booster（独立对象，predict 直接走它）
            - `self._reg = None`：不再依赖 LGBMRegressor wrapper
            - `self.feature_names_`：从 sidecar JSON 读回

        sidecar 文件缺失时回退用 booster 内置 feature_name + warn。
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")
        self._booster = lgb.Booster(model_file=str(path))
        self._reg = None  # load 后不再用 sklearn wrapper

        sidecar = self._sidecar_path(path)
        if sidecar.exists():
            with sidecar.open("r", encoding="utf-8") as f:
                self.feature_names_ = json.load(f)
        else:
            logger.warning(
                "load: sidecar 文件 %s 不存在，回退用 booster 内置 feature_name —— "
                "建议用本类的 save() 重新保存以生成 sidecar", sidecar,
            )
            self.feature_names_ = list(self._booster.feature_name())
        return self

    # ------------------------------------------------------------------
    # 学习辅助：特征重要性
    # ------------------------------------------------------------------

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """因子重要性 —— 学习时看哪些因子在起作用.

        importance_type='gain'（默认）：累计分裂带来的损失下降。
        importance_type='split'：作为分裂特征的次数。
        """
        if self._booster is None or not self.feature_names_:
            raise RuntimeError("feature_importance: 模型未训练")
        imp = self._booster.feature_importance(importance_type=importance_type)
        return pd.Series(imp, index=self.feature_names_).sort_values(ascending=False)


__all__ = ["RankingModel"]
