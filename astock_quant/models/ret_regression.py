"""② 收益率/价格回归 —— P9 实装.

复用 DirectionModel 的设计哲学（copy-then-modify 纪律，对应 Stage 3 启动设计 §2.1）：
- 同一 BasePredictor 接口（fit / predict / save / load）
- 同样的 Booster.save_model + sidecar JSON 持久化（H1 修复后的全公开 API 路径）
- 同样的 feature_names_ 严格列对齐
- LightGBM 原生支持 NaN，因子层不填充直接喂

与 DirectionModel 的差异（核心三处）：
- `LGBMRegressor` 替 `LGBMClassifier`，`objective="regression"`（MSE loss）
- `predict()` 输出 float 收益率，不是 P(涨)；`Prediction.value` 直接是预测收益率，
  `Prediction.score = value`（连续可比，下游 signals/backtest 按阈值过滤），`proba=None`
- 评估指标：训练时 eval_metric=["rmse", "l1"]；上层算 RMSE / MAE / R² / IC / rank-IC

为什么不另开 ReturnModel 类：项目纪律是「一个 target_type 一个 class，复用骨架」——
ret_regression.py 的命名已经摆在 Stage 1 P1 骨架里，Stage 3 把 stub 填实即可。
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


class ReturnRegressor(BasePredictor):
    """LightGBM 收益率回归器 —— ② Stage 3 P9 实装.

    用法：
        model = ReturnRegressor()
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
        preds = model.predict(X_valid)            # → list[Prediction]，value=预测收益率
        model.save("artifacts/return_lgbm.txt")
        loaded = ReturnRegressor().load("artifacts/return_lgbm.txt")
    """

    target_type = "return"

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
        """构造。`params` 覆盖 DEFAULT_PARAMS（例：`ReturnRegressor(num_leaves=63)`）。"""
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
    ) -> "ReturnRegressor":
        """训练.

        参数：
            X, y:                    训练数据，索引一致（labels.align_xy 已对齐）。
                                     y 是 float（未来 horizon 日累计收益率，return_label 的输出）。
            eval_set:                [(X_valid, y_valid)] 用于 early stopping / 监控。
            early_stopping_rounds:   连续 N 轮验证集 RMSE 无提升就停。eval_set 为空则忽略。
            kwargs:                  透传给 LGBMRegressor.fit（如 sample_weight）。
        """
        if X is None or len(X) == 0:
            raise ValueError("ReturnRegressor.fit: 训练集为空（X 没有样本）")
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
            # 回归用 RMSE / L1（MAE），不是 binary_logloss/auc
            fit_kwargs["eval_metric"] = ["rmse", "l1"]
            if early_stopping_rounds:
                callbacks.append(
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
                )
                callbacks.append(lgb.log_evaluation(period=0))
        if callbacks:
            fit_kwargs["callbacks"] = callbacks

        logger.info(
            "ReturnRegressor.fit: n_samples=%d, n_features=%d, eval_set=%s, "
            "y stats: mean=%.4f std=%.4f min=%.4f max=%.4f",
            len(X), X.shape[1], "yes" if eval_set else "no",
            float(y_float.mean()), float(y_float.std()),
            float(y_float.min()), float(y_float.max()),
        )
        self._reg.fit(X, y_float, **fit_kwargs)
        # 与 DirectionModel 一致：fit 完成后 hook booster 到 self._booster
        # 让 predict 路径与 load 后路径一致（公开 API）
        self._booster = self._reg.booster_
        return self

    def predict(self, X: pd.DataFrame) -> list[Prediction]:
        """预测，返回 list[Prediction]，按 X.index 顺序对齐.

        每条 Prediction:
            value:  预测的未来 horizon 日累计收益率（float）
            score:  = value（连续可比，下游按阈值过滤）
            proba:  None（回归任务无概率）
        """
        if X is None or len(X) == 0:
            return []
        if self._booster is None:
            raise RuntimeError("ReturnRegressor.predict: 模型未训练，请先 fit 或 load")

        # 列对齐 —— LightGBM 不在意列顺序，但保持稳定性
        cols = [c for c in self.feature_names_ if c in X.columns]
        if len(cols) != len(self.feature_names_):
            missing = set(self.feature_names_) - set(X.columns)
            raise ValueError(f"predict: X 缺少训练时的因子列: {missing}")
        X_aligned = X[cols]

        # booster.predict 是公开 API，直接出预测收益率
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
                    target_type="return",
                    value=v,
                    score=v,  # 回归任务 value 本身就连续，直接当 score
                    proba=None,
                )
            )
        return out

    def predict_score_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """便利方法：返回索引与 X 一致的 DataFrame，列 = ['value', 'score'].

        用于「不需要 Pydantic 对象、直接喂回测引擎」的场景，比 list[Prediction] 再转回
        DataFrame 高效。和 predict() 走的是同一份 booster.predict 计算。
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
    # 持久化 —— 全公开 API（与 DirectionModel H1 修复模式一致）
    # ------------------------------------------------------------------

    @staticmethod
    def _sidecar_path(path: Path) -> Path:
        """模型文件对应的 feature_names 边车 JSON 路径."""
        return path.with_suffix(path.suffix + ".feature_names.json")

    def save(self, path: str | Path) -> None:
        """保存 LightGBM Booster + sidecar feature_names JSON.

        产物（与 DirectionModel 同款）：
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

    def load(self, path: str | Path) -> "ReturnRegressor":
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


__all__ = ["ReturnRegressor"]
