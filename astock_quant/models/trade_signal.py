"""④ 买卖信号三分类 —— P11 实装.

复用 DirectionModel 的设计哲学（copy-then-modify 纪律，对应 Stage 3 启动设计 §2.5 + §3.1）：
- 同一 BasePredictor 接口（fit / predict / save / load）
- 同样的 Booster.save_model + sidecar JSON 持久化（H1 修复后的全公开 API 路径）
- 同样的 feature_names_ 严格列对齐
- LightGBM 原生支持 NaN，因子层不填充直接喂

与 DirectionModel 的差异（核心）：
- target_type = "trade_signal"
- y 来自 `labels.trade_signal_label`：3 类 {-1, 0, +1}（SL / HOLD / TP）
- `LGBMClassifier(objective="multiclass", num_class=3)`，**不是** binary
- **内部 LabelEncoder**：用户传 {-1, 0, +1} → LightGBM 训练用 {0, 1, 2}；predict 时反向映回
  原因：LightGBM multiclass 要求标签是非负连续整数 0..num_class-1，但用户接口保留 {-1, 0, +1} 的
  语义化标签（更可读）
- `predict()` 输出 Prediction:
    value:  ∈ {-1.0, 0.0, +1.0}（argmax 类别的语义标签）
    score:  最大类的概率（max_proba），表征模型置信度
    proba:  None（contracts.py 的 `tuple[float, float]` 只支持 2 类，3 分类无法塞）
            完整 proba 矩阵通过 `predict_score_frame()` 拿（含 proba_sl/hold/tp 三列）
- 评估：3 类 accuracy + per-class precision/recall + macro F1（由 pipeline/run_trade_signal 上层算）

为什么不复用 DirectionModel：multiclass 训练 + LabelEncoder 编解码 + 3 类 proba 处理，与
binary 路径分叉点太多，强行继承反而绑死。保持 copy-then-modify 纪律。
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


# ---------------------------------------------------------------------------
# Label 编解码
# ---------------------------------------------------------------------------
#
# 用户层语义标签：{-1.0: SL, 0.0: HOLD, 1.0: TP}（来自 trade_signal_label）
# LightGBM 内部标签：{0: SL, 1: HOLD, 2: TP}
#
# 映射固定（不用 sklearn LabelEncoder 学，避免顺序漂移）：
#     user -1 → lgbm 0
#     user  0 → lgbm 1
#     user +1 → lgbm 2

_USER_TO_LGBM = {-1.0: 0, 0.0: 1, 1.0: 2}
_LGBM_TO_USER = {0: -1.0, 1: 0.0, 2: 1.0}
_CLASS_NAMES = {-1.0: "SL", 0.0: "HOLD", 1.0: "TP"}


class TradeSignalModel(BasePredictor):
    """LightGBM 3 分类买卖信号模型 —— ④ Stage 3 P11 实装.

    用法：
        model = TradeSignalModel()
        # y 是 {-1, 0, +1} float Series（来自 trade_signal_label）
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
        preds = model.predict(X_valid)            # → list[Prediction]，value ∈ {-1, 0, +1}
        frame = model.predict_score_frame(X_valid)  # → DataFrame[value, score, proba_sl, proba_hold, proba_tp]
        model.save("artifacts/trade_signal_lgbm.txt")
        loaded = TradeSignalModel().load("artifacts/trade_signal_lgbm.txt")
    """

    target_type = "trade_signal"

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
        """构造。`params` 覆盖 DEFAULT_PARAMS（例：`TradeSignalModel(num_leaves=63)`）。"""
        merged = {**self.DEFAULT_PARAMS, **params}
        # multiclass + num_class=3 是 ④ 任务的核心配置
        self._clf: lgb.LGBMClassifier | None = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, **merged
        )
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
    ) -> "TradeSignalModel":
        """训练.

        参数：
            X, y:                    训练数据，索引一致。y 来自 trade_signal_label，是 {-1, 0, +1}
                                     float Series。内部用 _USER_TO_LGBM 映成 {0, 1, 2} 喂 LightGBM。
            eval_set:                [(X_valid, y_valid)] 用于 early stopping / 监控（y 也是 {-1,0,+1}）。
            early_stopping_rounds:   连续 N 轮无提升就停。eval_set 为空则忽略。
            kwargs:                  透传给 LGBMClassifier.fit（如 sample_weight）。
        """
        if X is None or len(X) == 0:
            raise ValueError("TradeSignalModel.fit: 训练集为空（X 没有样本）")
        if len(X) != len(y):
            raise ValueError(f"X / y 行数不一致: {len(X)} vs {len(y)}")
        if y.isna().any():
            raise ValueError(
                f"训练集 y 含 {int(y.isna().sum())} 个 NaN —— 请先用 labels.align_xy(drop_label_nan=True)"
            )

        # 校验 y 值域 ⊂ {-1, 0, 1}
        unique_vals = set(y.unique())
        allowed = {-1.0, 0.0, 1.0}
        if not unique_vals.issubset(allowed):
            raise ValueError(
                f"训练集 y 含非法值 {unique_vals - allowed}，trade_signal_label 仅允许 {{-1, 0, 1}}"
            )

        self.feature_names_ = list(X.columns)
        # 用户层 {-1,0,1} → LightGBM {0,1,2}
        y_int = y.map(_USER_TO_LGBM).astype(int)

        callbacks = []
        fit_kwargs = dict(kwargs)
        if eval_set:
            eval_set_int = [(xv, yv.map(_USER_TO_LGBM).astype(int)) for xv, yv in eval_set]
            fit_kwargs["eval_set"] = eval_set_int
            # multiclass 的 eval_metric：multi_logloss（交叉熵）+ multi_error（1 - accuracy）
            fit_kwargs["eval_metric"] = ["multi_logloss", "multi_error"]
            if early_stopping_rounds:
                callbacks.append(
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
                )
                callbacks.append(lgb.log_evaluation(period=0))
        if callbacks:
            fit_kwargs["callbacks"] = callbacks

        # 类别分布日志（看类别失衡情况，下游可能要 class_weight）
        class_dist = y.value_counts().sort_index()
        logger.info(
            "TradeSignalModel.fit: n_samples=%d, n_features=%d, eval_set=%s, "
            "class dist: SL=%d, HOLD=%d, TP=%d",
            len(X), X.shape[1], "yes" if eval_set else "no",
            int(class_dist.get(-1.0, 0)),
            int(class_dist.get(0.0, 0)),
            int(class_dist.get(1.0, 0)),
        )
        self._clf.fit(X, y_int, **fit_kwargs)
        # fit 完后 hook booster —— predict 路径统一走 booster.predict（公开 API），与 load 后一致
        self._booster = self._clf.booster_
        return self

    def predict(self, X: pd.DataFrame) -> list[Prediction]:
        """预测，返回 list[Prediction]，按 X.index 顺序对齐.

        每条 Prediction:
            value:  ∈ {-1.0, 0.0, +1.0}（argmax 类别的语义标签）
            score:  最大类的概率 max_proba ∈ [1/3, 1.0]（模型置信度）
            proba:  None（3 分类无法塞进 contracts.py 的 `tuple[float, float]`；
                    需要完整概率矩阵走 `predict_score_frame`）
        """
        if X is None or len(X) == 0:
            return []
        if self._booster is None:
            raise RuntimeError("TradeSignalModel.predict: 模型未训练，请先 fit 或 load")

        cols = [c for c in self.feature_names_ if c in X.columns]
        if len(cols) != len(self.feature_names_):
            missing = set(self.feature_names_) - set(X.columns)
            raise ValueError(f"predict: X 缺少训练时的因子列: {missing}")
        X_aligned = X[cols]

        # multiclass 时 booster.predict 直出 (n, 3) 概率矩阵
        proba = np.asarray(self._booster.predict(X_aligned), dtype=float)
        if proba.ndim != 2 or proba.shape[1] != 3:
            raise RuntimeError(
                f"booster.predict 输出形状 {proba.shape}，预期 (n, 3) —— 模型不是 3 分类 multiclass？"
            )
        argmax = proba.argmax(axis=1)
        max_proba = proba.max(axis=1)
        # LightGBM {0,1,2} → 用户语义 {-1, 0, +1}
        user_values = np.vectorize(_LGBM_TO_USER.get)(argmax)

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
            out.append(
                Prediction(
                    ticker=str(tk),
                    date=pd.Timestamp(dt).date(),
                    target_type="trade_signal",
                    value=float(user_values[i]),
                    score=float(max_proba[i]),
                    proba=None,  # 3 分类完整概率走 predict_score_frame
                )
            )
        return out

    def predict_score_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """便利方法：返回索引与 X 一致的 DataFrame，列 = ['value', 'score', 'proba_sl', 'proba_hold', 'proba_tp'].

        用于「需要完整 3 类概率矩阵」的场景（如调阈值实验、conformal prediction）。
        和 predict() 走同一份 booster.predict 计算。

        列语义：
            value:      argmax 类别的语义标签 ∈ {-1, 0, +1}
            score:      max_proba（最大类的概率，最高置信度）
            proba_sl:   P(SL)，对应用户标签 -1
            proba_hold: P(HOLD)，对应用户标签 0
            proba_tp:   P(TP)，对应用户标签 +1
        """
        if X is None or len(X) == 0:
            return pd.DataFrame(
                index=X.index if X is not None else None,
                columns=["value", "score", "proba_sl", "proba_hold", "proba_tp"],
            )
        if self._booster is None:
            raise RuntimeError("predict_score_frame: 模型未训练")
        X_aligned = X[[c for c in self.feature_names_ if c in X.columns]]
        proba = np.asarray(self._booster.predict(X_aligned), dtype=float)
        argmax = proba.argmax(axis=1)
        max_proba = proba.max(axis=1)
        user_values = np.vectorize(_LGBM_TO_USER.get)(argmax)
        return pd.DataFrame(
            {
                "value": user_values.astype(float),
                "score": max_proba,
                "proba_sl": proba[:, 0],    # LightGBM 0 → SL
                "proba_hold": proba[:, 1],  # LightGBM 1 → HOLD
                "proba_tp": proba[:, 2],    # LightGBM 2 → TP
            },
            index=X.index,
        )

    # ------------------------------------------------------------------
    # 持久化 —— 全公开 API（与 DirectionModel / ReturnRegressor / RankingModel H1 一致）
    # ------------------------------------------------------------------

    @staticmethod
    def _sidecar_path(path: Path) -> Path:
        """模型文件对应的 feature_names 边车 JSON 路径."""
        return path.with_suffix(path.suffix + ".feature_names.json")

    def save(self, path: str | Path) -> None:
        """保存 LightGBM Booster + sidecar feature_names JSON.

        产物（与其他 3 个 model 类同款）：
            - `<path>`：LightGBM 文本格式 booster（multiclass + num_class=3）
            - `<path>.feature_names.json`：训练时的 feature_names_（list[str]）
        """
        if self._booster is None or not self.feature_names_:
            raise RuntimeError("save: 模型未训练，无可保存")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(path))
        with self._sidecar_path(path).open("w", encoding="utf-8") as f:
            json.dump(self.feature_names_, f, ensure_ascii=False, indent=2)

    def load(self, path: str | Path) -> "TradeSignalModel":
        """从磁盘加载 Booster + sidecar feature_names —— 全公开 API.

        加载后状态：
            - `self._booster`：可用的 lgb.Booster（独立对象，predict 直接走它）
            - `self._clf = None`：不再依赖 LGBMClassifier wrapper
            - `self.feature_names_`：从 sidecar JSON 读回

        sidecar 文件缺失时回退用 booster 内置 feature_name + warn。
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")
        self._booster = lgb.Booster(model_file=str(path))
        self._clf = None

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


__all__ = ["TradeSignalModel", "_CLASS_NAMES"]
