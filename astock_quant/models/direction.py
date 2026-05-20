"""① 涨跌方向二分类 —— Stage 1 做透.

这是 Stage 1 的完整纵向切片：用 LightGBM 二分类预测「未来 N 日累计收益 > 阈值」。
实现 models/base.py 的 BasePredictor 接口，下游 backtest / signals 面对统一形态。

设计：
- 使用 `lightgbm.LGBMClassifier`（sklearn 风格高级 API）—— 比低层 `lgb.train` 干净。
- **LightGBM 原生支持 NaN**：因子层留下的 NaN 不需要填充，LightGBM 内部把 NaN 作为
  一个独立的分支条件来学。这也是为什么 P3a 因子层「不填充」的纪律可以一路通到模型。
- 默认超参偏保守（n_estimators=300, num_leaves=31, learning_rate=0.05）—— Stage 1
  以「跑通 + 能给出合理 AUC」为目标，不做盲目调参；调参留给后续阶段。
- 特征重要性：fit 完后 `model.feature_importance_` 暴露 gain 重要性（学习用途，便于
  理解哪些因子在起作用）。

持久化（P5 reviewer H1 → 收尾 polish 修复）：
    - 老实现戳 `LGBMClassifier._Booster / _n_features / _classes` 私有属性，LightGBM
      升级时会碎。现改用 LightGBM 公开 API：`Booster.save_model` 落模型文件 +
      sidecar JSON 落 `feature_names_`，`load()` 直接用 `lgb.Booster(model_file=...)`
      重建独立 booster，predict 路径全部走 `booster.predict(X)`（公开 API），不依赖
      sklearn wrapper 的内部状态。
    - 训练后 `self._booster` 与 `self._clf.booster_` 指向同一对象（fit 中 hook），
      load 后只有 `self._booster`（无 `_clf`）—— 两路 predict 接口一致。
    - sidecar 文件路径：`<model_path>.feature_names.json`。

预测输出（Prediction）：
    value:  0.0 / 1.0（按默认阈值 0.5 由 score 离散化得到的硬分类结果）
    score:  P(涨) ∈ [0, 1]
    proba:  (P(跌), P(涨))

下游 backtest / signals 可以直接看 score 做阈值实验，不依赖 value。
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


class DirectionModel(BasePredictor):
    """LightGBM 涨跌方向二分类器 —— Stage 1 做透.

    用法：
        model = DirectionModel()
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
        preds = model.predict(X_valid)            # → list[Prediction]
        model.save("artifacts/direction_lgbm.txt")
        loaded = DirectionModel().load("artifacts/direction_lgbm.txt")
    """

    target_type = "direction"

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
        "verbose": -1,  # 静默
    }

    def __init__(self, **params: Any) -> None:
        """构造。`params` 覆盖 DEFAULT_PARAMS（例：`DirectionModel(num_leaves=63)`）。"""
        merged = {**self.DEFAULT_PARAMS, **params}
        self._clf: lgb.LGBMClassifier | None = lgb.LGBMClassifier(objective="binary", **merged)
        # _booster：load 后的预测入口（也指向 fit 完后 _clf.booster_，二者一致）
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
    ) -> "DirectionModel":
        """训练.

        参数：
            X, y:                    训练数据，必须索引一致（上游用 labels.align_xy 对齐）。
            eval_set:                [(X_valid, y_valid)] 用于 early stopping / 监控。
            early_stopping_rounds:   连续 N 轮验证集无提升就停。eval_set 为空则忽略。
            kwargs:                  透传给 LGBMClassifier.fit（如 sample_weight）。
        """
        if X is None or len(X) == 0:
            raise ValueError("DirectionModel.fit: 训练集为空（X 没有样本）")
        if len(X) != len(y):
            raise ValueError(f"X / y 行数不一致: {len(X)} vs {len(y)}")
        if y.isna().any():
            raise ValueError(
                f"训练集 y 含 {int(y.isna().sum())} 个 NaN —— 请先用 labels.align_xy(drop_label_nan=True)"
            )

        self.feature_names_ = list(X.columns)
        # y 转 0/1 int
        y_int = y.astype(int)

        callbacks = []
        fit_kwargs = dict(kwargs)
        if eval_set:
            eval_set_int = [(xv, yv.astype(int)) for xv, yv in eval_set]
            fit_kwargs["eval_set"] = eval_set_int
            fit_kwargs["eval_metric"] = ["binary_logloss", "auc"]
            if early_stopping_rounds:
                callbacks.append(
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
                )
                callbacks.append(lgb.log_evaluation(period=0))  # 静默 eval 打印
        if callbacks:
            fit_kwargs["callbacks"] = callbacks

        logger.info(
            "DirectionModel.fit: n_samples=%d, n_features=%d, eval_set=%s",
            len(X), X.shape[1], "yes" if eval_set else "no",
        )
        self._clf.fit(X, y_int, **fit_kwargs)
        # fit 完成后把 booster 也挂到 self._booster —— predict / predict_score_frame
        # 统一走 booster.predict（公开 API），与 load 后的 predict 路径一致。
        self._booster = self._clf.booster_

        # Sanity check：检测退化模型（树太少 = early stopping 选了 1 棵树）
        # 不 raise（否则 daily.py 完全跑不出来），而是 logger.warning + 标记 degenerate flag。
        # renderer 会通过 conf_std < 0.02 单独显示「模型严重退化警告」给用户。
        n_trees = self._booster.num_trees()
        self._degenerate = n_trees < 5
        if self._degenerate:
            logger.warning(
                "DirectionModel 退化：best_iteration=%d（< 5 棵树）。"
                "特征几乎无信号，常见原因：因子 100%% NaN（akshare 失败）。"
                "模型仍 save，但 renderer 会显示退化警告。",
                n_trees,
            )

        return self

    def predict(self, X: pd.DataFrame) -> list[Prediction]:
        """预测，返回 list[Prediction]，按 X.index 顺序对齐."""
        if X is None or len(X) == 0:
            return []
        if self._booster is None:
            raise RuntimeError("DirectionModel.predict: 模型未训练，请先 fit 或 load")

        # 列对齐 —— LightGBM 不在意列顺序，但保持稳定性
        cols = [c for c in self.feature_names_ if c in X.columns]
        if len(cols) != len(self.feature_names_):
            missing = set(self.feature_names_) - set(X.columns)
            raise ValueError(f"predict: X 缺少训练时的因子列: {missing}")
        X_aligned = X[cols]

        # 用 booster.predict（公开 API）直出 P(涨) —— 二分类 binary objective 默认返回正类概率。
        score = self._booster.predict(X_aligned)
        score = np.asarray(score, dtype=float)
        proba = np.column_stack([1.0 - score, score])  # (P(跌), P(涨))
        value = (score >= 0.5).astype(float)

        out: list[Prediction] = []
        # MultiIndex=(date, ticker) → 取 date / ticker
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
                    target_type="direction",
                    value=float(value[i]),
                    score=float(score[i]),
                    proba=(float(proba[i, 0]), float(proba[i, 1])),
                )
            )
        return out

    def predict_score_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """便利方法：返回索引与 X 一致的 DataFrame，列 = ['value', 'score', 'proba_down', 'proba_up'].

        用于「不需要 Pydantic 对象、直接喂回测引擎」的场景，比 list[Prediction] 再转回
        DataFrame 高效。和 predict() 走的是同一份 predict_proba 计算。
        """
        if X is None or len(X) == 0:
            return pd.DataFrame(
                index=X.index if X is not None else None,
                columns=["value", "score", "proba_down", "proba_up"],
            )
        if self._booster is None:
            raise RuntimeError("predict_score_frame: 模型未训练")
        X_aligned = X[[c for c in self.feature_names_ if c in X.columns]]
        score = np.asarray(self._booster.predict(X_aligned), dtype=float)
        return pd.DataFrame(
            {
                "value": (score >= 0.5).astype(float),
                "score": score,
                "proba_down": 1.0 - score,
                "proba_up": score,
            },
            index=X.index,
        )

    # ------------------------------------------------------------------
    # 持久化 —— 全公开 API（reviewer H1 修复）
    # ------------------------------------------------------------------

    @staticmethod
    def _sidecar_path(path: Path) -> Path:
        """模型文件对应的 feature_names 边车 JSON 路径."""
        return path.with_suffix(path.suffix + ".feature_names.json")

    def save(self, path: str | Path) -> None:
        """保存 LightGBM Booster + sidecar feature_names JSON.

        产物（两个文件）：
            - `<path>`：LightGBM 文本格式 booster（`Booster.save_model` 公开 API）
            - `<path>.feature_names.json`：训练时的 feature_names_（list[str]）

        放 sidecar 而不是塞进 booster 文件本身：因为 booster 里的 `feature_name()`
        是训练时给的列名 —— 实测可靠，但 H1 修法里把它显式落 JSON 作为「数据契约」，
        让 load 不依赖 booster 内部字段顺序。两份相等即视为一致。
        """
        if self._booster is None or not self.feature_names_:
            raise RuntimeError("save: 模型未训练，无可保存")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(path))
        with self._sidecar_path(path).open("w", encoding="utf-8") as f:
            json.dump(self.feature_names_, f, ensure_ascii=False, indent=2)

    def load(self, path: str | Path) -> "DirectionModel":
        """从磁盘加载 Booster + sidecar feature_names —— 全公开 API.

        加载后状态：
            - `self._booster`：可用的 lgb.Booster（独立对象，predict 直接走它）
            - `self._clf = None`：不再依赖 LGBMClassifier wrapper（H1 修复点）
            - `self.feature_names_`：从 sidecar JSON 读回

        sidecar 文件缺失时回退用 booster 内置的 feature_name（兼容老存档）+ warn。
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")
        self._booster = lgb.Booster(model_file=str(path))
        self._clf = None  # load 后不再用 sklearn wrapper

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

        importance_type='gain'（默认）：累计分裂带来的损失下降，业内常用。
        importance_type='split'：作为分裂特征的次数。
        """
        if self._booster is None or not self.feature_names_:
            raise RuntimeError("feature_importance: 模型未训练")
        imp = self._booster.feature_importance(importance_type=importance_type)
        return pd.Series(imp, index=self.feature_names_).sort_values(ascending=False)


__all__ = ["DirectionModel"]
