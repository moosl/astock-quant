"""预测器基类 —— 4 类预测目标的统一接口.

所有目标的模型（direction / ret_regression / ranking / trade_signal）都实现 BasePredictor，
下游 backtest / signals 面对统一接口，按 target_type 拿到 Prediction。

设计约定（Stage 1 做透 direction，②③④ 留 stub）：
- `fit(X, y, **kwargs)`：X 是因子矩阵 DataFrame（MultiIndex=(date, ticker)、columns=因子名），
  y 是对齐到 X.index 的 Series；额外超参 / eval_set 走 kwargs。
- `predict(X)`：返回 list[Prediction]，每行一个 Prediction 契约对象，按 X.index 顺序
  对齐。下游 backtest / signals 不操心 X 是 DataFrame 还是 numpy。
- `save(path)` / `load(path)`：模型持久化到 `artifacts/`（详见 P1 6.x 节）。LightGBM 用
  `booster_.save_model(path)` / `lgb.Booster(model_file=path)`。

注意：BasePredictor 不规定 target_type —— 子类各自在构造时声明（写到产出的 Prediction
里）。这样同一个 BasePredictor 接口下，direction / return / ranking / trade_signal 子类
对调用方完全无差别，下游按 target_type 分派即可。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from astock_quant.contracts import Prediction


class BasePredictor(ABC):
    """预测器抽象基类。所有 4 类目标的模型实现这套接口。"""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "BasePredictor":
        """训练模型.

        参数：
            X:      因子矩阵 DataFrame，MultiIndex=(date, ticker)，columns=因子名。
                    NaN 由子类决定如何处理（LightGBM 原生支持 NaN，无需填充）。
            y:      标签 Series，索引必须与 X 完全一致（上游用 align_xy 对齐）。
            kwargs: 子类约定的额外参数（如 eval_set / early_stopping_rounds）。

        返回 self（便于链式：`model.fit(X, y).predict(X_valid)`）。
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> list[Prediction]:
        """预测，返回 list[Prediction]，按 X.index 顺序对齐.

        子类实现要点：
        - X.index 必须是 (date, ticker) 的 MultiIndex，输出 Prediction 的 ticker /
          date 字段就从这取。
        - target_type 写当前子类的目标类型字面量。
        - 二分类（direction）的 value=hard label、score=P(涨)、proba=(P跌,P涨)；
          其它目标见 contracts.Prediction docstring。
        """
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """保存模型到磁盘（一般是 artifacts/ 下的 .txt / .pkl）。"""
        ...

    @abstractmethod
    def load(self, path: str | Path) -> "BasePredictor":
        """从磁盘加载模型，返回 self（便于 `model = SomeModel().load(path)`）。"""
        ...
