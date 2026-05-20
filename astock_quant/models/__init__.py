"""模型层 —— 4 类预测目标的核心预测引擎.

BasePredictor (ABC) ← direction / ret_regression / ranking / trade_signal 各自实现
splits：时序安全的数据切分（防 look-ahead，含 purge gap）。

Stage 1 把 ① direction（LightGBM 二分类）做透，②③④ 是实现 BasePredictor 的扩展点 stub。
LSTM 若 P3 要做，作为 direction.py 同级新增文件，同样实现 BasePredictor。
"""
