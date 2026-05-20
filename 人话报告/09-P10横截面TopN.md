# 人话报告 09 —— P10 ③ 横截面 Top N：从「打分」到「排名」+ 三人协作首演

> 给你（项目主人）看的「通俗讲解版」。延续报告 01-08 的风格和标准。
> 对应技术文档：`审核/P10-横截面TopN-审核.md`（auditor PASS）+ 三人协作的代码改动。那一份你可以不看，看这篇就够。
> explainer 出品 · 2026-05-16

---

## 一、Stage 3 走到 ③

报告 08 末尾讲过 Stage 3 总共要做 ② return / ③ ranking / ④ trade_signal 三件事。**报告 08 = ②，本报告 = ③，还剩 ④**：

```
Stage 3：① direction（已交付）→ [② return P9 ✅] → [③ ranking P10 ✅] → [④ trade_signal P11]
                                                       ▲▲▲▲ 本报告
```

这次 ③ 是 **Stage 3 设计文档列为最高风险的一站**——技术上有个特别阴险的坑（横截面 look-ahead），团队同时也是**第一次三人协作做一个阶段**。两条线一起讲。

---

## 二、③ 横截面 Top N 是什么——班里 50 人考试，只猜前 5 名是谁

### 2.1 跟 ① / ② 的区别

报告 08 讲过 ① vs ② 的区别——**硬币正反 vs 骰子点数**。这次 ③ 又是另一种玩法：

| 任务 | 类比 | 关键问题 |
|---|---|---|
| ① direction（涨跌方向）| 猜硬币正反 | 这只票明天涨还是跌？ |
| ② return（收益率回归）| 猜骰子点数 | 这只票明天涨多少？ |
| **③ ranking（横截面 Top N）** | **班里前 5 名是谁** | **30 只票里，今天哪 5 只最值得买？** |

用最直观的类比：

> **③ ranking = 班里 50 人考试，不要求你预测每个人考多少分，只要求你猜每天的「前 5 名是谁」**。
>
> 区别在哪：
> - **① / ②**：每只票**单独打分**——茅台明天涨/跌、涨多少。30 只票 30 个独立预测。
> - **③**：30 只票**横向比较**——「今天这 30 只里，哪 5 只最好？」**不关心每只票绝对值多少，只关心它们之间的相对排名**。

### 2.2 为什么要做 ③

打个比方：

> 你不需要知道班里第 6 名考了 89 分还是 91 分——**你只需要知道他不在前 5**。
>
> 同样道理，做实战策略时，你不一定关心「茅台明天涨 0.5% 还是 1.2%」——**你关心「在 30 只候选票里，茅台今天是不是最该买的那几只」**。

排名信息直接对应"该不该买"的决策——比单只票预测幅度更接近实战需求。报告 01 当年说 4 类目标"渐进式覆盖复杂度"，③ 就是从"单票预测"升级到"组合选股"的关键一步。

---

## 三、最高风险：横截面 rank look-ahead

这是 Stage 3 设计文档里 architect-2 明确列为 **最高风险** 的事。Stage 3 启动前他专门预警了一段——P10 这次终于把这个坑守住了。技术词比较抽象，必须讲透。

### 3.1 类比讲清楚问题

> **类比：今天给班里同学排名，不能用下周成绩参与排名**。
>
> 6 月 1 日要给班里同学打个"今天的排名"。你只能用 5 月 31 日及之前的成绩——**绝对不能把 6 月 5 日下周的考试成绩也拉进来一起算名次**。
>
> 听起来理所当然，但**代码层面有种特别阴险的写法会不小心做这件事**：

具体到我们的项目：

```python
# ❌ 错误写法（全样本一起算排名）
future_ret = return_label(panel, horizon=5)   # 计算每只票未来 5 天涨幅
ranks = future_ret.rank(pct=True)              # ← 致命：对全 panel（2022-2026 所有日期）一起算排名
```

这看起来无害，但**「`.rank(pct=True)` 全样本」=「拿 2026 年的票和 2022 年的票一起排名」**——等于让模型看到了未来的横截面分布。回测会假性漂亮，实盘立刻崩。

### 3.2 正确写法

```python
# ✅ 正确写法（按日期分组排名）
future_ret = return_label(panel, horizon=5)
ranks = future_ret.groupby(level="date").rank(pct=True)
#                  ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 关键：先按日期分组
```

意思是：「先按日期把数据分成 1000 多组（每个交易日一组），**每组内部 30 只票算排名**——不同日期之间绝不混在一起」。

### 3.3 命门测试守住这条铁律

光说"我们用了正确写法"不够。Stage 1 那次 winsorize bug（报告 03 讲过）就是看似无害的写法把未来信息渗透回来。所以这次 P10 写了**专门的对抗性测试**：

```
test_ranking_label_no_full_sample_rank：
  1. 构造一份 10 天的数据
  2. 跑一遍：只用前 5 天，得到前 5 天每天的 rank
  3. 跑一遍：用全 10 天，再取前 5 天每天的 rank
  4. 断言：两次的前 5 天 rank 必须完全相同
```

**为什么这能守住**：
- 如果用**正确写法**（按日期 groupby）—— 前 5 天每天的排名只受当天 30 只票影响，跟后 5 天有没有数据完全无关 → 两次断言相等 ✅
- 如果用**错误写法**（全样本 rank）—— 后 5 天数据进来会改变全样本分布 → 前 5 天每只票的"百分位排名"必变 → 两次断言不等 → 测试当场 CI 红 ❌

类比：

> 这就像在给班里学生排名的代码里装个"防作弊摄像头"——一旦有人改回去用下周成绩参与排名，摄像头立刻报警。

### 3.4 三道独立守门

P10 ranking_label 一共有 **3 道独立守门** 防这个坑：

| 第几道 | 守在哪 | 做什么 |
|---|---|---|
| 第 1 道 | 代码本身 | 用正确写法 `groupby(level="date").rank(pct=True)` |
| 第 2 道 | docstring 警告 | 注释里明确列出 2 种禁忌写法，让 reviewer 一眼看见 |
| 第 3 道 | 命门测试 | 上面那个对抗性测试 |

这跟 Stage 1 三道防 look-ahead 防线（数据层 / 切分层 / 回测层）思想完全一致——**一道防线再可靠也不够，必须独立多层防御**。

---

## 四、三人协作首演：考试出题人不能是监考老师

这次 P10 是项目第一次**三人协作做一个阶段**。这件事值得专门讲。

### 4.1 为什么要三人协作

报告 05 / 07 都讲过 Stage 1 / Stage 2 时，code-reviewer 和 verifier 跟代码作者是分开的——**作者写代码、审查另有其人**。这次 P10 把这条原则推到极致：

> **类比：考试**。
> - **出题老师** 出试卷
> - **监考老师** 监考
> - **阅卷老师** 改卷
>
> 这三人**不能是同一个人**——否则出题老师故意出他改过的题，监考时还能放水。
>
> 软件工程同款：**写代码的人不写自己的测试**。否则他会下意识写"我知道我代码能过的测试"——回避真正的边界。

### 4.2 P10 的三人分工

| 角色 | 谁干 | 写了什么 |
|---|---|---|
| **写核心算法** | model-engineer（模型工程）| `ranking_label`（横截面排名计算）+ `RankingModel`（LambdaRank 训练） |
| **写基础设施** | factor-integrator（因子集成）| `models/splits.py` 加 `group_by="date"` 参数 + `pipeline/run_ranking.py`（7 步骨架）|
| **写测试 + 复审** | traditional-factor-engineer（传统因子工程）| 3 个新测试文件（含 3 个命门测试）+ `splits.py` 的命门测试 |

**关键点**：
- 横截面 look-ahead 命门测试是 **traditional-factor-engineer 写的**——他独立审视 model-engineer 的代码，构造对抗场景
- model-engineer **没自己写自己代码的核心测试**——避免"我知道我代码能过的测试"
- factor-integrator 写 pipeline，traditional 写 pipeline 测试——同样分离

### 4.3 这次协作的成绩

mtime 实证三人确实独立工作：

```
17:26  model-engineer 写 ranking_label
17:27  model-engineer 写 RankingModel
17:27  factor-integrator 写 splits group_by
17:28  factor-integrator 写 splits 测试
17:31  traditional-factor-engineer 写 ranking_label 测试（含横截面命门）
17:32  traditional-factor-engineer 写 ranking_model 测试
17:34  traditional-factor-engineer 写 pipeline 测试（含 wiring 命门）
17:35  factor-integrator 写 pipeline 新建
```

三人各做各的，**核心命门测试由第三方独立写**——这次"作者审查强制分离"真落地了。

### 4.4 中途 2 个 bug 漂亮 triage

三人协作时不可避免会有"接口不匹配"。这次出了 2 个，**都被命门测试当场抓到**：

#### Bug 1：n_quantiles API 错位

factor-integrator 写 pipeline 时**假设了一个不存在的参数 `n_quantiles`**（以为这个参数能控制分多少档），但 model-engineer 的 `ranking_label` 实际上没这个参数——它返回的是连续百分位（0~1 之间的小数）。

**怎么发现的**：pipeline 测试跑起来直接报错——`TypeError: ranking_label() got an unexpected keyword argument 'n_quantiles'`。

**怎么修的**：factor-integrator 自己把这个参数清掉。auditor 实测 `grep -rn "n_quantiles" astock_quant/ tests/` 现在生产代码 0 残留。

#### Bug 2：datetime key 类型不匹配

traditional 写测试时**用字符串 `"2024-01-01"` 作为日期 key**，但实际代码里日期 key 是 `pd.Timestamp` 对象——dict 找不到。

**怎么发现的**：测试断言挂——预期日期有数据但读出来是 None。

**怎么修的**：traditional 自己把测试里的 key 类型改成 `pd.Timestamp`。

#### 为什么这是好事

这两个 bug **不是漏检**，**是被测试当场抓到的**：

- 如果没有严格测试，n_quantiles 可能跑通（python 动态类型有时容忍这种），datetime key 错位可能给个 silent 0
- 严格测试 + 命门设计 = **bug 被强制 surface 出来**

类比：

> 这跟开车撞栏杆比开车开下悬崖好得多——**撞栏杆只是擦伤，开下悬崖是车毁人亡**。
>
> bug 越早被测试抓到 = 越像撞栏杆。

---

## 五、又是诚信弱基线（跟 ① / ② / Stage 2 LLM 一脉相承）

P10 也跑通了端到端 pipeline。不出意外——**模型在合成数据上还是没 alpha**，跟 ① / ② / Stage 2 LLM 完全一致。

### 5.1 诚信验证：测试没偷偷期望 alpha

auditor 专门 grep 检查了一遍——所有 ranking 测试里**没有任何「IC > 0.05」「年化收益 > 5%」「Sharpe > 1」这种偷偷期望 alpha 的断言**。

实际测试断言只验证：
- 值域约束（`0 ≤ rank ≤ 1`，符合百分位定义）
- 数学恒等关系（用 Spearman 相关系数验证 ranking 跟 return 单调一致，要求 `rho > 0.99`）
- 横截面平均（pct rank 数学期望应该在 0.5 附近）

**没有任何"模型应该赚钱"的断言**。

### 5.2 跟之前 4 次结果一脉相承

| 阶段 | 关键指标 | 诚信解读 |
|---|---|---|
| Stage 1 ① direction | AUC = 0.5131 | 略高于随机 0.5 |
| Stage 2 LLM | importance = 0.0 | 模型完全不看 LLM 列 |
| Stage 3 ② return | R² = -0.0019 / IC = -0.04 | 比常数均值差一点点 |
| **Stage 3 ③ ranking 本期** | rho > 0.99（数学恒等验证）/ 无 alpha 期望断言 | **跟前面一样，没装** |

> **系统继续是「写得对但模型不会赚钱」的状态**。这不是 bug，是 A股 短期方向真的接近随机过程——没做任何提升技巧前的 baseline 该有的样子。

---

## 六、团队事件：又换了一个角色

报告 05 讲过 Stage 2 中段把 factor-engineer 拆成 3 个专职新人。**这次 P10 又出了一件类似的事——auditor 装死**。

### 6.1 怎么发生的

P10 三人协作完代码后，触发 auditor 复审。结果 auditor **25 分钟一言不发**——既不出报告、也不报错、也不汇报进度。lead 数次催进度无回应。

> **名词解释·装死症**：worker 干完任务后不发完工消息，或者派活后长时间无响应。
>
> **类比**：员工开会全程一言不发但工卡显示在岗——人在工位但工作没动。

### 6.2 怎么处理的

用户授权按 **Stage 3 启动时同款模式**——直接 shutdown 旧 auditor，spawn `auditor-2` 替补。auditor-2 接手后 10 分钟内完成完整复审，PASS 通过。

> 这是 Stage 3 第 **2 次** 队员重组：
> - 第 1 次：architect 50 分钟装死 → architect-2 替补（Stage 3 设计 v1.1 PASS，"罕见纪律深度"评价）
> - **第 2 次（本期）**：auditor 25 分钟装死 → auditor-2 替补（P10 复审 PASS）
>
> 加上 Stage 2 中段：factor-engineer 8 次装死 → 拆 3 专职。
>
> **整个项目至今 3 次队员重组**。

### 6.3 教训

报告 05 末尾讲过——**worker 装死 + 沟通失灵也是工程风险**，跟代码 bug 一样需要被工程化处理。这次教训复用：

- **不等死党回应**：超时立刻 spawn 替补（lead 不浪费整个团队的时间）
- **新人 prompt 钉死纪律**：「干完立刻发消息、不发 idle ping 代替进度报告、卡住要直接发事实」
- **加 `progress-reporter`** 自动每 15 分钟汇报 team 状态——主动监控代替被动等

这套机制持续见效：traditional / llm-factor / factor-integrator / architect-2 / auditor-2 / progress-reporter——**6 个新人全部工作正常**。

---

## 七、auditor（auditor-2 接手）第 14 次实战 + PASS

回顾你设的自动质量门，到 P10 已经第 14 次跑：

| 轮次 | 阶段 | 结果 | 谁审 |
|---|---|---|---|
| 第 1 轮 | P3a winsorize | FAIL → 抓 bug | auditor |
| 第 2-13 轮 | P3-P9 全部阶段 | 全 PASS | auditor |
| **第 14 轮** | **P10 ③ ranking** | **PASS** | **auditor-2**（接手） |

auditor-2 做了 7 条核对（**全过**）：

1. **横截面 rank 真守住** ✅ —— grep 全 repo 无任何全样本 `.rank()` 调用
2. **3 个命门测试有效性** ✅ —— 包括横截面、splits group-aware、winsorize per-date
3. **作者审查分离落地** ✅ —— mtime 间隔证明 3 人分工无交叉
4. **scope 严守** ✅ —— 只动 4 代码 + 4 测试 + 1 文档
5. **n_quantiles 清干净** ✅ —— 生产代码 0 残留
6. **诚信弱基线** ✅ —— 测试无偷偷期望 alpha 断言
7. **没引入新 high/critical** ✅ —— 170/170 测试全过

**整个 P10 过程，你又是 0 操心**。换 auditor-2 这件事完全在团队内消化掉了。

---

## 八、接下来：P11 ④ trade_signal —— Stage 3 最后一步

P10 完成 ③，Stage 3 还剩 ④：

```
② return ✅ ──→ ③ ranking ✅ ──→ [④ trade_signal P11] ──→ Stage 3 收尾
                                   ▲▲▲▲ 下一站
```

### 8.1 ④ 跟 ①②③ 的区别

| 任务 | 输出 |
|---|---|
| ① direction | 涨 / 跌（概率）|
| ② return | 涨多少（连续值）|
| ③ ranking | 排名（百分位）|
| **④ trade_signal** | **直接：买 / 卖 / 持有 + 几手 + 啥时候止盈止损** |

简单说，④ 比前 3 个**更接近实战**——前 3 个出"信号"，④ 出"具体操作"。

### 8.2 ④ 的特别复杂之处——路径标注

> **名词解释·路径标注（path-dependent labeling）**：以前的标签都看「终点」（5 天后涨多少）。但 ④ 要看「路径」——这 5 天中间，有没有先涨到止盈线？有没有先跌到止损线？哪个先发生，决定怎么贴标签。
>
> **类比**：开车从 A 到 B，前 3 个目标只关心你最终到没到 B；④ 关心你**中间有没有先撞了护栏**——撞了就拉回去，没撞才到 B。

这块需要 panel 含 `high / low` 列（已确认有），引擎要补**止盈止损触发逻辑**。lead 拍板「收盘价触发 vs 日内最高/最低价触发」（这是个工程取舍）。

### 8.3 节奏

- P11 ④ trade_signal —— 三人协作模式延续（factor-integrator + model-engineer + traditional-factor-engineer）
- Stage 3 收尾 —— code-reviewer 深审 + verifier 端到端再跑一遍
- 然后整个项目（Stage 1 + 2 + 3）就**全部交付完毕**

---

## 九、一页纸总结

- **P10 干了啥**：Stage 3 第二阶段——实装 ③ 横截面 Top N 选股。把"每只票单独打分"升级到"30 只票横向比较挑前 5 名"。
- **③ vs ①②**：
  - **① direction = 猜硬币正反**（涨/跌）
  - **② return = 猜骰子点数**（涨多少）
  - **③ ranking = 班里 50 人考试，只猜前 5 名是谁**（不要求预测每个人考几分）
- **最高风险：横截面 rank look-ahead**（Stage 3 设计列为最高风险）：
  - 类比：**今天给班里同学排名，不能用下周成绩参与排名**
  - 错误写法 `future_ret.rank(pct=True)`（全样本一起 rank）会让未来分布渗透到过去
  - 正确写法 `future_ret.groupby(level="date").rank(pct=True)`（先按日期分组再 rank）
  - **三道独立守门**：代码本身用正确写法 + docstring 列出禁忌写法 + 命门测试用「前 5 天 vs 全 10 天双跑断言相同」对抗性验证
- **三人协作首演**（项目第一次）：
  - **类比：考试出题人 / 监考 / 阅卷 不能是同一人**——否则会下意识写"我知道我代码能过的测试"
  - 分工：model-engineer 写核心算法（ranking_label + RankingModel）/ factor-integrator 写基础设施（splits + pipeline）/ traditional-factor-engineer 写测试 + 复审（含横截面命门）
  - mtime 实证三人独立工作，**核心命门测试由第三方独立写**——作者审查强制分离真落地
- **中途 2 个 bug 漂亮 triage**：
  - n_quantiles API 错位（factor-integrator 假设了不存在的参数）→ 被 pipeline 测试当场抓到 → 自修
  - datetime key 类型不匹配（traditional 测试用字符串当 key 但实际是 pd.Timestamp）→ 被测试断言抓到 → 自修
  - **bug 越早被测试抓到 = 越像撞栏杆而不是开下悬崖**
- **诚信弱基线延续**（跟 ①②③ + LLM 一脉相承）：
  - auditor grep 验证测试**没有任何「IC > 0.05」/「Sharpe > 1」这种偷偷期望 alpha 的断言**
  - 测试只验值域 / 数学恒等关系（rho > 0.99）/ 横截面平均
  - 模型在合成数据上没 alpha 是预期，**跟 AUC=0.5131 / importance=0 / R²=-0.0019 完全一脉相承**
- **团队事件**：auditor 25 分钟装死 → 用户授权 spawn auditor-2 替补，10 分钟内 PASS。**类比：员工开会全程一言不发但工卡显示在岗**。这是 Stage 3 第 2 次队员重组（项目至今总共 3 次），新人 zero-tolerance 模式 + progress-reporter 主动监控持续见效。
- **auditor-2 第 14 次实战 PASS**：7 条核对全过，170/170 pytest 全过，scope 严守，ranking_label 复用 P9 return_label 的 shift 链（数学一致性自动跟上）。
- **下一步 P11 ④ trade_signal**（Stage 3 最后一步）：从「信号」升级到「具体操作」（买/卖/持有 + 几手 + 止盈止损）。技术上要做**路径标注**（不只看终点 5 天后涨多少，要看中间有没有先触及止盈止损线）—— 类比：开车从 A 到 B，要看中间有没有先撞护栏。

---

走到这里，③ ranking 实装完成 + 项目第一次三人协作首演成功。**没有 alpha，但工程纪律 + 团队协作纪律 100% 守住**——这正是这个项目最值得自豪的事：**从 P3a winsorize bug 到现在 P10 横截面命门 + 三人协作分离 + 装死症 zero-tolerance，一路诚信红线 0 次被踩，团队还在自我进化**。下一份报告等 P11 ④ trade_signal 跑完。
