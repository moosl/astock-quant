# progress.md — A股 量化预测系统

> Lead 维护。每阶段产物落盘。
>
> **🎊 Stage 4 全收 — A股 量化预测系统真的能用了**
>
> 4 阶段全交付 / 326 测试 / 11 人话报告 / 4 收尾说明 / 30+ 审核 / 6 个新人 0 装死 / 5 次队员重组 / 诚信红线 0 次踩 / 每天 30 秒出预测报告

---

## 🚀 Stage 6 进行中 — 报告网页化 + GitHub Pages 部署（2026-05-20 启动）

**用户决策**：网页公开访问，本地训练保留。

### 团队 `astock-web-deploy`（5 人，2026-05-20 组）
| name | 中文 | 状态 | 任务 |
|---|---|---|---|
| team-lead | Lead/调度 | active | 协调 + 维护 progress.md |
| architect | 架构设计 | **active**（in_progress Task #1）| 设计部署架构 → `.debug/web-deploy-architecture.md` |
| executor | 执行 | idle (BLOCKED on #1) | docs/index.html + wrapper.sh push 集成 |
| security-reviewer | 安全审查 | idle (BLOCKED on #2) | 审 .env / API key 不进 git |
| verifier | 验证 | idle (BLOCKED on #3) | 端到端实跑 push + Pages serve |
| writer | 文档 | idle (BLOCKED on #1+#4) | README + DEPLOY.md |

### 流水线
```
architect → executor → security-reviewer → verifier
              ↓
            writer（要 architect + verifier 产出）
```

### Stage 6 关键路径产物（恢复优先读）
- `.debug/web-deploy-architecture.md`（architect）
- `量化/docs/index.html` + `量化/docs/reports/`（executor）
- `量化/scripts/daily_predict_wrapper.sh`（executor 改造）
- `量化/.gitignore`（executor 更新）
- `.debug/security-review-deploy.md`（security-reviewer）
- `.debug/verify-deploy.md`（verifier）
- `量化/docs/README.md` + `量化/DEPLOY.md`（writer）

### Stage 5 衔接（已完成，遗留 Stage 6 之前）
- P22 drop-NaN 修复：`compute_factor_frame` 默认 drop ≥95% NaN 列，n_features 26→13
- 全量测试 381/381 PASS（含 P22 5 个新测试 + 1 个更新契约的旧测试）
- 5/19 模型 4 个全真训：direction/return/ranking 退化（量价因子对 5 日二分类无 alpha），trade_signal 健康 (1.1MB)
- 退化警告 banner 在 5/19 报告顶部正确显示

---

## 团队 `astock-quant`（12 人，2026-05-16 重组 + 加 progress-reporter）

| 成员 | 角色 | 状态 |
|---|---|---|
| document-specialist | 文档调研 | idle |
| ~~architect~~ | ~~架构设计~~ | shutdown ❌ Stage 3 规划 50 分钟装死 |
| **architect-2** | **架构设计替补**（新）| Stage 4 设计 ✅ 400 行 / P12-P15（idle 待 auditor）|
| ~~data-engineer~~ | ~~数据工程~~ | shutdown ❌ P15 15 分钟零响应 |
| **data-engineer-2** | **数据工程替补**（新）| P15 ✅ 4 文件 + 烟测（idle）|
| **traditional-factor-engineer** | **传统因子工程**（新）| P12 测试 + 复审（待 daily.py 通知接力）|
| **llm-factor-engineer** | **LLM 因子工程**（新）| M1 防线 ✅ + 顺手清 news_fetcher 容错测试（idle）|
| **factor-integrator** | **因子集成**（新）| P12 ✅ renderer.py + HTML/MD 模板（idle）|
| ~~model-engineer~~ | ~~模型工程~~ | shutdown ❌ P12 bug fix 30 分钟装死 |
| **model-engineer-2** | **模型工程替补**（新）| P12 2 bug fix 接手 |
| code-reviewer | 代码审查 | P8 总审 ✅ Conditional PASS（idle） |
| verifier-2 | 验证 | P8 端到端 PASS ✅（idle）|
| explainer | 科普讲解 | 11 报告 + 4 收尾说明 ✅（idle 待 Stage 5）|
| ~~auditor~~ | ~~审核~~ | shutdown 前其实完工 P10 PASS（消息延迟）|
| **auditor-2** | **审核替补**（新）| P12-P15 复审全 PASS ✅（idle）|
| **progress-reporter** | **进度汇报员**（新）| 每 15 分钟自动汇报 team 状态给用户（cron `7-59/15 * * * *`）|

> ~~factor-engineer~~ 已 shutdown — 8 次装死症（干完不发消息 / idle 心跳代替汇报 / 过夜不干活），用户 2026-05-16 决定拆 3。

## Stage 1 全部已交付 ✅（详见 `量化/人话报告/05-Stage1收官.md`）

- 9 模块项目骨架 / 62 测试全过 / ruff clean / 6 技术文档 / 5 人话报告 / 12 审核记录
- 关键指标：AUC=0.5131 / 训练 25283 / 验证 5780 / 回测 193d / 三道防 look-ahead 立住 / 诚信红线 0 次被踩
- Stage 2 prep 4 项债务清理 PASS（M4 停牌估值 / M3 transform 性能 / H4 默认 hold / N1 注释）

## 进行中 — Stage 2 P6 LLM 情绪因子模块

factor-engineer 在做（接 P3a 时预留好的 LLM 因子插槽）：
- 研读 `.p0-repos/TradingAgents-astock/` 的 `llm_clients/`（多供应商封装）+ `agents/utils/structured.py`（结构化输出）+ 7 分析师 prompt（A股 语境）
- 实现 `astock_quant/factors/llm_factor.py`（继承 `BaseFactor`，产出 `FactorValue`，与量价因子平级）
- 多供应商可切换封装，**默认 Anthropic Claude**（可通过 env var 切 OpenAI / DeepSeek / Kimi）
- 文本源：复用 P2 的 `get_news`（akshare 个股新闻），Stage 2 后续可扩研报/公告
- 注册到 `factors/registry.py`，下游模型/回测/信号一行不改
- 测试：mock LLM 响应 + 真实小样本端到端
- 产出：`量化/P6-LLM因子.md`

## 下一步
- factor-engineer 完工 → auditor 审 → PASS → explainer 出报告 06（人话版 Stage 2 P6） → 发用户
- → P7 对比验证（加 LLM 因子前后回测对比，verifier）
- → P8 审查 + 验证（code-reviewer + verifier）
- → Stage 2 收尾

## 关键产物路径
- 计划：`~/.claude/plans/zazzy-wondering-tiger.md`
- 项目根：`量化/`
- 人话报告：`量化/人话报告/`（01-05 ✅）
- 审核记录：`量化/审核/`（12 份 ✅）
- Stage 1 收尾：`量化/Stage1-收尾说明.md` ✅
- Stage 2 准备：`量化/Stage2-启动准备.md` ✅
- P6 LLM 因子：`量化/P6-LLM因子.md`（产出中）
