# 部署手册 —— 把每日报告发布到 GitHub Pages

> 这份文档给「未来的你」看。讲清楚怎么把报告网站部署上线、日常怎么跑、出错怎么查。

---

## 这套东西是怎么运转的

```
你的 Mac（每天 8:00 / 16:30 自动跑）
   │
   ├─ 1. 跑模型预测  →  生成今天的 HTML 报告
   │
   ├─ 2. 把报告复制到 docs/reports/，重建首页 index.html
   │
   └─ 3. git push 推到 GitHub
              │
              ▼
        GitHub Pages 自动发布
              │
              ▼
   你打开固定网址  https://<你的用户名>.github.io/astock-quant/
```

打个比方：你的 Mac 是「后厨」每天做菜（跑模型），GitHub Pages 是「橱窗」摆出来给人看。`git push` 就是把做好的菜端到橱窗里。

---

## 一、首次部署（只做一次）

好消息：这台 Mac 上 **`gh`（GitHub 命令行工具）已经装好并登录了**，认证这一关已经过了。所以首次部署只剩 3 步。

### 步骤 1：创建 GitHub 仓库

在项目目录下跑：

```bash
cd "/Users/wujiangjingcai/claude code/量化"
git init -b main
gh repo create astock-quant --public --source=. --remote=origin
```

- `git init -b main` —— 把本地目录变成一个 git 仓库（主分支叫 main）
- `gh repo create` —— 在 GitHub 上建一个**公开**仓库，名字 `astock-quant`，并和本地关联

### 步骤 2：首次提交 + 推送

**先验证敏感文件不会被推上去**（这步很重要）：

```bash
git add .
git status --short | grep -E "\.env$|models/|\.lgb"
```

> 上面这条命令**应该没有任何输出**。如果输出了 `.env` 或 `.lgb` 文件，**停下来**，说明 `.gitignore` 没生效，别往下做。

确认没问题后，提交并推送：

```bash
git commit -m "init: A股 量化预测系统"
git push -u origin main
```

### 步骤 3：打开 GitHub Pages 开关

1. 浏览器打开 `https://github.com/<你的用户名>/astock-quant/settings/pages`
2. **Build and deployment** → Source 选 **Deploy from a branch**
3. Branch 选 **main**，文件夹选 **/docs**，点 **Save**
4. 等 1-2 分钟，页面顶部会出现一行绿色字：
   `Your site is live at https://<你的用户名>.github.io/astock-quant/`

**这个网址就是你以后每天打开的固定网址。** 存进浏览器书签。

---

## 二、验证部署成功

手动跑一次完整流程（不用等到第二天）：

```bash
cd "/Users/wujiangjingcai/claude code/量化"
bash scripts/daily_predict_wrapper.sh
```

跑完你应该看到：
1. Mac 弹通知「今日预测报告已生成」
2. 浏览器自动打开本地报告
3. Mac 再弹一条通知「今日报告已发布到 GitHub Pages」
4. 1-2 分钟后打开你的固定网址，能看到今天的报告

---

## 三、日常运行（全自动，不用管）

部署完成后，`launchd`（Mac 的定时任务系统）每个交易日会自动：
- **早 8:00** 跑一次（看 T+1 信号）
- **下午 16:30** 跑一次（收盘后看 T+2 信号）

跑完自动 push。你只要每天打开固定网址看就行。

> ⚠️ **前提：Mac 当时要开机且联网**。合上盖子睡眠 / 关机的话那个时段就跳过了，等下一个时段。

---

## 四、出错怎么查

### 网页打不开 / 没更新

```bash
# 看最近一次自动运行的日志
tail -100 "/Users/wujiangjingcai/claude code/量化/artifacts/daily_reports/launchd_stdout.log"
```

在输出里找 `[publish]` 开头的行：
- `[publish] OK` —— 推送成功
- `[publish] 不是 git repo` —— 你还没做「首次部署」
- `[publish] 未配 origin remote` —— 步骤 1 没做完
- `[publish] push 失败` —— 网络问题或认证过期，见下

### push 失败 / 认证过期

`gh` 的登录凭证一般长期有效。万一过期，重新登录：

```bash
gh auth login
```

按提示选 GitHub.com → HTTPS → 用浏览器登录即可。

### 报告显示「模型严重退化警告」

这**不是部署的 bug**，是模型本身的问题（数据源 akshare 不稳，详见项目 `progress.md`）。报告还是会正常发布，只是顶部带个红色警告。属于「诚实地告诉你今天结果不可信」。

---

## 五、怎么关闭 / 改私密

### 暂停每天自动跑

```bash
launchctl unload ~/Library/LaunchAgents/com.astock.daily.plist
```

恢复就把 `unload` 换成 `load`。

### 把网站改成私密（不让别人看）

GitHub 免费版的 Pages 只能公开。要私密有两条路：
1. 把仓库设为 Private + 升级 GitHub Pro（$4/月，Pages 就能私密）
2. 仓库直接删掉：`gh repo delete astock-quant`

> ⚠️ 注意：一旦公开 push 过，就算后来删库，搜索引擎/缓存可能还留有副本。介意的话一开始就别公开。

---

## 附：关键文件清单

| 文件 | 作用 |
|---|---|
| `scripts/daily_predict_wrapper.sh` | 每天跑的主脚本（含预测 + 推送）|
| `scripts/build_index.py` | 生成网站首页 `docs/index.html` |
| `docs/index.html` | 网站首页（报告列表）|
| `docs/reports/*.html` | 每天的报告 |
| `.gitignore` | 挡住 `.env` / 模型大文件不进 git |
| `~/Library/LaunchAgents/com.astock.daily.plist` | 定时任务配置 |

---

*A股量化预测学习项目 · 仅供学习研究 · 不构成投资建议*
