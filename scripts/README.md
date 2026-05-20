# scripts/ — A股每日预测脚本

## 每日自动运行（launchd，macOS）

### 首次安装

**第 1 步：编辑 plist，替换项目路径**

```bash
# 复制模板
cp scripts/com.astock.daily.plist.template /tmp/com.astock.daily.plist

# 把 {project_root} 替换为你的项目绝对路径
sed -i '' 's|{project_root}|/Users/yourname/claude code/量化|g' /tmp/com.astock.daily.plist
```

**第 2 步：复制到 LaunchAgents**

```bash
cp /tmp/com.astock.daily.plist ~/Library/LaunchAgents/com.astock.daily.plist
```

**第 3 步：加载**

```bash
launchctl load ~/Library/LaunchAgents/com.astock.daily.plist
```

**第 4 步：验证已注册**

```bash
launchctl list | grep astock
# 应看到：-  0  com.astock.daily
```

**第 5 步：手动触发测试（可选）**

```bash
launchctl start com.astock.daily
# 稍等几秒后检查日志
tail -f artifacts/daily_reports/launchd_stdout.log
```

### 关闭 / 卸载

```bash
# 停止并卸载（不再自动跑）
launchctl unload ~/Library/LaunchAgents/com.astock.daily.plist

# 彻底删除
rm ~/Library/LaunchAgents/com.astock.daily.plist
```

### 调度时间

默认每天 **16:30** 触发（A股 16:00 收盘，留 30 分钟等数据同步）。
修改时间：编辑 plist 里 `StartCalendarInterval` 的 `Hour` / `Minute`，重新 `launchctl unload` + `launchctl load`。

### 日志位置

| 文件 | 说明 |
|------|------|
| `artifacts/daily_reports/launchd_stdout.log` | launchd 捕获的标准输出 |
| `artifacts/daily_reports/launchd_stderr.log` | launchd 捕获的标准错误 |
| `artifacts/daily_reports/wrapper_error_YYYY-MM-DD.log` | wrapper 捕获的 Python 异常 |

---

## 手动运行

```bash
# 跑今天（自动推断交易日）
uv run python -m astock_quant.predict.daily

# 或用 wrapper（含通知）
bash scripts/daily_predict_wrapper.sh
```

---

## 通知说明

成功时弹出 macOS 系统通知"今日预测报告已生成"并自动打开浏览器。
失败时弹出错误通知，详情见 `wrapper_error_YYYY-MM-DD.log`。

通知依赖 `osascript`（macOS 内置），无需额外安装。
