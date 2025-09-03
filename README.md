# 期权套利发现系统

专注于发现期权市场套利机会的量化交易工具集。

## 🎯 核心工具

| 文件 | 用途 | 推荐场景 |
|------|------|----------|
| **`simple_arbitrage_demo.py`** | **入门工具** | 学习套利概念，快速发现机会 |
| **`option_arbitrage_scanner.py`** | **专业工具** | 精确分析，完整套利策略 |
| **`arbitrage_monitor.py`** | **监控系统** | 24/7自动监控套利机会 |

## 🚀 快速开始

```bash
# 激活环境
source venv/bin/activate

# 快速发现套利机会（推荐）
python simple_arbitrage_demo.py

# 单次专业扫描
python arbitrage_monitor.py --single

# 启动实时监控
python arbitrage_monitor.py
```

## 📊 套利策略

1. **定价套利** - 市场价格与理论价格偏差
2. **期权平价** - Put-Call Parity违背
3. **波动率套利** - 隐含波动率异常
4. **时间价值** - 日历价差机会

## ⚙️ 配置

编辑 `arbitrage_config.json` 调整监控参数：
- 扫描间隔
- 偏差阈值 
- 过滤条件

详细说明见 `OPTIONS_ARBITRAGE_GUIDE.md`

## 📈 实际成果

系统已成功发现真实套利机会，如：
- PS2511-P-61000.GFE 认沽期权价格异常高

## 🔧 环境要求

- Python 3.8+
- 依赖：tushare, pandas, numpy, scipy
- 需要配置 TUSHARE_TOKEN