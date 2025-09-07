# 期权套利交易系统
## Options Arbitrage Trading System

专业的期权套利机会扫描与分析系统，基于真实市场数据进行套利机会发现。

## 🚀 快速启动

```bash
# 1. 激活虚拟环境
source venv/bin/activate

# 2. 启动应用
streamlit run app.py
```

系统将在 http://localhost:8501 启动

## 🎯 核心功能

- **🎲 策略选择**: 支持定价套利、看跌看涨平价、波动率套利等多种策略
- **📊 真实数据**: 基于Tushare API获取真实期权市场数据  
- **⚡ 实时扫描**: 智能缓存系统实现高效数据获取
- **🎨 Web界面**: 直观的Streamlit界面，支持参数配置和结果可视化
- **🔧 模块化**: 插件化架构，支持策略扩展

## 📁 项目结构

```
├── app.py              # Streamlit主应用入口
├── run.py              # 多模式启动器
├── CLAUDE.md           # 开发指导文档
├── src/                # 核心源代码
├── config/             # 配置文件
├── tests/              # 测试代码  
├── archive/            # 开发过程归档文件
└── requirements.txt    # 依赖包
```

## 🔑 环境配置

系统需要Tushare API token，已在`.env`文件中配置：
```
TUSHARE_TOKEN=2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211
```

## 📖 详细文档

- 系统架构和开发指导：`CLAUDE.md`
- 开发过程文档：`archive/development_docs/`
- 测试和验证脚本：`archive/development_files/`