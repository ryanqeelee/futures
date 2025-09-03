#!/bin/bash

# 期权套利交易系统 - Streamlit启动脚本

echo "🚀 启动期权套利交易系统..."

# 激活虚拟环境
source venv/bin/activate

# 检查依赖
echo "📦 检查依赖..."
python -c "
import streamlit
import plotly
import pandas
import numpy
print('✅ 所有依赖已安装')
"

# 检查环境变量
echo "🔧 检查配置..."
if [ -f .env ]; then
    echo "✅ .env文件已找到"
    if grep -q "TUSHARE_TOKEN" .env; then
        echo "✅ TUSHARE_TOKEN已配置"
    else
        echo "❌ TUSHARE_TOKEN未配置，请在.env文件中添加"
        exit 1
    fi
else
    echo "❌ .env文件不存在，请创建.env文件并添加TUSHARE_TOKEN"
    exit 1
fi

# 设置Streamlit配置
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_GLOBAL_DEVELOPMENT_MODE=True

echo "🌐 启动Web界面..."
echo "访问地址: http://localhost:8501"
echo "按 Ctrl+C 停止服务"
echo ""

# 启动Streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0