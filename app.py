"""
Streamlit主界面应用程序 - 期权套利交易系统

这是期权套利交易系统的主Web界面入口，提供：
- 配置管理界面
- 一键套利扫描
- 实时进度监控
- 结果展示和分析

使用方法:
    streamlit run app.py
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 环境变量设置
os.environ.setdefault('PYTHONPATH', str(project_root))

# 加载.env文件
def load_env_file():
    """加载.env文件中的环境变量"""
    env_file = project_root / '.env'
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# 加载环境变量
load_env_file()

try:
    import streamlit as st
    from src.ui.streamlit_app import TradingSystemUI
    
    # Streamlit页面配置
    st.set_page_config(
        page_title="期权套利交易机会扫描",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 创建并运行主应用
    app = TradingSystemUI()
    app.run()
    
except ImportError as e:
    st.error(f"导入错误: {e}")
    st.error("请确保已安装所有依赖包和正确设置Python环境")
    st.stop()
except Exception as e:
    st.error(f"应用启动失败: {e}")
    st.error("请检查系统配置和依赖")
    st.stop()