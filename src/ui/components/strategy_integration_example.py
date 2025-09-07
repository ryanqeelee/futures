"""
策略配置系统集成示例

展示如何在现有的Streamlit应用中集成新的多策略参数配置系统
"""

import streamlit as st
from typing import Dict, List, Optional, Any

from ...config.models import StrategyType, StrategyConfig
from ...config.strategy_parameter_manager import StrategyParameterManager
from .enhanced_strategy_config import EnhancedStrategyConfigPanel


def integrate_enhanced_strategy_config():
    """
    集成增强策略配置到现有UI的示例函数
    
    这个函数展示了如何替换现有的简单策略选择逻辑
    """
    
    # 初始化组件
    @st.cache_resource
    def get_parameter_manager():
        return StrategyParameterManager()
    
    parameter_manager = get_parameter_manager()
    config_panel = EnhancedStrategyConfigPanel(parameter_manager)
    
    # 渲染增强的策略配置面板
    selected_strategies, strategy_configs = config_panel.render()
    
    # 显示配置摘要
    if strategy_configs:
        config_panel.render_configuration_summary(strategy_configs)
    
    return selected_strategies, strategy_configs


def replace_existing_strategy_selection():
    """
    展示如何替换现有的策略选择逻辑
    
    在现有的streamlit_app.py中，替换以下代码块：
    
    原代码：
    ```python
    # 策略选择
    with st.expander("🎲 策略选择", expanded=True):
        selected_strategies = []
        if st.checkbox("定价套利", value=True, help="通过理论价格与市场价格偏差获利"):
            selected_strategies.append(StrategyType.PRICING_ARBITRAGE)
        # ... 其他策略选择
    ```
    
    新代码：
    ```python
    # 使用增强的策略配置面板
    selected_strategies, strategy_configs = integrate_enhanced_strategy_config()
    ```
    """
    
    st.header("🔄 系统升级说明")
    st.write("""
    ### 原有策略选择系统的问题：
    
    1. **参数配置限制**: 所有策略共享相同的基础参数（利润阈值、风险容忍度）
    2. **缺乏预设配置**: 用户需要手动配置所有参数，学习成本高
    3. **界面复杂度**: 多个策略同时选择时，界面混乱
    4. **参数验证不足**: 缺乏参数有效性检查和约束
    5. **配置管理困难**: 无法保存和加载配置方案
    
    ### 新系统的优势：
    
    1. **分层配置模式**: 
       - 预设模式：快速开始，适合新手
       - 自定义模式：平衡易用性和灵活性
       - 专家模式：完全自定义，适合高级用户
    
    2. **策略独立配置**:
       - 每个策略有独立的参数空间
       - 策略特定的参数定义和验证
       - 预设配置模板
    
    3. **智能参数管理**:
       - 参数约束和验证
       - 分类显示（基础/高级参数）
       - 实时参数检查
    
    4. **配置持久化**:
       - 保存自定义配置
       - 加载历史配置
       - 配置版本管理
    """)


def demonstrate_migration_path():
    """演示系统迁移路径"""
    
    st.header("🚀 系统迁移指南")
    
    st.subheader("步骤1: 更新配置模型")
    st.code("""
# 在 src/config/models.py 中添加新的配置模型
from .models import ParameterDefinition, StrategyParameterSet
    """)
    
    st.subheader("步骤2: 初始化参数管理器")
    st.code("""
# 在应用初始化时添加
from ...config.strategy_parameter_manager import StrategyParameterManager

self.parameter_manager = StrategyParameterManager()
    """)
    
    st.subheader("步骤3: 替换策略选择UI")
    st.code("""
# 替换原有的策略选择代码
from .components.enhanced_strategy_config import EnhancedStrategyConfigPanel

config_panel = EnhancedStrategyConfigPanel(self.parameter_manager)
selected_strategies, strategy_configs = config_panel.render()
    """)
    
    st.subheader("步骤4: 更新扫描逻辑")
    st.code("""
# 更新套利扫描调用，传入详细的策略配置
async def _run_arbitrage_scan(self, strategy_configs: Dict[StrategyType, StrategyConfig]):
    for strategy_type, config in strategy_configs.items():
        if config.enabled:
            # 使用 config.parameters 中的策略特定参数
            # 而不是全局的 min_profit_threshold 等参数
            pass
    """)
    
    st.subheader("步骤5: 配置向后兼容")
    st.code("""
# 提供向后兼容的配置转换
def convert_legacy_config(old_config):
    new_configs = {}
    for strategy_type in old_config.selected_strategies:
        config = StrategyConfig(
            type=strategy_type,
            min_profit_threshold=old_config.min_profit,
            max_risk_tolerance=old_config.max_risk,
            parameters={}  # 使用默认参数
        )
        new_configs[strategy_type] = config
    return new_configs
    """)


def show_configuration_examples():
    """展示配置示例"""
    
    st.header("📋 配置示例")
    
    tabs = st.tabs(["预设配置", "自定义配置", "专家配置"])
    
    with tabs[0]:
        st.subheader("预设配置示例")
        st.write("用户选择预设配置类型，系统自动填充所有参数：")
        
        st.json({
            "strategy_type": "pricing_arbitrage",
            "preset_name": "平衡型",
            "parameters": {
                "price_deviation_threshold": 2.0,
                "min_volume_threshold": 50,
                "volatility_range": "0.1,0.8"
            },
            "min_profit_threshold": 0.01,
            "max_risk_tolerance": 0.10
        })
    
    with tabs[1]:
        st.subheader("自定义配置示例")
        st.write("用户在预设基础上调整关键参数：")
        
        st.json({
            "strategy_type": "volatility_arbitrage", 
            "custom_config": True,
            "parameters": {
                "volatility_spread_threshold": 2.5,  # 用户调整
                "lookback_period": 15,               # 用户调整
                "delta_neutral": True                # 保持默认
            },
            "min_profit_threshold": 0.015,          # 用户调整
            "max_risk_tolerance": 0.12              # 用户调整
        })
    
    with tabs[2]:
        st.subheader("专家配置示例")
        st.write("完全自定义所有参数，包括高级参数：")
        
        st.json({
            "strategy_type": "prediction_based",
            "custom_config": True,
            "parameters": {
                "prediction_confidence_threshold": 65.0,
                "prediction_horizon": 7,
                "model_type": "ensemble",
                "feature_importance_threshold": 0.03,  # 高级参数
                "max_position_ratio": 0.25,           # 高级参数
                "stop_loss_threshold": 0.10,          # 高级参数
                "reinforcement_learning": False,      # 实验性参数
                "adaptive_parameters": True           # 实验性参数
            },
            "min_profit_threshold": 0.02,
            "max_risk_tolerance": 0.15
        })


if __name__ == "__main__":
    st.set_page_config(page_title="策略配置系统升级", layout="wide")
    
    st.title("🎯 多策略参数配置系统")
    st.write("增强的期权套利策略配置管理系统")
    
    menu = st.sidebar.selectbox(
        "选择查看内容",
        ["系统集成示例", "升级说明", "迁移指南", "配置示例"]
    )
    
    if menu == "系统集成示例":
        integrate_enhanced_strategy_config()
    elif menu == "升级说明":
        replace_existing_strategy_selection()
    elif menu == "迁移指南":
        demonstrate_migration_path()
    elif menu == "配置示例":
        show_configuration_examples()