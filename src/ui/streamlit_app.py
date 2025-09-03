"""
核心Streamlit应用程序 - 期权套利交易系统主界面

提供完整的Web界面功能，包括：
- 系统配置和状态监控
- 一键扫描功能
- 实时进度显示
- 结果分析和可视化
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 导入核心系统组件
from ..engine.arbitrage_engine import ArbitrageEngine, ScanParameters
from ..core.plugin_manager import PluginManager, PluginManagerConfig
from ..core.intelligent_cache_manager import TradingCacheManager
from ..adapters.tushare_adapter import TushareAdapter
from ..config.manager import ConfigManager
from ..config.models import StrategyType, SystemConfig

# 导入UI组件
from .components.config_panel import ConfigPanel
from .components.progress_monitor import ProgressMonitor
from .components.results_display import ResultsDisplay
from .components.enhanced_results_display import EnhancedResultsDisplay
from .components.data_visualization import DataVisualization
from .components.data_filters import AdvancedDataFilters
from .utils.export_utils import ExportUtils


class TradingSystemUI:
    """
    期权套利交易系统的主用户界面
    
    提供完整的Web界面功能，集成所有后端系统组件
    """
    
    def __init__(self):
        """初始化UI系统"""
        self.logger = self._setup_logging()
        
        # 系统组件
        self.config_manager = None
        self.plugin_manager = None
        self.cache_manager = None
        self.arbitrage_engine = None
        self.data_adapters = {}
        
        # UI组件
        self.config_panel = ConfigPanel()
        self.progress_monitor = ProgressMonitor()
        self.results_display = ResultsDisplay()
        self.enhanced_results_display = EnhancedResultsDisplay()
        self.data_visualization = DataVisualization()
        self.data_filters = AdvancedDataFilters()
        self.export_utils = ExportUtils()
        
        # 状态管理
        self.system_initialized = False
        self.scan_running = False
        self.last_scan_results = []
        
        # 会话状态初始化
        self._initialize_session_state()
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志系统"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_session_state(self):
        """初始化Streamlit会话状态"""
        if 'system_status' not in st.session_state:
            st.session_state.system_status = 'initializing'
        
        if 'scan_progress' not in st.session_state:
            st.session_state.scan_progress = 0
        
        if 'scan_results' not in st.session_state:
            st.session_state.scan_results = []
        
        if 'scan_history' not in st.session_state:
            st.session_state.scan_history = []
        
        if 'error_messages' not in st.session_state:
            st.session_state.error_messages = []
        
        if 'config_valid' not in st.session_state:
            st.session_state.config_valid = False
    
    async def initialize_system(self) -> bool:
        """
        初始化整个交易系统
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            # 1. 检查和设置配置
            self.logger.info("正在初始化系统配置...")
            
            # 检查.env文件
            env_path = Path(".env")
            if not env_path.exists():
                self.logger.error(".env文件不存在")
                st.error("缺少.env配置文件")
                return False
            
            # 检查TUSHARE_TOKEN
            tushare_token = os.getenv('TUSHARE_TOKEN')
            if not tushare_token:
                self.logger.error("TUSHARE_TOKEN未配置")
                st.error("请在.env文件中配置TUSHARE_TOKEN")
                return False
            
            # 2. 初始化配置管理器
            self.config_manager = ConfigManager()
            system_config = self.config_manager.get_system_config()
            self.logger.info("配置管理器初始化完成")
            
            # 3. 初始化缓存管理器
            cache_config = {
                'memory': {'max_entries': 10000, 'max_size_mb': 512},
                'disk': {'enabled': True, 'max_size_gb': 2.0},
                'redis': {'enabled': False}
            }
            self.cache_manager = TradingCacheManager(cache_config)
            await self.cache_manager.initialize()
            self.logger.info("缓存管理器初始化完成")
            
            # 4. 初始化插件管理器
            plugin_config = PluginManagerConfig(
                plugin_directories=['src/strategies'],
                enable_hot_reload=True,
                parallel_loading=True
            )
            self.plugin_manager = PluginManager(plugin_config)
            await self.plugin_manager.initialize()
            self.logger.info(f"插件管理器初始化完成，加载了 {len(self.plugin_manager._plugins)} 个策略")
            
            # 5. 初始化数据适配器
            tushare_config = {
                'api_token': tushare_token,
                'timeout': 30,
                'risk_free_rate': 0.03,
                'max_days_back': 5
            }
            self.data_adapters['tushare'] = TushareAdapter(config=tushare_config)
            await self.data_adapters['tushare'].connect()
            self.logger.info("数据适配器初始化完成")
            
            # 6. 初始化套利引擎
            self.arbitrage_engine = ArbitrageEngine(
                config_manager=self.config_manager,
                data_adapters=self.data_adapters
            )
            self.logger.info("套利引擎初始化完成")
            
            # 更新系统状态
            self.system_initialized = True
            st.session_state.system_status = 'ready'
            st.session_state.config_valid = True
            
            self.logger.info("系统初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}", exc_info=True)
            st.session_state.system_status = 'error'
            st.session_state.error_messages.append(f"系统初始化失败: {str(e)}")
            return False
    
    def _get_or_create_components(self):
        """获取或创建系统组件（使用Streamlit缓存机制）"""
        try:
            # 检查必要的环境变量
            tushare_token = os.getenv('TUSHARE_TOKEN')
            if not tushare_token:
                self.logger.error("TUSHARE_TOKEN环境变量未设置")
                return False
            
            # 使用缓存的组件获取函数
            self.config_manager = self._get_config_manager()
            self.cache_manager = self._get_cache_manager()
            self.data_adapters = self._get_data_adapters()
            self.arbitrage_engine = self._get_arbitrage_engine()
            
            return True
            
        except Exception as e:
            self.logger.error(f"获取系统组件失败: {str(e)}")
            return False
    
    @staticmethod
    @st.cache_resource
    def _get_config_manager():
        """获取配置管理器（缓存版本）"""
        logging.getLogger('src.ui.streamlit_app').info("创建配置管理器（缓存版本）")
        return ConfigManager()
    
    @staticmethod
    @st.cache_resource 
    def _get_cache_manager():
        """获取缓存管理器（缓存版本）"""
        cache_config = {
            'memory': {'max_entries': 10000, 'max_size_mb': 512},
            'disk': {'enabled': True, 'max_size_gb': 2.0},
            'redis': {'enabled': False}
        }
        logging.getLogger('src.ui.streamlit_app').info("创建缓存管理器（缓存版本）")
        return TradingCacheManager(cache_config)
    
    @staticmethod
    @st.cache_resource
    def _get_data_adapters():
        """获取数据适配器（缓存版本）"""
        tushare_token = os.getenv('TUSHARE_TOKEN')
        data_adapters = {}
        if tushare_token:
            tushare_config = {
                'api_token': tushare_token,
                'timeout': 30,
                'risk_free_rate': 0.03,
                'max_days_back': 5
            }
            data_adapters['tushare'] = TushareAdapter(config=tushare_config)
            logging.getLogger('src.ui.streamlit_app').info("创建数据适配器（缓存版本）")
        return data_adapters
    
    @staticmethod
    @st.cache_resource
    def _get_arbitrage_engine():
        """获取套利引擎（缓存版本）"""
        config_manager = TradingSystemUI._get_config_manager()
        cache_manager = TradingSystemUI._get_cache_manager() 
        data_adapters = TradingSystemUI._get_data_adapters()
        
        system_config = config_manager.get_system_config()
        strategy_configs = config_manager.get_enabled_strategies()
        
        logging.getLogger('src.ui.streamlit_app').info("创建套利引擎（缓存版本）")
        return ArbitrageEngine(
            config_manager=config_manager,
            data_adapters=data_adapters
        )
    
    def run(self):
        """运行主界面"""
        # 页面标题
        st.title("💰 期权套利交易机会扫描")
        st.markdown("---")
        
        # 如果系统已初始化但组件未创建，获取缓存的组件
        if (st.session_state.get('system_status') == 'ready' and 
            (not hasattr(self, 'arbitrage_engine') or self.arbitrage_engine is None)):
            self._get_or_create_components()
        
        # 侧边栏
        self._render_sidebar()
        
        # 主界面内容
        if st.session_state.get('system_status') != 'ready':
            self._render_initialization_page()
        else:
            self._render_main_interface()
    
    def _render_sidebar(self):
        """渲染侧边栏"""
        with st.sidebar:
            st.header("📊 系统控制面板")
            
            # 系统状态
            self._render_system_status()
            
            st.markdown("---")
            
            # 配置管理
            self._render_config_section()
            
            st.markdown("---")
            
            # 扫描控制
            self._render_scan_controls()
            
            st.markdown("---")
            
            # 系统信息
            self._render_system_info()
    
    def _render_system_status(self):
        """渲染系统状态"""
        st.subheader("🔧 系统状态")
        
        status = st.session_state.system_status
        if status == 'ready':
            st.success("✅ 系统就绪")
        elif status == 'initializing':
            st.warning("⏳ 正在初始化...")
        elif status == 'scanning':
            st.info("🔍 正在扫描...")
        elif status == 'error':
            st.error("❌ 系统错误")
        else:
            st.warning("⚠️ 状态未知")
        
        # 显示错误信息
        if st.session_state.error_messages:
            with st.expander("⚠️ 错误信息", expanded=True):
                for error in st.session_state.error_messages[-5:]:  # 最近5个错误
                    st.error(error)
                
                if st.button("清除错误信息"):
                    st.session_state.error_messages.clear()
                    st.rerun()
    
    def _render_config_section(self):
        """渲染配置区域"""
        st.subheader("⚙️ 配置管理")
        
        # 检查.env文件
        env_path = Path(".env")
        if env_path.exists():
            st.success("✅ .env文件已找到")
            
            # 检查TUSHARE_TOKEN
            if os.getenv('TUSHARE_TOKEN'):
                st.success("✅ TUSHARE_TOKEN已配置")
            else:
                st.error("❌ TUSHARE_TOKEN未配置")
        else:
            st.error("❌ .env文件不存在")
        
        # 重新加载配置按钮
        if st.button("🔄 重新加载配置"):
            st.session_state.config_valid = False
            st.rerun()
    
    def _render_scan_controls(self):
        """渲染扫描控制区域"""
        st.subheader("🎯 扫描控制")
        
        if not st.session_state.config_valid:
            st.warning("请先初始化系统")
            return
        
        # 扫描参数
        with st.expander("📋 扫描参数", expanded=False):
            min_profit = st.slider(
                "最小利润阈值 (%)",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1
            )
            
            max_risk = st.slider(
                "最大风险容忍度 (%)",
                min_value=1.0,
                max_value=20.0,
                value=10.0,
                step=1.0
            )
            
            max_results = st.number_input(
                "最大结果数量",
                min_value=10,
                max_value=1000,
                value=100,
                step=10
            )
        
        # 一键扫描按钮
        scan_disabled = st.session_state.system_status == 'scanning'
        
        if st.button(
            "🚀 一键扫描套利机会", 
            disabled=scan_disabled,
            use_container_width=True
        ):
            # 开始扫描
            asyncio.run(self._run_arbitrage_scan(
                min_profit_threshold=min_profit / 100,
                max_risk_tolerance=max_risk / 100,
                max_results=max_results
            ))
    
    def _render_system_info(self):
        """渲染系统信息"""
        st.subheader("ℹ️ 系统信息")
        
        if st.session_state.config_valid and hasattr(self, 'plugin_manager') and self.plugin_manager:
            plugin_count = len(self.plugin_manager._plugins)
            st.metric("策略插件数量", plugin_count)
        
        if st.session_state.scan_history:
            st.metric("历史扫描次数", len(st.session_state.scan_history))
    
    def _render_initialization_page(self):
        """渲染初始化页面"""
        st.header("🔧 系统初始化")
        
        # 显示初始化状态
        if st.session_state.system_status == 'initializing':
            with st.spinner("正在初始化系统..."):
                # 异步初始化系统
                success = asyncio.run(self.initialize_system())
                if success:
                    st.success("✅ 系统初始化成功！")
                    st.rerun()
                else:
                    st.error("❌ 系统初始化失败")
        
        elif st.session_state.system_status == 'error':
            st.error("❌ 系统初始化失败")
            
            # 显示错误信息
            if st.session_state.error_messages:
                for error in st.session_state.error_messages:
                    st.error(error)
            
            # 重试按钮
            if st.button("🔄 重新初始化"):
                st.session_state.system_status = 'initializing'
                st.session_state.error_messages.clear()
                st.rerun()
        
        else:
            # 手动初始化按钮
            if st.button("🚀 初始化系统", use_container_width=True):
                st.session_state.system_status = 'initializing'
                st.rerun()
        
        # 显示配置检查结果
        st.markdown("---")
        st.subheader("📋 配置检查")
        
        env_path = Path(".env")
        if env_path.exists():
            st.success("✅ .env文件存在")
        else:
            st.error("❌ .env文件不存在")
            st.info("请在项目根目录创建.env文件并添加TUSHARE_TOKEN配置")
        
        if os.getenv('TUSHARE_TOKEN'):
            st.success("✅ TUSHARE_TOKEN已配置")
        else:
            st.error("❌ TUSHARE_TOKEN未配置")
            st.info("请在.env文件中添加: TUSHARE_TOKEN=your_token_here")
    
    def _render_main_interface(self):
        """渲染主界面"""
        # 创建标签页
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 扫描结果", 
            "📊 高级分析", 
            "🔍 数据筛选", 
            "📈 数据可视化", 
            "⚙️ 系统设置"
        ])
        
        with tab1:
            self._render_enhanced_scan_results()
        
        with tab2:
            self._render_advanced_analysis()
        
        with tab3:
            self._render_data_filtering()
        
        with tab4:
            self._render_data_visualization()
        
        with tab5:
            self._render_settings_panel()
    
    def _render_enhanced_scan_results(self):
        """渲染增强版扫描结果页面"""
        st.header("🎯 增强版套利机会分析")
        
        if st.session_state.scan_results:
            # 显示增强版概览
            self.enhanced_results_display.render_enhanced_overview(st.session_state.scan_results)
            
            st.markdown("---")
            
            # 数据准备
            df = pd.DataFrame(st.session_state.scan_results)
            
            # 高级排序控制
            sorted_df = self.enhanced_results_display.render_advanced_sorting_controls(df)
            
            # 高级筛选
            filtered_df = self.enhanced_results_display.render_advanced_filters(sorted_df)
            
            st.markdown("---")
            
            # 分页表格显示
            self.enhanced_results_display.render_paginated_table(filtered_df)
            
            st.markdown("---")
            
            # 导出功能
            self.export_utils.create_export_interface(filtered_df, "arbitrage_results")
            
            # 期权Greeks显示（如果启用）
            if st.session_state.get('results_show_greeks', False):
                st.markdown("---")
                self.enhanced_results_display.render_greeks_display(st.session_state.scan_results)
            
            # 高级图表
            if not filtered_df.empty:
                st.markdown("---")
                self.enhanced_results_display.render_advanced_charts(filtered_df)
        
        else:
            st.info("暂无扫描结果，请点击'一键扫描'开始搜索套利机会")
            
            # 显示示例数据说明
            with st.expander("📖 功能说明", expanded=False):
                st.markdown("""
                **增强版套利扫描功能**：
                - 🔍 实时扫描期权市场中的套利机会
                - 📊 多维度数据分析和可视化
                - ⚡ 高性能并行计算提供快速结果
                - 🎯 智能风险评估和机会排序
                - 🔄 高级排序和筛选功能
                - 📥 多格式数据导出
                - 📈 期权Greeks分析
                
                **使用方法**：
                1. 在侧边栏调整扫描参数
                2. 点击"一键扫描套利机会"按钮
                3. 使用高级筛选和排序功能
                4. 分析可视化图表和数据
                5. 导出分析结果
                """)
            
            # 模板导出功能
            with st.expander("📄 下载导入模板", expanded=False):
                self.export_utils.create_template_export()
    
    
    def _render_advanced_analysis(self):
        """渲染高级分析面板"""
        st.header("📊 高级数据分析")
        
        if st.session_state.scan_results:
            # 综合分析选项
            analysis_options = st.multiselect(
                "选择分析类型",
                options=[
                    "结果摘要分析",
                    "策略效果对比", 
                    "风险收益分析",
                    "时间序列分析",
                    "相关性分析"
                ],
                default=["结果摘要分析", "策略效果对比"],
                help="选择要显示的分析类型"
            )
            
            df = pd.DataFrame(st.session_state.scan_results)
            
            if "结果摘要分析" in analysis_options:
                st.markdown("---")
                self.enhanced_results_display.render_results_summary(st.session_state.scan_results)
            
            if "策略效果对比" in analysis_options:
                st.markdown("---")
                st.subheader("📊 策略效果对比")
                self.data_visualization.render_strategy_performance_analysis(df)
            
            if "风险收益分析" in analysis_options:
                st.markdown("---")
                st.subheader("🎯 风险收益分析")
                if 'profit_margin' in df.columns and 'risk_score' in df.columns:
                    self.data_visualization._render_risk_return_chart(df)
                    self.data_visualization._render_quadrant_analysis(df)
            
            if "时间序列分析" in analysis_options:
                st.markdown("---")
                self.data_visualization.render_time_series_analysis(df)
            
            if "相关性分析" in analysis_options:
                st.markdown("---")
                self.data_visualization.render_correlation_analysis(df)
        
        else:
            st.info("请先进行扫描以获取分析数据")
            
            # 系统健康状态（移到这里）
            if self.system_initialized:
                st.markdown("---")
                st.subheader("🔧 系统监控")
                self._render_system_health()
    
    def _render_system_health(self):
        """渲染系统健康状态"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("系统状态", "正常" if st.session_state.system_status == 'ready' else "异常")
        
        with col2:
            if hasattr(self, 'cache_manager') and self.cache_manager:
                cache_stats = self.cache_manager.get_statistics()
                hit_rate = cache_stats.get('hit_rate', 0) * 100
                st.metric("缓存命中率", f"{hit_rate:.1f}%")
            else:
                st.metric("缓存命中率", "N/A")
        
        with col3:
            if hasattr(self, 'arbitrage_engine') and self.arbitrage_engine:
                metrics = self.arbitrage_engine.get_performance_metrics()
                st.metric("平均扫描时间", f"{metrics.avg_scan_time:.2f}s")
            else:
                st.metric("平均扫描时间", "N/A")
    
    def _render_data_filtering(self):
        """渲染数据筛选面板"""
        st.header("🔍 高级数据筛选")
        
        if st.session_state.scan_results:
            df = pd.DataFrame(st.session_state.scan_results)
            
            # 显示原始数据概览
            st.write(f"**原始数据**: {len(df)} 条记录, {len(df.columns)} 个字段")
            
            # 渲染筛选界面
            filtered_df = self.data_filters.render_filter_interface(df)
            
            # 显示筛选后的数据预览
            if not filtered_df.empty and len(filtered_df) != len(df):
                st.markdown("---")
                st.subheader("📊 筛选结果预览")
                
                # 显示前几行数据
                preview_rows = min(10, len(filtered_df))
                st.write(f"显示前 {preview_rows} 行筛选结果：")
                st.dataframe(
                    filtered_df.head(preview_rows),
                    use_container_width=True,
                    hide_index=True
                )
                
                # 导出筛选结果
                st.markdown("---")
                self.export_utils.create_export_interface(filtered_df, "filtered_results")
                
                # 更新会话状态中的筛选结果供其他标签页使用
                st.session_state.filtered_results = filtered_df.to_dict('records')
        
        else:
            st.info("请先进行扫描以获取筛选数据")
            
            # 筛选器配置管理
            st.markdown("---")
            st.subheader("⚙️ 筛选器配置管理")
            
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                if st.button("📥 导出筛选配置"):
                    config_json = self.data_filters.export_filter_config()
                    st.download_button(
                        label="下载配置文件",
                        data=config_json,
                        file_name="filter_config.json",
                        mime="application/json"
                    )
            
            with config_col2:
                uploaded_config = st.file_uploader(
                    "📤 导入筛选配置",
                    type="json",
                    help="上传之前导出的筛选配置文件"
                )
                
                if uploaded_config is not None:
                    config_content = uploaded_config.read().decode('utf-8')
                    if self.data_filters.import_filter_config(config_content):
                        st.success("筛选配置导入成功！")
                        st.rerun()
    
    def _render_data_visualization(self):
        """渲染数据可视化面板"""
        st.header("📈 数据可视化")
        
        # 选择数据源
        data_source = st.radio(
            "数据源选择",
            options=["原始扫描结果", "筛选后结果"],
            horizontal=True,
            help="选择用于可视化的数据源"
        )
        
        # 获取数据
        if data_source == "筛选后结果" and 'filtered_results' in st.session_state:
            visualization_data = st.session_state.filtered_results
            st.info(f"使用筛选后的数据: {len(visualization_data)} 条记录")
        elif st.session_state.scan_results:
            visualization_data = st.session_state.scan_results
            st.info(f"使用原始扫描数据: {len(visualization_data)} 条记录")
        else:
            visualization_data = None
        
        if visualization_data:
            # 渲染综合可视化仪表板
            self.data_visualization.render_comprehensive_dashboard(visualization_data)
        
        else:
            st.info("请先进行扫描以获取可视化数据")
            
            # 扫描历史可视化（如果有的话）
            if st.session_state.scan_history:
                st.markdown("---")
                st.subheader("📊 扫描历史分析")
                
                history_df = pd.DataFrame(st.session_state.scan_history)
                
                # 时间序列图
                fig_timeline = px.line(
                    history_df,
                    x='timestamp',
                    y='opportunities_found',
                    title='扫描历史趋势',
                    labels={'timestamp': '时间', 'opportunities_found': '发现机会数'}
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                # 扫描效率分析
                if 'scan_time' in history_df.columns:
                    fig_efficiency = px.scatter(
                        history_df,
                        x='scan_time',
                        y='opportunities_found',
                        title='扫描效率分析',
                        labels={'scan_time': '扫描时间(秒)', 'opportunities_found': '发现机会数'},
                        hover_data=['timestamp']
                    )
                    st.plotly_chart(fig_efficiency, use_container_width=True)
    
    def _render_settings_panel(self):
        """渲染设置面板"""
        st.header("⚙️ 系统设置")
        
        # 插件管理
        if hasattr(self, 'plugin_manager') and self.plugin_manager:
            st.subheader("🔌 策略插件管理")
            
            plugins = self.plugin_manager.list_plugins()
            for plugin_name, plugin_info in plugins.items():
                with st.expander(f"📦 {plugin_name}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**类型**: {plugin_info.strategy_type}")
                        st.write(f"**状态**: {'启用' if plugin_info.is_enabled else '禁用'}")
                    
                    with col2:
                        st.write(f"**加载次数**: {plugin_info.load_count}")
                        st.write(f"**错误次数**: {plugin_info.error_count}")
        
        # 缓存管理
        if hasattr(self, 'cache_manager') and self.cache_manager:
            st.subheader("💾 缓存管理")
            
            cache_stats = self.cache_manager.get_statistics()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("总键数", cache_stats.get('total_keys', 0))
            
            with col2:
                st.metric("命中次数", cache_stats.get('hits', 0))
            
            with col3:
                st.metric("未命中次数", cache_stats.get('misses', 0))
            
            if st.button("🗑️ 清空缓存"):
                if hasattr(self.cache_manager, 'clear_all'):
                    asyncio.run(self.cache_manager.clear_all())
                    st.success("缓存已清空")
                    st.rerun()
        
        # 高级设置
        st.subheader("⚙️ 界面设置")
        
        interface_col1, interface_col2 = st.columns(2)
        
        with interface_col1:
            # 结果显示设置
            st.session_state.results_show_greeks = st.checkbox(
                "显示期权Greeks",
                value=st.session_state.get('results_show_greeks', False),
                help="在结果页面显示期权Greeks分析"
            )
            
            st.session_state.results_show_risk_metrics = st.checkbox(
                "显示风险指标",
                value=st.session_state.get('results_show_risk_metrics', True),
                help="在结果页面显示详细风险指标"
            )
        
        with interface_col2:
            # 图表设置
            st.session_state.chart_theme = st.selectbox(
                "图表主题",
                options=["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"],
                index=0,
                help="选择数据可视化的图表主题"
            )
            
            st.session_state.max_chart_points = st.number_input(
                "图表最大数据点",
                min_value=100,
                max_value=10000,
                value=st.session_state.get('max_chart_points', 1000),
                help="限制图表显示的最大数据点数量以提升性能"
            )
        
        # 批量导出功能
        if st.session_state.scan_results or st.session_state.scan_history:
            st.markdown("---")
            st.subheader("📦 批量数据导出")
            
            export_datasets = {}
            if st.session_state.scan_results:
                export_datasets["扫描结果"] = pd.DataFrame(st.session_state.scan_results)
            
            if st.session_state.scan_history:
                export_datasets["扫描历史"] = pd.DataFrame(st.session_state.scan_history)
            
            if 'filtered_results' in st.session_state:
                export_datasets["筛选结果"] = pd.DataFrame(st.session_state.filtered_results)
            
            if export_datasets:
                self.export_utils.create_batch_export_interface(export_datasets)
    
    async def _run_arbitrage_scan(
        self,
        min_profit_threshold: float = 0.01,
        max_risk_tolerance: float = 0.1,
        max_results: int = 100
    ):
        """
        执行套利扫描
        
        Args:
            min_profit_threshold: 最小利润阈值
            max_risk_tolerance: 最大风险容忍度
            max_results: 最大结果数量
        """
        # 检查系统状态
        if st.session_state.system_status != 'ready':
            st.error("系统未初始化")
            return
        
        # 检查组件是否可用
        if not hasattr(self, 'arbitrage_engine') or not self.arbitrage_engine:
            st.error("系统组件未加载，请刷新页面")
            return
        
        try:
            st.session_state.system_status = 'scanning'
            
            # 显示进度
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with status_text:
                st.info("🔍 正在扫描套利机会...")
            
            # 创建扫描参数
            scan_params = ScanParameters(
                min_profit_threshold=min_profit_threshold,
                max_risk_tolerance=max_risk_tolerance,
                max_results=max_results,
                include_greeks=True,
                include_iv=True
            )
            
            # 模拟进度更新
            for i in range(0, 101, 10):
                progress_bar.progress(i)
                await asyncio.sleep(0.1)
            
            # 执行扫描
            start_time = time.time()
            opportunities = await self.arbitrage_engine.scan_opportunities(scan_params)
            scan_time = time.time() - start_time
            
            # 更新结果
            results = []
            for opp in opportunities:
                results.append({
                    'id': opp.id,
                    'strategy_type': opp.strategy_type.value,
                    'profit_margin': opp.profit_margin,
                    'expected_profit': opp.expected_profit,
                    'risk_score': opp.risk_score,
                    'confidence_score': opp.confidence_score,
                    'instruments': ', '.join(opp.instruments),
                    'timestamp': opp.timestamp
                })
            
            st.session_state.scan_results = results
            
            # 添加到历史记录
            st.session_state.scan_history.append({
                'timestamp': datetime.now(),
                'scan_time': scan_time,
                'opportunities_found': len(results),
                'parameters': scan_params.__dict__
            })
            
            # 更新状态
            st.session_state.system_status = 'ready'
            
            # 清除进度显示
            progress_bar.empty()
            
            with status_text:
                st.success(f"✅ 扫描完成！发现 {len(results)} 个套利机会，用时 {scan_time:.2f}s")
            
            # 强制刷新页面以显示新结果
            st.rerun()
            
        except Exception as e:
            self.logger.error(f"扫描失败: {e}", exc_info=True)
            st.session_state.system_status = 'ready'
            st.session_state.error_messages.append(f"扫描失败: {str(e)}")
            st.error(f"❌ 扫描失败: {str(e)}")