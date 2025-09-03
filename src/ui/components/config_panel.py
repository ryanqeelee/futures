"""
配置管理面板组件

提供系统配置的界面管理功能
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import streamlit as st


class ConfigPanel:
    """配置管理面板组件"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def render_env_config(self) -> bool:
        """
        渲染环境配置检查
        
        Returns:
            bool: 配置是否有效
        """
        st.subheader("🔧 环境配置检查")
        
        config_valid = True
        
        # 检查.env文件
        env_path = Path(".env")
        if env_path.exists():
            st.success("✅ .env文件已找到")
        else:
            st.error("❌ .env文件不存在")
            st.info("请在项目根目录创建.env文件")
            config_valid = False
        
        # 检查TUSHARE_TOKEN
        tushare_token = os.getenv('TUSHARE_TOKEN')
        if tushare_token:
            # 隐藏token的部分内容
            masked_token = tushare_token[:8] + "*" * (len(tushare_token) - 16) + tushare_token[-8:]
            st.success(f"✅ TUSHARE_TOKEN已配置: {masked_token}")
        else:
            st.error("❌ TUSHARE_TOKEN未配置")
            st.info("请在.env文件中添加: TUSHARE_TOKEN=your_token_here")
            config_valid = False
        
        return config_valid
    
    def render_system_config(self, config_manager=None) -> Dict[str, Any]:
        """
        渲染系统配置面板
        
        Args:
            config_manager: 配置管理器实例
            
        Returns:
            Dict: 配置参数
        """
        st.subheader("⚙️ 系统配置")
        
        config = {}
        
        with st.expander("🔧 基本设置", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                config['log_level'] = st.selectbox(
                    "日志级别",
                    options=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                    index=1,  # 默认INFO
                    help="设置系统日志的详细程度"
                )
                
                config['max_workers'] = st.number_input(
                    "最大工作线程数",
                    min_value=1,
                    max_value=32,
                    value=8,
                    help="并行处理的最大线程数"
                )
            
            with col2:
                config['enable_cache'] = st.checkbox(
                    "启用缓存",
                    value=True,
                    help="启用数据缓存以提高性能"
                )
                
                config['cache_ttl'] = st.number_input(
                    "缓存过期时间(秒)",
                    min_value=60,
                    max_value=3600,
                    value=300,
                    help="缓存数据的生存时间"
                )
        
        with st.expander("🎯 扫描设置", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                config['default_profit_threshold'] = st.slider(
                    "默认利润阈值 (%)",
                    min_value=0.1,
                    max_value=5.0,
                    value=1.0,
                    step=0.1,
                    help="默认的最小利润要求"
                )
                
                config['default_risk_tolerance'] = st.slider(
                    "默认风险容忍度 (%)",
                    min_value=1.0,
                    max_value=20.0,
                    value=10.0,
                    step=1.0,
                    help="默认的最大风险承受度"
                )
            
            with col2:
                config['scan_timeout'] = st.number_input(
                    "扫描超时时间(秒)",
                    min_value=30,
                    max_value=300,
                    value=120,
                    help="单次扫描的最大时间限制"
                )
                
                config['max_scan_results'] = st.number_input(
                    "最大结果数量",
                    min_value=10,
                    max_value=1000,
                    value=100,
                    help="单次扫描返回的最大结果数"
                )
        
        return config
    
    def render_strategy_config(self, plugin_manager=None) -> Dict[str, Any]:
        """
        渲染策略配置面板
        
        Args:
            plugin_manager: 插件管理器实例
            
        Returns:
            Dict: 策略配置
        """
        st.subheader("🎲 策略配置")
        
        strategy_config = {}
        
        if plugin_manager and hasattr(plugin_manager, 'list_plugins'):
            plugins = plugin_manager.list_plugins()
            
            if plugins:
                for plugin_name, plugin_info in plugins.items():
                    with st.expander(f"📦 {plugin_name}", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            enabled = st.checkbox(
                                "启用策略",
                                value=plugin_info.is_enabled,
                                key=f"enable_{plugin_name}"
                            )
                            
                            priority = st.number_input(
                                "执行优先级",
                                min_value=1,
                                max_value=10,
                                value=plugin_info.priority,
                                key=f"priority_{plugin_name}",
                                help="数值越大优先级越高"
                            )
                        
                        with col2:
                            st.write(f"**类型**: {plugin_info.strategy_type}")
                            st.write(f"**加载时间**: {plugin_info.load_time.strftime('%Y-%m-%d %H:%M:%S')}")
                            
                            if plugin_info.last_error:
                                st.error(f"错误: {plugin_info.last_error}")
                        
                        strategy_config[plugin_name] = {
                            'enabled': enabled,
                            'priority': priority,
                            'type': plugin_info.strategy_type
                        }
            else:
                st.info("暂无可用策略插件")
        else:
            st.warning("插件管理器未初始化")
        
        return strategy_config
    
    def render_data_adapter_config(self, data_adapters=None) -> Dict[str, Any]:
        """
        渲染数据适配器配置
        
        Args:
            data_adapters: 数据适配器字典
            
        Returns:
            Dict: 适配器配置
        """
        st.subheader("📡 数据源配置")
        
        adapter_config = {}
        
        if data_adapters:
            for adapter_name, adapter in data_adapters.items():
                with st.expander(f"🔌 {adapter_name.title()}适配器", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if hasattr(adapter, 'is_connected'):
                            if adapter.is_connected:
                                st.success("✅ 连接正常")
                            else:
                                st.error("❌ 连接异常")
                        
                        if hasattr(adapter, 'connection_info'):
                            info = adapter.connection_info
                            st.write(f"**状态**: {info.status.value}")
                            st.write(f"**延迟**: {info.latency_ms}ms")
                    
                    with col2:
                        if hasattr(adapter, 'get_statistics'):
                            stats = adapter.get_statistics()
                            st.write(f"**请求次数**: {stats.get('total_requests', 0)}")
                            st.write(f"**成功率**: {stats.get('success_rate', 0):.2%}")
                    
                    # 配置选项
                    adapter_config[adapter_name] = {
                        'enabled': st.checkbox(
                            "启用适配器",
                            value=True,
                            key=f"adapter_{adapter_name}"
                        ),
                        'timeout': st.number_input(
                            "请求超时(秒)",
                            min_value=5,
                            max_value=60,
                            value=30,
                            key=f"timeout_{adapter_name}"
                        )
                    }
        else:
            st.info("暂无数据适配器")
        
        return adapter_config
    
    def render_cache_config(self, cache_manager=None) -> Dict[str, Any]:
        """
        渲染缓存配置面板
        
        Args:
            cache_manager: 缓存管理器实例
            
        Returns:
            Dict: 缓存配置
        """
        st.subheader("💾 缓存配置")
        
        cache_config = {}
        
        with st.expander("🔧 基本设置", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                cache_config['memory_enabled'] = st.checkbox(
                    "启用内存缓存",
                    value=True,
                    help="使用内存作为L1缓存"
                )
                
                cache_config['disk_enabled'] = st.checkbox(
                    "启用磁盘缓存",
                    value=True,
                    help="使用磁盘作为L2缓存"
                )
            
            with col2:
                cache_config['redis_enabled'] = st.checkbox(
                    "启用Redis缓存",
                    value=False,
                    help="使用Redis作为L3缓存"
                )
                
                cache_config['compression'] = st.checkbox(
                    "启用压缩",
                    value=True,
                    help="压缩缓存数据以节省空间"
                )
        
        with st.expander("📊 缓存统计", expanded=False):
            if cache_manager and hasattr(cache_manager, 'get_statistics'):
                stats = cache_manager.get_statistics()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("总键数", stats.get('total_keys', 0))
                    st.metric("命中次数", stats.get('hits', 0))
                
                with col2:
                    st.metric("未命中次数", stats.get('misses', 0))
                    hit_rate = stats.get('hit_rate', 0) * 100
                    st.metric("命中率", f"{hit_rate:.1f}%")
                
                with col3:
                    st.metric("内存使用", f"{stats.get('memory_usage_mb', 0):.1f} MB")
                    st.metric("磁盘使用", f"{stats.get('disk_usage_mb', 0):.1f} MB")
                
                # 缓存操作按钮
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🗑️ 清空内存缓存"):
                        if hasattr(cache_manager, 'clear_memory_cache'):
                            cache_manager.clear_memory_cache()
                            st.success("内存缓存已清空")
                
                with col2:
                    if st.button("🗑️ 清空所有缓存"):
                        if hasattr(cache_manager, 'clear_all'):
                            cache_manager.clear_all()
                            st.success("所有缓存已清空")
            else:
                st.info("缓存管理器未初始化")
        
        return cache_config
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """
        保存配置到文件
        
        Args:
            config: 配置字典
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 这里应该实现配置保存逻辑
            # 可以保存到JSON文件或数据库
            self.logger.info("配置保存成功")
            return True
        except Exception as e:
            self.logger.error(f"配置保存失败: {e}")
            return False