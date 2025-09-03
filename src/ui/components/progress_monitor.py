"""
进度监控组件

提供实时的进度显示和状态监控功能
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ProgressMonitor:
    """进度监控组件"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._start_time = None
        self._current_step = 0
        self._total_steps = 0
        self._step_descriptions = []
    
    def start_monitoring(self, total_steps: int, step_descriptions: List[str] = None):
        """
        开始监控进度
        
        Args:
            total_steps: 总步骤数
            step_descriptions: 每个步骤的描述
        """
        self._start_time = time.time()
        self._current_step = 0
        self._total_steps = total_steps
        self._step_descriptions = step_descriptions or [f"步骤 {i+1}" for i in range(total_steps)]
    
    def update_progress(self, step: int, description: str = None):
        """
        更新进度
        
        Args:
            step: 当前步骤（从0开始）
            description: 当前步骤描述
        """
        self._current_step = step
        if description and step < len(self._step_descriptions):
            self._step_descriptions[step] = description
    
    def render_progress_bar(self, 
                           progress_placeholder=None,
                           status_placeholder=None,
                           show_eta: bool = True) -> tuple:
        """
        渲染进度条
        
        Args:
            progress_placeholder: 进度条占位符
            status_placeholder: 状态文本占位符
            show_eta: 是否显示预计完成时间
            
        Returns:
            tuple: (progress_placeholder, status_placeholder)
        """
        if progress_placeholder is None:
            progress_placeholder = st.empty()
        if status_placeholder is None:
            status_placeholder = st.empty()
        
        # 计算进度百分比
        if self._total_steps > 0:
            progress = self._current_step / self._total_steps
        else:
            progress = 0.0
        
        # 显示进度条
        with progress_placeholder:
            st.progress(progress)
        
        # 显示状态信息
        with status_placeholder:
            if self._current_step < len(self._step_descriptions):
                current_desc = self._step_descriptions[self._current_step]
            else:
                current_desc = f"步骤 {self._current_step + 1}"
            
            status_text = f"🔄 {current_desc} ({self._current_step + 1}/{self._total_steps})"
            
            if show_eta and self._start_time:
                elapsed = time.time() - self._start_time
                if self._current_step > 0:
                    avg_time_per_step = elapsed / self._current_step
                    remaining_steps = self._total_steps - self._current_step
                    eta_seconds = avg_time_per_step * remaining_steps
                    eta_str = self._format_time(eta_seconds)
                    status_text += f" - 预计剩余时间: {eta_str}"
            
            st.info(status_text)
        
        return progress_placeholder, status_placeholder
    
    def render_detailed_progress(self) -> None:
        """渲染详细的进度信息"""
        st.subheader("📊 详细进度")
        
        if not self._start_time:
            st.info("监控尚未开始")
            return
        
        # 创建进度数据
        progress_data = []
        for i, desc in enumerate(self._step_descriptions):
            status = "已完成" if i < self._current_step else ("进行中" if i == self._current_step else "待执行")
            progress_data.append({
                '步骤': i + 1,
                '描述': desc,
                '状态': status
            })
        
        progress_df = pd.DataFrame(progress_data)
        
        # 显示进度表格
        st.dataframe(
            progress_df,
            use_container_width=True,
            hide_index=True
        )
        
        # 时间统计
        elapsed = time.time() - self._start_time
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("已用时间", self._format_time(elapsed))
        
        with col2:
            if self._current_step > 0:
                avg_time = elapsed / self._current_step
                st.metric("平均每步", self._format_time(avg_time))
            else:
                st.metric("平均每步", "计算中...")
        
        with col3:
            if self._current_step > 0 and self._current_step < self._total_steps:
                avg_time = elapsed / self._current_step
                remaining_steps = self._total_steps - self._current_step
                eta = avg_time * remaining_steps
                st.metric("预计剩余", self._format_time(eta))
            else:
                st.metric("预计剩余", "N/A")
    
    def render_system_metrics(self, 
                             arbitrage_engine=None,
                             cache_manager=None,
                             data_adapters=None) -> None:
        """
        渲染系统性能指标
        
        Args:
            arbitrage_engine: 套利引擎实例
            cache_manager: 缓存管理器实例
            data_adapters: 数据适配器字典
        """
        st.subheader("⚡ 系统性能指标")
        
        # 套利引擎指标
        if arbitrage_engine:
            self._render_engine_metrics(arbitrage_engine)
        
        # 缓存性能指标
        if cache_manager:
            self._render_cache_metrics(cache_manager)
        
        # 数据适配器指标
        if data_adapters:
            self._render_adapter_metrics(data_adapters)
    
    def _render_engine_metrics(self, engine) -> None:
        """渲染引擎性能指标"""
        st.write("🚀 **套利引擎性能**")
        
        try:
            metrics = engine.get_performance_metrics()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "总扫描次数",
                    getattr(metrics, 'scans_completed', 0)
                )
            
            with col2:
                st.metric(
                    "平均扫描时间",
                    f"{metrics.avg_scan_time:.2f}s"
                )
            
            with col3:
                st.metric(
                    "发现机会总数",
                    metrics.total_opportunities_found
                )
            
            with col4:
                st.metric(
                    "策略执行数",
                    metrics.strategies_executed
                )
            
            # 性能图表
            if hasattr(metrics, 'scan_history') and metrics.scan_history:
                self._render_performance_chart(metrics.scan_history)
                
        except Exception as e:
            st.error(f"获取引擎指标失败: {e}")
    
    def _render_cache_metrics(self, cache_manager) -> None:
        """渲染缓存性能指标"""
        st.write("💾 **缓存性能**")
        
        try:
            stats = cache_manager.get_statistics()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("缓存键数", stats.get('total_keys', 0))
            
            with col2:
                hit_rate = stats.get('hit_rate', 0) * 100
                st.metric("命中率", f"{hit_rate:.1f}%")
            
            with col3:
                st.metric("命中次数", stats.get('hits', 0))
            
            with col4:
                st.metric("未命中次数", stats.get('misses', 0))
            
            # 缓存层级性能
            if 'tier_details' in stats:
                self._render_cache_tier_chart(stats['tier_details'])
                
        except Exception as e:
            st.error(f"获取缓存指标失败: {e}")
    
    def _render_adapter_metrics(self, adapters) -> None:
        """渲染适配器性能指标"""
        st.write("📡 **数据适配器性能**")
        
        adapter_data = []
        
        for name, adapter in adapters.items():
            try:
                if hasattr(adapter, 'get_statistics'):
                    stats = adapter.get_statistics()
                    adapter_data.append({
                        '适配器': name.title(),
                        '请求总数': stats.get('total_requests', 0),
                        '成功率': f"{stats.get('success_rate', 0):.1%}",
                        '平均延迟': f"{stats.get('avg_latency_ms', 0):.1f}ms",
                        '状态': '正常' if adapter.is_connected else '异常'
                    })
            except Exception as e:
                adapter_data.append({
                    '适配器': name.title(),
                    '请求总数': 'N/A',
                    '成功率': 'N/A',
                    '平均延迟': 'N/A',
                    '状态': f'错误: {e}'
                })
        
        if adapter_data:
            adapter_df = pd.DataFrame(adapter_data)
            st.dataframe(
                adapter_df,
                use_container_width=True,
                hide_index=True
            )
    
    def _render_performance_chart(self, scan_history: List[Dict]) -> None:
        """渲染性能趋势图表"""
        if not scan_history:
            return
        
        # 转换为DataFrame
        history_df = pd.DataFrame(scan_history)
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('扫描时间趋势', '发现机会数趋势', 
                          '扫描时间分布', '机会数分布'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 时间趋势线
        fig.add_trace(
            go.Scatter(
                x=history_df.index,
                y=history_df['scan_time'],
                mode='lines+markers',
                name='扫描时间'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=history_df.index,
                y=history_df['opportunities_found'],
                mode='lines+markers',
                name='发现机会数',
                line=dict(color='orange')
            ),
            row=1, col=2
        )
        
        # 分布直方图
        fig.add_trace(
            go.Histogram(
                x=history_df['scan_time'],
                name='扫描时间分布',
                nbinsx=10
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=history_df['opportunities_found'],
                name='机会数分布',
                nbinsx=10
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="系统性能趋势分析"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_cache_tier_chart(self, tier_details: Dict) -> None:
        """渲染缓存层级性能图表"""
        if not tier_details:
            return
        
        # 准备数据
        tier_names = []
        hit_rates = []
        
        for tier_name, details in tier_details.items():
            tier_names.append(tier_name)
            hit_rate = details.get('hit_rate', 0) * 100
            hit_rates.append(hit_rate)
        
        # 创建条形图
        fig = px.bar(
            x=tier_names,
            y=hit_rates,
            title="缓存层级命中率",
            labels={'x': '缓存层级', 'y': '命中率 (%)'}
        )
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{int(minutes)}分{int(secs)}秒"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{int(hours)}小时{int(minutes)}分钟"
    
    def render_real_time_monitor(self, update_interval: int = 5) -> None:
        """
        渲染实时监控面板
        
        Args:
            update_interval: 更新间隔（秒）
        """
        st.subheader("📊 实时系统监控")
        
        # 创建占位符
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        # 自动刷新
        if st.button("🔄 开启自动刷新"):
            st.session_state.auto_refresh = True
        
        if st.button("⏹️ 停止自动刷新"):
            st.session_state.auto_refresh = False
        
        # 实时数据更新逻辑
        if getattr(st.session_state, 'auto_refresh', False):
            # 这里应该实现实时数据获取和更新
            # 由于Streamlit的限制，真正的实时更新需要配合其他技术
            st.info(f"每{update_interval}秒自动更新...")
    
    def export_metrics(self, 
                      arbitrage_engine=None,
                      cache_manager=None) -> Optional[pd.DataFrame]:
        """
        导出性能指标数据
        
        Args:
            arbitrage_engine: 套利引擎实例
            cache_manager: 缓存管理器实例
            
        Returns:
            pd.DataFrame: 指标数据框
        """
        try:
            data = []
            
            if arbitrage_engine:
                metrics = arbitrage_engine.get_performance_metrics()
                data.append({
                    '类型': '套利引擎',
                    '指标': '平均扫描时间',
                    '数值': metrics.avg_scan_time,
                    '单位': '秒',
                    '时间': datetime.now()
                })
                
                data.append({
                    '类型': '套利引擎',
                    '指标': '总机会数',
                    '数值': metrics.total_opportunities_found,
                    '单位': '个',
                    '时间': datetime.now()
                })
            
            if cache_manager:
                stats = cache_manager.get_statistics()
                hit_rate = stats.get('hit_rate', 0)
                
                data.append({
                    '类型': '缓存系统',
                    '指标': '命中率',
                    '数值': hit_rate,
                    '单位': '%',
                    '时间': datetime.now()
                })
            
            if data:
                return pd.DataFrame(data)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"导出指标失败: {e}")
            return None