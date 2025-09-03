"""
增强版结果显示组件

提供高级结果展示、多维度排序、交互式筛选和数据可视化功能
包含期权Greeks展示、风险指标计算、智能排序和导出功能
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64


class SortOrder(Enum):
    """排序顺序枚举"""
    ASC = "ascending"
    DESC = "descending"


class SortCriteria(Enum):
    """排序标准枚举"""
    PROFIT_MARGIN = "profit_margin"
    RISK_SCORE = "risk_score"
    CONFIDENCE_SCORE = "confidence_score"
    EXPECTED_PROFIT = "expected_profit"
    TIMESTAMP = "timestamp"
    STRATEGY_TYPE = "strategy_type"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"


@dataclass
class FilterCriteria:
    """筛选条件"""
    strategy_types: Optional[List[str]] = None
    profit_range: Optional[Tuple[float, float]] = None
    risk_range: Optional[Tuple[float, float]] = None
    confidence_range: Optional[Tuple[float, float]] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    min_expected_profit: Optional[float] = None
    max_risk_score: Optional[float] = None
    instruments: Optional[List[str]] = None


@dataclass
class GreeksData:
    """期权Greeks数据"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


class EnhancedResultsDisplay:
    """增强版结果显示组件"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._cached_df = None
        self._last_sort_criteria = None
        self._last_sort_order = None
        self._current_filters = FilterCriteria()
        
        # 初始化会话状态
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """初始化Streamlit会话状态"""
        default_states = {
            'results_sort_criteria': SortCriteria.PROFIT_MARGIN.value,
            'results_sort_order': SortOrder.DESC.value,
            'results_page_size': 50,
            'results_current_page': 0,
            'results_show_greeks': False,
            'results_show_risk_metrics': True,
            'results_export_format': 'CSV',
            'results_chart_type': '散点图',
            'multi_sort_enabled': False,
            'sort_criteria_list': []
        }
        
        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @st.cache_data
    def _prepare_display_data(_self, results: List[Dict]) -> pd.DataFrame:
        """准备显示数据（带缓存）"""
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # 计算额外指标
        if 'profit_margin' in df.columns and 'risk_score' in df.columns:
            # 夏普比率（简化版）
            df['sharpe_ratio'] = np.where(
                df['risk_score'] > 0,
                df['profit_margin'] / df['risk_score'],
                0
            )
        
        # 添加收益风险比
        if 'expected_profit' in df.columns and 'risk_score' in df.columns:
            df['profit_risk_ratio'] = np.where(
                df['risk_score'] > 0,
                df['expected_profit'] / df['risk_score'],
                0
            )
        
        # 转换时间戳
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def render_enhanced_overview(self, results: List[Dict]) -> None:
        """渲染增强版结果概览"""
        if not results:
            st.info("暂无扫描结果")
            return
        
        df = self._prepare_display_data(results)
        
        st.subheader("📊 增强版结果概览")
        
        # 主要指标
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "总机会数",
                len(df),
                help="发现的套利机会总数"
            )
        
        with col2:
            if 'profit_margin' in df.columns:
                avg_profit = df['profit_margin'].mean() * 100
                best_profit = df['profit_margin'].max() * 100
                delta = best_profit - avg_profit
                st.metric(
                    "平均利润率",
                    f"{avg_profit:.2f}%",
                    delta=f"+{delta:.2f}%",
                    help="所有机会的平均利润率"
                )
            else:
                st.metric("平均利润率", "N/A")
        
        with col3:
            if 'risk_score' in df.columns:
                avg_risk = df['risk_score'].mean()
                min_risk = df['risk_score'].min()
                risk_delta = min_risk - avg_risk
                st.metric(
                    "平均风险",
                    f"{avg_risk:.3f}",
                    delta=f"{risk_delta:.3f}",
                    delta_color="inverse",
                    help="平均风险评分（越低越好）"
                )
            else:
                st.metric("平均风险", "N/A")
        
        with col4:
            if 'sharpe_ratio' in df.columns:
                avg_sharpe = df['sharpe_ratio'].mean()
                st.metric(
                    "平均夏普比率",
                    f"{avg_sharpe:.2f}",
                    help="收益风险调整后的表现指标"
                )
            else:
                st.metric("平均夏普比率", "N/A")
        
        with col5:
            if 'strategy_type' in df.columns:
                strategy_count = df['strategy_type'].nunique()
                st.metric(
                    "策略数量",
                    strategy_count,
                    help="涉及的套利策略种类"
                )
            else:
                st.metric("策略数量", "N/A")
        
        # 详细统计表
        with st.expander("📈 详细统计信息", expanded=False):
            self._render_detailed_statistics(df)
    
    def _render_detailed_statistics(self, df: pd.DataFrame):
        """渲染详细统计信息"""
        if df.empty:
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**数值型指标统计**")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                stats_df = df[numeric_cols].describe().round(4)
                st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            st.write("**分类型指标统计**")
            if 'strategy_type' in df.columns:
                strategy_stats = df['strategy_type'].value_counts()
                fig = px.pie(
                    values=strategy_stats.values,
                    names=strategy_stats.index,
                    title="策略类型分布"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def render_advanced_sorting_controls(self, df: pd.DataFrame) -> pd.DataFrame:
        """渲染高级排序控制"""
        st.subheader("🔄 高级排序和筛选")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # 多级排序
            multi_sort = st.checkbox(
                "启用多级排序",
                value=st.session_state.get('multi_sort_enabled', False),
                help="支持按多个条件进行排序"
            )
            st.session_state.multi_sort_enabled = multi_sort
        
        if multi_sort:
            return self._render_multi_level_sorting(df)
        else:
            return self._render_single_level_sorting(df)
    
    def _render_single_level_sorting(self, df: pd.DataFrame) -> pd.DataFrame:
        """渲染单级排序"""
        col1, col2 = st.columns(2)
        
        with col1:
            # 排序字段选择
            available_columns = [col for col in df.columns if col in [
                'profit_margin', 'risk_score', 'confidence_score', 
                'expected_profit', 'timestamp', 'strategy_type', 'sharpe_ratio'
            ]]
            
            sort_by = st.selectbox(
                "排序字段",
                options=available_columns,
                index=available_columns.index(st.session_state.results_sort_criteria) 
                if st.session_state.results_sort_criteria in available_columns else 0,
                help="选择主要排序字段"
            )
            st.session_state.results_sort_criteria = sort_by
        
        with col2:
            # 排序方向
            sort_order = st.selectbox(
                "排序方向",
                options=[SortOrder.DESC.value, SortOrder.ASC.value],
                index=0 if st.session_state.results_sort_order == SortOrder.DESC.value else 1,
                format_func=lambda x: "降序 (高到低)" if x == SortOrder.DESC.value else "升序 (低到高)"
            )
            st.session_state.results_sort_order = sort_order
        
        # 应用排序
        ascending = sort_order == SortOrder.ASC.value
        sorted_df = df.sort_values(by=sort_by, ascending=ascending)
        
        return sorted_df
    
    def _render_multi_level_sorting(self, df: pd.DataFrame) -> pd.DataFrame:
        """渲染多级排序"""
        available_columns = [col for col in df.columns if col in [
            'profit_margin', 'risk_score', 'confidence_score', 
            'expected_profit', 'timestamp', 'strategy_type', 'sharpe_ratio'
        ]]
        
        # 排序条件列表
        if 'sort_criteria_list' not in st.session_state:
            st.session_state.sort_criteria_list = [
                {'column': 'profit_margin', 'ascending': False}
            ]
        
        st.write("**排序条件列表**")
        
        # 显示当前排序条件
        for i, criteria in enumerate(st.session_state.sort_criteria_list):
            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
            
            with col1:
                new_column = st.selectbox(
                    f"排序字段 {i+1}",
                    options=available_columns,
                    index=available_columns.index(criteria['column']) 
                    if criteria['column'] in available_columns else 0,
                    key=f"sort_col_{i}"
                )
                criteria['column'] = new_column
            
            with col2:
                ascending = st.selectbox(
                    f"方向 {i+1}",
                    options=[False, True],
                    index=0 if not criteria['ascending'] else 1,
                    format_func=lambda x: "降序" if not x else "升序",
                    key=f"sort_order_{i}"
                )
                criteria['ascending'] = ascending
            
            with col3:
                if st.button("🗑️", key=f"delete_{i}", help="删除此条件"):
                    st.session_state.sort_criteria_list.pop(i)
                    st.rerun()
            
            with col4:
                if i < len(st.session_state.sort_criteria_list) - 1:
                    if st.button("↑", key=f"up_{i}", help="上移"):
                        criteria_list = st.session_state.sort_criteria_list
                        criteria_list[i], criteria_list[i+1] = criteria_list[i+1], criteria_list[i]
                        st.rerun()
        
        # 添加新排序条件
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ 添加排序条件"):
                st.session_state.sort_criteria_list.append({
                    'column': available_columns[0],
                    'ascending': False
                })
                st.rerun()
        
        with col2:
            if st.button("🔄 重置排序"):
                st.session_state.sort_criteria_list = [
                    {'column': 'profit_margin', 'ascending': False}
                ]
                st.rerun()
        
        # 应用多级排序
        if st.session_state.sort_criteria_list:
            columns = [criteria['column'] for criteria in st.session_state.sort_criteria_list]
            ascending = [criteria['ascending'] for criteria in st.session_state.sort_criteria_list]
            sorted_df = df.sort_values(by=columns, ascending=ascending)
        else:
            sorted_df = df
        
        return sorted_df
    
    def render_advanced_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """渲染高级筛选器"""
        with st.expander("🔍 高级筛选器", expanded=False):
            filtered_df = df.copy()
            
            # 创建筛选器布局
            filter_tabs = st.tabs(["📊 数值筛选", "📋 分类筛选", "⏰ 时间筛选"])
            
            with filter_tabs[0]:  # 数值筛选
                filtered_df = self._render_numerical_filters(filtered_df)
            
            with filter_tabs[1]:  # 分类筛选
                filtered_df = self._render_categorical_filters(filtered_df)
            
            with filter_tabs[2]:  # 时间筛选
                filtered_df = self._render_time_filters(filtered_df)
            
            # 显示筛选结果统计
            st.markdown(f"**筛选结果**: {len(filtered_df)} / {len(df)} 条记录")
            
            return filtered_df
    
    def _render_numerical_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """渲染数值筛选器"""
        col1, col2 = st.columns(2)
        
        with col1:
            # 利润率筛选
            if 'profit_margin' in df.columns:
                min_profit = float(df['profit_margin'].min())
                max_profit = float(df['profit_margin'].max())
                
                profit_range = st.slider(
                    "利润率范围 (%)",
                    min_value=min_profit * 100,
                    max_value=max_profit * 100,
                    value=(min_profit * 100, max_profit * 100),
                    step=0.01,
                    format="%.2f%%"
                )
                
                df = df[
                    (df['profit_margin'] >= profit_range[0] / 100) &
                    (df['profit_margin'] <= profit_range[1] / 100)
                ]
            
            # 预期利润筛选
            if 'expected_profit' in df.columns:
                min_exp_profit = float(df['expected_profit'].min())
                max_exp_profit = float(df['expected_profit'].max())
                
                if max_exp_profit > min_exp_profit:
                    exp_profit_range = st.slider(
                        "预期利润范围",
                        min_value=min_exp_profit,
                        max_value=max_exp_profit,
                        value=(min_exp_profit, max_exp_profit),
                        step=0.01
                    )
                    
                    df = df[
                        (df['expected_profit'] >= exp_profit_range[0]) &
                        (df['expected_profit'] <= exp_profit_range[1])
                    ]
        
        with col2:
            # 风险评分筛选
            if 'risk_score' in df.columns:
                min_risk = float(df['risk_score'].min())
                max_risk = float(df['risk_score'].max())
                
                risk_range = st.slider(
                    "风险评分范围",
                    min_value=min_risk,
                    max_value=max_risk,
                    value=(min_risk, max_risk),
                    step=0.001,
                    format="%.3f"
                )
                
                df = df[
                    (df['risk_score'] >= risk_range[0]) &
                    (df['risk_score'] <= risk_range[1])
                ]
            
            # 置信度筛选
            if 'confidence_score' in df.columns:
                min_conf = float(df['confidence_score'].min())
                max_conf = float(df['confidence_score'].max())
                
                confidence_range = st.slider(
                    "置信度范围",
                    min_value=min_conf,
                    max_value=max_conf,
                    value=(min_conf, max_conf),
                    step=0.01
                )
                
                df = df[
                    (df['confidence_score'] >= confidence_range[0]) &
                    (df['confidence_score'] <= confidence_range[1])
                ]
        
        return df
    
    def _render_categorical_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """渲染分类筛选器"""
        col1, col2 = st.columns(2)
        
        with col1:
            # 策略类型筛选
            if 'strategy_type' in df.columns:
                strategy_types = df['strategy_type'].unique().tolist()
                selected_strategies = st.multiselect(
                    "策略类型",
                    options=strategy_types,
                    default=strategy_types,
                    help="选择要显示的策略类型"
                )
                
                if selected_strategies:
                    df = df[df['strategy_type'].isin(selected_strategies)]
        
        with col2:
            # 工具筛选
            if 'instruments' in df.columns:
                # 解析工具列表
                all_instruments = set()
                for instruments_str in df['instruments'].dropna():
                    if isinstance(instruments_str, str):
                        instruments = [inst.strip() for inst in instruments_str.split(',')]
                        all_instruments.update(instruments)
                
                if all_instruments:
                    selected_instruments = st.multiselect(
                        "相关工具",
                        options=sorted(list(all_instruments)),
                        help="选择要显示的相关工具"
                    )
                    
                    if selected_instruments:
                        def contains_instrument(instruments_str):
                            if pd.isna(instruments_str):
                                return False
                            instruments = [inst.strip() for inst in str(instruments_str).split(',')]
                            return any(inst in selected_instruments for inst in instruments)
                        
                        mask = df['instruments'].apply(contains_instrument)
                        df = df[mask]
        
        return df
    
    def _render_time_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """渲染时间筛选器"""
        if 'timestamp' not in df.columns:
            st.info("无时间数据可筛选")
            return df
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        min_time = df['timestamp'].min()
        max_time = df['timestamp'].max()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 快速时间选择
            time_filter_option = st.selectbox(
                "快速时间筛选",
                options=["全部", "最近1小时", "最近6小时", "最近24小时", "自定义范围"],
                help="选择时间筛选范围"
            )
        
        with col2:
            # 自定义时间范围
            if time_filter_option == "自定义范围":
                time_range = st.date_input(
                    "选择时间范围",
                    value=(min_time.date(), max_time.date()),
                    min_value=min_time.date(),
                    max_value=max_time.date()
                )
                
                if len(time_range) == 2:
                    start_date, end_date = time_range
                    df = df[
                        (df['timestamp'].dt.date >= start_date) &
                        (df['timestamp'].dt.date <= end_date)
                    ]
        
        # 应用快速时间筛选
        if time_filter_option != "全部" and time_filter_option != "自定义范围":
            now = datetime.now()
            if time_filter_option == "最近1小时":
                cutoff = now - timedelta(hours=1)
            elif time_filter_option == "最近6小时":
                cutoff = now - timedelta(hours=6)
            elif time_filter_option == "最近24小时":
                cutoff = now - timedelta(hours=24)
            else:
                cutoff = min_time
            
            df = df[df['timestamp'] >= cutoff]
        
        return df
    
    def render_paginated_table(self, df: pd.DataFrame) -> None:
        """渲染分页表格"""
        if df.empty:
            st.info("没有数据可显示")
            return
        
        st.subheader("📋 详细结果表格")
        
        # 分页控制
        col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
        
        with col1:
            page_size = st.selectbox(
                "每页显示",
                options=[25, 50, 100, 200],
                index=[25, 50, 100, 200].index(st.session_state.results_page_size),
                help="选择每页显示的记录数"
            )
            st.session_state.results_page_size = page_size
        
        # 计算分页
        total_rows = len(df)
        total_pages = (total_rows - 1) // page_size + 1 if total_rows > 0 else 0
        
        with col2:
            current_page = st.number_input(
                "页码",
                min_value=1,
                max_value=max(1, total_pages),
                value=min(st.session_state.results_current_page + 1, total_pages),
                step=1
            ) - 1
            st.session_state.results_current_page = current_page
        
        with col3:
            st.write(f"共 {total_pages} 页")
            st.write(f"总计 {total_rows} 条")
        
        with col4:
            # 分页导航按钮
            nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
            
            with nav_col1:
                if st.button("⏪", help="首页"):
                    st.session_state.results_current_page = 0
                    st.rerun()
            
            with nav_col2:
                if st.button("◀️", help="上一页"):
                    if st.session_state.results_current_page > 0:
                        st.session_state.results_current_page -= 1
                        st.rerun()
            
            with nav_col3:
                if st.button("▶️", help="下一页"):
                    if st.session_state.results_current_page < total_pages - 1:
                        st.session_state.results_current_page += 1
                        st.rerun()
            
            with nav_col4:
                if st.button("⏩", help="末页"):
                    st.session_state.results_current_page = max(0, total_pages - 1)
                    st.rerun()
        
        # 获取当前页数据
        start_idx = current_page * page_size
        end_idx = start_idx + page_size
        page_df = df.iloc[start_idx:end_idx]
        
        # 格式化显示数据
        display_df = self._format_display_data(page_df)
        
        # 显示表格
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config=self._get_column_config()
        )
    
    def _format_display_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """格式化显示数据"""
        if df.empty:
            return df
        
        display_df = df.copy()
        
        # 列名映射
        column_mapping = {
            'id': 'ID',
            'strategy_type': '策略类型',
            'profit_margin': '利润率',
            'expected_profit': '预期利润',
            'risk_score': '风险评分',
            'confidence_score': '置信度',
            'instruments': '相关工具',
            'timestamp': '发现时间',
            'sharpe_ratio': '夏普比率',
            'profit_risk_ratio': '收益风险比'
        }
        
        # 重命名存在的列
        rename_dict = {k: v for k, v in column_mapping.items() if k in display_df.columns}
        display_df = display_df.rename(columns=rename_dict)
        
        # 格式化数值列
        format_rules = {
            '利润率': lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A",
            '预期利润': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            '风险评分': lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A",
            '置信度': lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A",
            '夏普比率': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
            '收益风险比': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"
        }
        
        for col, formatter in format_rules.items():
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(formatter)
        
        # 格式化时间
        if '发现时间' in display_df.columns:
            display_df['发现时间'] = pd.to_datetime(display_df['发现时间']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return display_df
    
    def _get_column_config(self) -> Dict:
        """获取列配置"""
        return {
            "利润率": st.column_config.ProgressColumn(
                "利润率",
                help="预期利润率",
                min_value=0,
                max_value=0.2,
                format="%.2f%%"
            ),
            "风险评分": st.column_config.ProgressColumn(
                "风险评分",
                help="风险评分（越低越好）",
                min_value=0,
                max_value=1,
                format="%.3f"
            ),
            "置信度": st.column_config.ProgressColumn(
                "置信度",
                help="策略置信度",
                min_value=0,
                max_value=1,
                format="%.3f"
            ),
            "预期利润": st.column_config.NumberColumn(
                "预期利润",
                help="预期收益金额",
                format="%.2f"
            )
        }
    
    def render_greeks_display(self, results: List[Dict]) -> None:
        """渲染期权Greeks展示"""
        if not st.session_state.get('results_show_greeks', False):
            return
        
        st.subheader("📊 期权Greeks分析")
        
        # 模拟Greeks数据（实际应用中从results中提取）
        greeks_data = []
        for result in results:
            # 这里应该从result中提取实际的Greeks数据
            # 现在使用模拟数据作为示例
            greeks_data.append({
                'id': result.get('id', 'Unknown'),
                'delta': np.random.normal(0.5, 0.2),
                'gamma': np.random.normal(0.1, 0.05),
                'theta': np.random.normal(-0.05, 0.02),
                'vega': np.random.normal(0.15, 0.05),
                'rho': np.random.normal(0.02, 0.01)
            })
        
        if not greeks_data:
            st.info("暂无Greeks数据")
            return
        
        greeks_df = pd.DataFrame(greeks_data)
        
        # Greeks可视化
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Delta分布', 'Gamma分布', 'Theta分布', 'Vega分布', 'Rho分布', 'Greeks热力图'],
            specs=[[{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "histogram"}, {"type": "heatmap"}]]
        )
        
        # 添加各个Greeks的直方图
        greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
        positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
        
        for greek, (row, col) in zip(greeks, positions):
            fig.add_trace(
                go.Histogram(x=greeks_df[greek], name=greek.capitalize(), showlegend=False),
                row=row, col=col
            )
        
        # Greeks相关性热力图
        corr_matrix = greeks_df[greeks].corr()
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                showscale=True
            ),
            row=2, col=3
        )
        
        fig.update_layout(height=800, title_text="期权Greeks综合分析")
        st.plotly_chart(fig, use_container_width=True)
    
    def render_advanced_charts(self, df: pd.DataFrame) -> None:
        """渲染高级图表"""
        if df.empty:
            return
        
        st.subheader("📈 高级数据可视化")
        
        # 图表类型选择
        chart_type = st.selectbox(
            "选择图表类型",
            options=['散点图', '热力图', '3D散点图', '雷达图', '小提琴图', '箱线图'],
            index=['散点图', '热力图', '3D散点图', '雷达图', '小提琴图', '箱线图'].index(
                st.session_state.get('results_chart_type', '散点图')
            )
        )
        st.session_state.results_chart_type = chart_type
        
        # 根据选择的类型渲染图表
        if chart_type == '散点图':
            self._render_scatter_plot(df)
        elif chart_type == '热力图':
            self._render_heatmap(df)
        elif chart_type == '3D散点图':
            self._render_3d_scatter(df)
        elif chart_type == '雷达图':
            self._render_radar_chart(df)
        elif chart_type == '小提琴图':
            self._render_violin_plot(df)
        elif chart_type == '箱线图':
            self._render_box_plot(df)
    
    def _render_scatter_plot(self, df: pd.DataFrame):
        """渲染散点图"""
        if 'profit_margin' not in df.columns or 'risk_score' not in df.columns:
            st.warning("缺少必要数据列")
            return
        
        color_by = st.selectbox(
            "颜色编码",
            options=['confidence_score', 'strategy_type', 'sharpe_ratio'],
            help="选择用于颜色编码的字段"
        )
        
        size_by = st.selectbox(
            "大小编码",
            options=['expected_profit', 'confidence_score', None],
            help="选择用于大小编码的字段"
        )
        
        fig = px.scatter(
            df,
            x='risk_score',
            y='profit_margin',
            color=color_by if color_by in df.columns else None,
            size=size_by if size_by and size_by in df.columns else None,
            hover_data=['id', 'strategy_type'] if 'id' in df.columns else None,
            title='风险-收益分析（增强版）',
            labels={
                'risk_score': '风险评分',
                'profit_margin': '利润率',
                'confidence_score': '置信度',
                'sharpe_ratio': '夏普比率'
            }
        )
        
        fig.update_yaxis(tickformat='.2%')
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_heatmap(self, df: pd.DataFrame):
        """渲染相关性热力图"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("数值列不足，无法生成热力图")
            return
        
        # 计算相关性矩阵
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="指标相关性热力图",
            color_continuous_scale='RdBu_r'
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_3d_scatter(self, df: pd.DataFrame):
        """渲染3D散点图"""
        required_cols = ['profit_margin', 'risk_score', 'confidence_score']
        
        if not all(col in df.columns for col in required_cols):
            st.warning("缺少必要数据列用于3D显示")
            return
        
        fig = px.scatter_3d(
            df,
            x='risk_score',
            y='profit_margin',
            z='confidence_score',
            color='strategy_type' if 'strategy_type' in df.columns else None,
            size='expected_profit' if 'expected_profit' in df.columns else None,
            hover_data=['id'] if 'id' in df.columns else None,
            title='三维风险-收益-置信度分析',
            labels={
                'risk_score': '风险评分',
                'profit_margin': '利润率',
                'confidence_score': '置信度'
            }
        )
        
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_radar_chart(self, df: pd.DataFrame):
        """渲染雷达图"""
        if 'strategy_type' not in df.columns:
            st.warning("需要策略类型数据")
            return
        
        # 按策略类型聚合数据
        strategy_stats = df.groupby('strategy_type').agg({
            'profit_margin': 'mean',
            'risk_score': 'mean',
            'confidence_score': 'mean',
            'expected_profit': 'mean',
            'sharpe_ratio': 'mean' if 'sharpe_ratio' in df.columns else lambda x: 0
        }).round(4)
        
        # 标准化数据到0-1范围
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(strategy_stats)
        
        fig = go.Figure()
        
        categories = ['利润率', '风险评分', '置信度', '预期利润', '夏普比率']
        
        for i, strategy in enumerate(strategy_stats.index):
            fig.add_trace(go.Scatterpolar(
                r=normalized_data[i],
                theta=categories,
                fill='toself',
                name=strategy,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="策略表现雷达图"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_violin_plot(self, df: pd.DataFrame):
        """渲染小提琴图"""
        if 'strategy_type' not in df.columns or 'profit_margin' not in df.columns:
            st.warning("需要策略类型和利润率数据")
            return
        
        fig = px.violin(
            df,
            x='strategy_type',
            y='profit_margin',
            title='各策略利润率分布（小提琴图）',
            labels={
                'strategy_type': '策略类型',
                'profit_margin': '利润率'
            }
        )
        
        fig.update_yaxis(tickformat='.2%')
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_box_plot(self, df: pd.DataFrame):
        """渲染箱线图"""
        numeric_cols = ['profit_margin', 'risk_score', 'confidence_score', 'expected_profit']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if not available_cols:
            st.warning("没有可用的数值列")
            return
        
        selected_cols = st.multiselect(
            "选择要显示的指标",
            options=available_cols,
            default=available_cols[:3],
            help="选择要在箱线图中显示的指标"
        )
        
        if selected_cols:
            # 标准化数据用于比较
            normalized_df = df[selected_cols].copy()
            for col in selected_cols:
                col_data = normalized_df[col]
                normalized_df[col] = (col_data - col_data.mean()) / col_data.std()
            
            # 转换为长格式
            melted_df = normalized_df.melt(
                var_name='指标',
                value_name='标准化值'
            )
            
            fig = px.box(
                melted_df,
                x='指标',
                y='标准化值',
                title='指标分布箱线图（标准化）'
            )
            
            st.plotly_chart(fig, use_container_width=True)