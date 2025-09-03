"""
数据可视化组件

提供高级数据可视化功能，包括时间序列分析、策略对比、
风险热力图、收益分布分析等专业图表
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff


class DataVisualization:
    """数据可视化组件"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.color_palette = px.colors.qualitative.Set3
    
    def render_comprehensive_dashboard(self, results: List[Dict]) -> None:
        """渲染综合仪表板"""
        if not results:
            st.info("暂无数据可供可视化")
            return
        
        df = pd.DataFrame(results)
        
        # 创建可视化标签页
        tabs = st.tabs([
            "📊 概览仪表板",
            "📈 时间序列分析", 
            "🎯 策略效果分析",
            "🔥 热力图分析",
            "📋 分布分析",
            "🌐 相关性分析"
        ])
        
        with tabs[0]:
            self.render_overview_dashboard(df)
        
        with tabs[1]:
            self.render_time_series_analysis(df)
        
        with tabs[2]:
            self.render_strategy_performance_analysis(df)
        
        with tabs[3]:
            self.render_heatmap_analysis(df)
        
        with tabs[4]:
            self.render_distribution_analysis(df)
        
        with tabs[5]:
            self.render_correlation_analysis(df)
    
    def render_overview_dashboard(self, df: pd.DataFrame) -> None:
        """渲染概览仪表板"""
        st.subheader("📊 概览仪表板")
        
        # KPI指标
        self._render_kpi_metrics(df)
        
        # 核心图表
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_profit_risk_scatter(df)
        
        with col2:
            self._render_strategy_pie_chart(df)
        
        # 趋势图表
        self._render_opportunity_timeline(df)
    
    def _render_kpi_metrics(self, df: pd.DataFrame) -> None:
        """渲染KPI指标"""
        metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)
        
        with metrics_col1:
            total_opportunities = len(df)
            st.metric(
                label="🎯 总机会数",
                value=f"{total_opportunities:,}",
                help="发现的套利机会总数量"
            )
        
        with metrics_col2:
            if 'profit_margin' in df.columns:
                avg_profit = df['profit_margin'].mean() * 100
                max_profit = df['profit_margin'].max() * 100
                delta = max_profit - avg_profit
                st.metric(
                    label="💰 平均利润率",
                    value=f"{avg_profit:.2f}%",
                    delta=f"最高 +{delta:.2f}%",
                    help="所有机会的平均和最高利润率"
                )
            else:
                st.metric("💰 平均利润率", "N/A")
        
        with metrics_col3:
            if 'risk_score' in df.columns:
                avg_risk = df['risk_score'].mean()
                min_risk = df['risk_score'].min()
                st.metric(
                    label="⚠️ 平均风险",
                    value=f"{avg_risk:.3f}",
                    delta=f"最低 {min_risk:.3f}",
                    delta_color="inverse",
                    help="平均风险评分和最低风险"
                )
            else:
                st.metric("⚠️ 平均风险", "N/A")
        
        with metrics_col4:
            if 'expected_profit' in df.columns:
                total_expected = df['expected_profit'].sum()
                st.metric(
                    label="📈 总预期收益",
                    value=f"{total_expected:,.2f}",
                    help="所有机会的总预期收益"
                )
            else:
                st.metric("📈 总预期收益", "N/A")
        
        with metrics_col5:
            if 'confidence_score' in df.columns:
                avg_confidence = df['confidence_score'].mean()
                high_confidence_count = len(df[df['confidence_score'] > 0.8])
                st.metric(
                    label="🎖️ 平均置信度",
                    value=f"{avg_confidence:.3f}",
                    delta=f"{high_confidence_count} 个高置信",
                    help="平均置信度和高置信度机会数量"
                )
            else:
                st.metric("🎖️ 平均置信度", "N/A")
    
    def _render_profit_risk_scatter(self, df: pd.DataFrame) -> None:
        """渲染利润-风险散点图"""
        if 'profit_margin' not in df.columns or 'risk_score' not in df.columns:
            st.warning("缺少利润率或风险评分数据")
            return
        
        fig = px.scatter(
            df,
            x='risk_score',
            y='profit_margin',
            color='strategy_type' if 'strategy_type' in df.columns else None,
            size='confidence_score' if 'confidence_score' in df.columns else None,
            hover_data=['id'] if 'id' in df.columns else None,
            title='风险-收益分析',
            labels={
                'risk_score': '风险评分',
                'profit_margin': '利润率',
                'strategy_type': '策略类型',
                'confidence_score': '置信度'
            }
        )
        
        # 添加象限分割线
        if len(df) > 0:
            risk_median = df['risk_score'].median()
            profit_median = df['profit_margin'].median()
            
            fig.add_hline(y=profit_median, line_dash="dash", line_color="gray")
            fig.add_vline(x=risk_median, line_dash="dash", line_color="gray")
            
            # 添加象限标注
            fig.add_annotation(
                x=risk_median/2, y=profit_median*1.5,
                text="高收益<br>低风险", showarrow=False,
                font=dict(size=12, color="green"), bgcolor="lightgreen", opacity=0.7
            )
            fig.add_annotation(
                x=risk_median*1.5, y=profit_median*1.5,
                text="高收益<br>高风险", showarrow=False,
                font=dict(size=12, color="orange"), bgcolor="lightyellow", opacity=0.7
            )
        
        fig.update_yaxis(tickformat='.2%')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_strategy_pie_chart(self, df: pd.DataFrame) -> None:
        """渲染策略分布饼图"""
        if 'strategy_type' not in df.columns:
            st.warning("缺少策略类型数据")
            return
        
        strategy_counts = df['strategy_type'].value_counts()
        
        fig = px.pie(
            values=strategy_counts.values,
            names=strategy_counts.index,
            title='策略类型分布',
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_traces(textinfo='percent+label+value')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_opportunity_timeline(self, df: pd.DataFrame) -> None:
        """渲染机会时间线"""
        if 'timestamp' not in df.columns:
            st.warning("缺少时间戳数据")
            return
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 按小时聚合
        df['hour'] = df['timestamp'].dt.floor('H')
        hourly_stats = df.groupby('hour').agg({
            'profit_margin': ['count', 'mean'],
            'risk_score': 'mean',
            'expected_profit': 'sum'
        }).round(4)
        
        # 扁平化列名
        hourly_stats.columns = ['机会数量', '平均利润率', '平均风险', '总预期收益']
        hourly_stats.reset_index(inplace=True)
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('机会发现趋势', '平均利润率趋势', '平均风险趋势', '预期收益趋势'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 机会数量趋势
        fig.add_trace(
            go.Scatter(
                x=hourly_stats['hour'],
                y=hourly_stats['机会数量'],
                mode='lines+markers',
                name='机会数量',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # 平均利润率趋势
        fig.add_trace(
            go.Scatter(
                x=hourly_stats['hour'],
                y=hourly_stats['平均利润率'] * 100,
                mode='lines+markers',
                name='平均利润率(%)',
                line=dict(color='green', width=2),
                marker=dict(size=6)
            ),
            row=1, col=2
        )
        
        # 平均风险趋势
        fig.add_trace(
            go.Scatter(
                x=hourly_stats['hour'],
                y=hourly_stats['平均风险'],
                mode='lines+markers',
                name='平均风险',
                line=dict(color='red', width=2),
                marker=dict(size=6)
            ),
            row=2, col=1
        )
        
        # 总预期收益趋势
        fig.add_trace(
            go.Scatter(
                x=hourly_stats['hour'],
                y=hourly_stats['总预期收益'],
                mode='lines+markers',
                name='总预期收益',
                line=dict(color='purple', width=2),
                marker=dict(size=6)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="套利机会时间序列分析",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_time_series_analysis(self, df: pd.DataFrame) -> None:
        """渲染时间序列分析"""
        st.subheader("📈 时间序列分析")
        
        if 'timestamp' not in df.columns:
            st.warning("缺少时间戳数据")
            return
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 时间聚合选项
        col1, col2 = st.columns([1, 3])
        
        with col1:
            time_freq = st.selectbox(
                "时间粒度",
                options=['5min', '15min', '30min', '1H', '4H', '1D'],
                index=3,
                help="选择时间序列的聚合粒度"
            )
        
        # 按选定频率聚合数据
        df_resampled = df.set_index('timestamp').resample(time_freq).agg({
            'profit_margin': ['count', 'mean', 'std'],
            'risk_score': ['mean', 'std'],
            'confidence_score': 'mean',
            'expected_profit': 'sum'
        }).round(4)
        
        # 扁平化列名
        df_resampled.columns = [
            '机会数量', '平均利润率', '利润率标准差',
            '平均风险', '风险标准差', '平均置信度', '总预期收益'
        ]
        df_resampled.reset_index(inplace=True)
        df_resampled = df_resampled.dropna()
        
        if df_resampled.empty:
            st.warning("聚合后无数据")
            return
        
        # 创建交互式时间序列图表
        self._render_interactive_timeseries(df_resampled)
        
        # 季节性分析
        if len(df) > 24:  # 至少24个数据点才进行季节性分析
            self._render_seasonality_analysis(df)
        
        # 波动性分析
        self._render_volatility_analysis(df_resampled)
    
    def _render_interactive_timeseries(self, df: pd.DataFrame) -> None:
        """渲染交互式时间序列图表"""
        # 指标选择
        available_metrics = [col for col in df.columns if col != 'timestamp']
        selected_metrics = st.multiselect(
            "选择要显示的指标",
            options=available_metrics,
            default=['机会数量', '平均利润率', '平均风险'][:len(available_metrics)],
            help="选择要在时间序列中显示的指标"
        )
        
        if not selected_metrics:
            st.warning("请至少选择一个指标")
            return
        
        # 创建子图
        fig = make_subplots(
            rows=len(selected_metrics), cols=1,
            shared_xaxes=True,
            subplot_titles=selected_metrics,
            vertical_spacing=0.05
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, metric in enumerate(selected_metrics):
            if metric in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df[metric],
                        mode='lines+markers',
                        name=metric,
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=4),
                        hovertemplate=f'{metric}: %{{y}}<br>时间: %{{x}}<extra></extra>'
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            height=200 * len(selected_metrics),
            title_text="时间序列详细分析",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_seasonality_analysis(self, df: pd.DataFrame) -> None:
        """渲染季节性分析"""
        st.write("**季节性模式分析**")
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_name'] = df['timestamp'].dt.day_name()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 按小时分析
            hourly_pattern = df.groupby('hour').agg({
                'profit_margin': ['count', 'mean'],
                'risk_score': 'mean'
            }).round(4)
            hourly_pattern.columns = ['机会数量', '平均利润率', '平均风险']
            
            fig_hourly = px.bar(
                hourly_pattern,
                y='机会数量',
                title='按小时分布模式',
                labels={'index': '小时', 'y': '机会数量'}
            )
            fig_hourly.update_layout(height=400)
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            # 按星期分析
            weekly_pattern = df.groupby('day_name').agg({
                'profit_margin': ['count', 'mean'],
                'risk_score': 'mean'
            }).round(4)
            weekly_pattern.columns = ['机会数量', '平均利润率', '平均风险']
            
            # 重新排序星期
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_pattern = weekly_pattern.reindex([day for day in day_order if day in weekly_pattern.index])
            
            fig_weekly = px.bar(
                weekly_pattern,
                y='机会数量',
                title='按星期分布模式',
                labels={'index': '星期', 'y': '机会数量'}
            )
            fig_weekly.update_layout(height=400)
            st.plotly_chart(fig_weekly, use_container_width=True)
    
    def _render_volatility_analysis(self, df: pd.DataFrame) -> None:
        """渲染波动性分析"""
        st.write("**波动性分析**")
        
        if '利润率标准差' not in df.columns or '风险标准差' not in df.columns:
            st.info("无波动性数据")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 利润率波动性
            fig_profit_vol = go.Figure()
            
            fig_profit_vol.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['平均利润率'],
                mode='lines',
                name='平均利润率',
                line=dict(color='blue')
            ))
            
            # 添加置信区间
            upper_bound = df['平均利润率'] + df['利润率标准差']
            lower_bound = df['平均利润率'] - df['利润率标准差']
            
            fig_profit_vol.add_trace(go.Scatter(
                x=df['timestamp'].tolist() + df['timestamp'][::-1].tolist(),
                y=upper_bound.tolist() + lower_bound[::-1].tolist(),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='波动区间'
            ))
            
            fig_profit_vol.update_layout(
                title='利润率波动性分析',
                height=400
            )
            st.plotly_chart(fig_profit_vol, use_container_width=True)
        
        with col2:
            # 风险波动性
            fig_risk_vol = go.Figure()
            
            fig_risk_vol.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['平均风险'],
                mode='lines',
                name='平均风险',
                line=dict(color='red')
            ))
            
            # 添加置信区间
            upper_bound = df['平均风险'] + df['风险标准差']
            lower_bound = df['平均风险'] - df['风险标准差']
            
            fig_risk_vol.add_trace(go.Scatter(
                x=df['timestamp'].tolist() + df['timestamp'][::-1].tolist(),
                y=upper_bound.tolist() + lower_bound[::-1].tolist(),
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='波动区间'
            ))
            
            fig_risk_vol.update_layout(
                title='风险波动性分析',
                height=400
            )
            st.plotly_chart(fig_risk_vol, use_container_width=True)
    
    def render_strategy_performance_analysis(self, df: pd.DataFrame) -> None:
        """渲染策略效果分析"""
        st.subheader("🎯 策略效果分析")
        
        if 'strategy_type' not in df.columns:
            st.warning("缺少策略类型数据")
            return
        
        # 策略对比分析
        self._render_strategy_comparison(df)
        
        # 策略表现矩阵
        self._render_strategy_performance_matrix(df)
        
        # 策略效率分析
        self._render_strategy_efficiency_analysis(df)
    
    def _render_strategy_comparison(self, df: pd.DataFrame) -> None:
        """渲染策略对比分析"""
        st.write("**策略综合对比**")
        
        # 计算策略统计
        strategy_stats = df.groupby('strategy_type').agg({
            'profit_margin': ['count', 'mean', 'std', 'min', 'max'],
            'risk_score': ['mean', 'std'],
            'confidence_score': 'mean',
            'expected_profit': ['sum', 'mean']
        }).round(4)
        
        # 扁平化列名
        strategy_stats.columns = [
            '机会数量', '平均利润率', '利润率标准差', '最低利润率', '最高利润率',
            '平均风险', '风险标准差', '平均置信度', '总预期收益', '平均预期收益'
        ]
        
        # 计算风险调整收益
        strategy_stats['夏普比率'] = np.where(
            strategy_stats['风险标准差'] > 0,
            strategy_stats['平均利润率'] / strategy_stats['风险标准差'],
            0
        )
        
        # 显示统计表
        st.dataframe(
            strategy_stats,
            use_container_width=True,
            column_config={
                "平均利润率": st.column_config.NumberColumn(format="%.4f"),
                "平均风险": st.column_config.NumberColumn(format="%.4f"),
                "夏普比率": st.column_config.NumberColumn(format="%.2f")
            }
        )
        
        # 策略对比图表
        col1, col2 = st.columns(2)
        
        with col1:
            # 策略效率散点图
            fig_efficiency = px.scatter(
                strategy_stats,
                x='平均风险',
                y='平均利润率',
                size='机会数量',
                color='夏普比率',
                hover_name=strategy_stats.index,
                title='策略效率分析（风险-收益）',
                color_continuous_scale='Viridis'
            )
            fig_efficiency.update_yaxis(tickformat='.2%')
            st.plotly_chart(fig_efficiency, use_container_width=True)
        
        with col2:
            # 策略收益对比
            fig_returns = px.bar(
                strategy_stats,
                y=strategy_stats.index,
                x='平均利润率',
                orientation='h',
                title='策略平均收益对比',
                color='平均利润率',
                color_continuous_scale='Blues'
            )
            fig_returns.update_xaxis(tickformat='.2%')
            st.plotly_chart(fig_returns, use_container_width=True)
    
    def _render_strategy_performance_matrix(self, df: pd.DataFrame) -> None:
        """渲染策略表现矩阵"""
        st.write("**策略表现热力图**")
        
        # 创建表现矩阵
        strategies = df['strategy_type'].unique()
        
        if len(strategies) < 2:
            st.info("策略类型过少，无法生成矩阵")
            return
        
        # 按策略和时间段分析
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour_group'] = df['timestamp'].dt.hour // 4 * 4  # 4小时分组
        
        performance_matrix = df.groupby(['strategy_type', 'hour_group'])['profit_margin'].mean().unstack(fill_value=0)
        
        fig_matrix = px.imshow(
            performance_matrix,
            title='策略时段表现矩阵',
            labels=dict(x="时段", y="策略类型", color="平均利润率"),
            color_continuous_scale='RdYlGn'
        )
        
        fig_matrix.update_layout(height=400)
        st.plotly_chart(fig_matrix, use_container_width=True)
    
    def _render_strategy_efficiency_analysis(self, df: pd.DataFrame) -> None:
        """渲染策略效率分析"""
        st.write("**策略效率分析**")
        
        # 计算效率指标
        strategy_efficiency = df.groupby('strategy_type').apply(
            lambda x: pd.Series({
                '成功率': len(x[x['profit_margin'] > 0]) / len(x) if len(x) > 0 else 0,
                '平均收益': x['profit_margin'].mean(),
                '收益标准差': x['profit_margin'].std(),
                '最大收益': x['profit_margin'].max(),
                '最小收益': x['profit_margin'].min(),
                '平均风险': x['risk_score'].mean(),
                '风险调整收益': x['profit_margin'].mean() / x['risk_score'].mean() if x['risk_score'].mean() > 0 else 0
            })
        ).round(4)
        
        # 效率雷达图
        fig_radar = go.Figure()
        
        categories = ['成功率', '平均收益', '风险调整收益', '稳定性', '最大收益潜力']
        
        for strategy in strategy_efficiency.index:
            # 标准化指标到0-1范围
            values = [
                strategy_efficiency.loc[strategy, '成功率'],
                (strategy_efficiency.loc[strategy, '平均收益'] - strategy_efficiency['平均收益'].min()) / 
                (strategy_efficiency['平均收益'].max() - strategy_efficiency['平均收益'].min() + 1e-10),
                (strategy_efficiency.loc[strategy, '风险调整收益'] - strategy_efficiency['风险调整收益'].min()) / 
                (strategy_efficiency['风险调整收益'].max() - strategy_efficiency['风险调整收益'].min() + 1e-10),
                1 - (strategy_efficiency.loc[strategy, '收益标准差'] - strategy_efficiency['收益标准差'].min()) / 
                (strategy_efficiency['收益标准差'].max() - strategy_efficiency['收益标准差'].min() + 1e-10),
                (strategy_efficiency.loc[strategy, '最大收益'] - strategy_efficiency['最大收益'].min()) / 
                (strategy_efficiency['最大收益'].max() - strategy_efficiency['最大收益'].min() + 1e-10)
            ]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=strategy,
                line=dict(width=2)
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="策略效率雷达图",
            height=500
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    def render_heatmap_analysis(self, df: pd.DataFrame) -> None:
        """渲染热力图分析"""
        st.subheader("🔥 热力图分析")
        
        # 相关性热力图
        self._render_correlation_heatmap(df)
        
        # 时间-策略热力图
        if 'strategy_type' in df.columns and 'timestamp' in df.columns:
            self._render_time_strategy_heatmap(df)
        
        # 风险-收益热力图
        self._render_risk_return_heatmap(df)
    
    def _render_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """渲染相关性热力图"""
        st.write("**指标相关性热力图**")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("数值列不足")
            return
        
        # 计算相关性矩阵
        corr_matrix = df[numeric_cols].corr()
        
        # 创建带注释的热力图
        fig = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.index),
            annotation_text=corr_matrix.round(2).values,
            colorscale='RdBu',
            reversescale=True,
            showscale=True
        )
        
        fig.update_layout(
            title='指标相关性分析',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_time_strategy_heatmap(self, df: pd.DataFrame) -> None:
        """渲染时间-策略热力图"""
        st.write("**时间-策略表现热力图**")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        
        # 创建时间-策略矩阵
        time_strategy_matrix = df.groupby(['hour', 'strategy_type'])['profit_margin'].mean().unstack(fill_value=0)
        
        if time_strategy_matrix.empty:
            st.info("数据不足以生成时间-策略热力图")
            return
        
        fig = px.imshow(
            time_strategy_matrix.T,  # 转置以便更好显示
            title='各策略按小时表现热力图',
            labels=dict(x="小时", y="策略类型", color="平均利润率"),
            color_continuous_scale='RdYlGn',
            aspect="auto"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_risk_return_heatmap(self, df: pd.DataFrame) -> None:
        """渲染风险-收益热力图"""
        st.write("**风险-收益分布热力图**")
        
        if 'profit_margin' not in df.columns or 'risk_score' not in df.columns:
            st.warning("缺少利润率或风险评分数据")
            return
        
        # 创建2D直方图
        fig = px.density_heatmap(
            df,
            x='risk_score',
            y='profit_margin',
            title='风险-收益分布密度图',
            labels={
                'risk_score': '风险评分',
                'profit_margin': '利润率'
            },
            color_continuous_scale='Viridis'
        )
        
        fig.update_yaxis(tickformat='.2%')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_distribution_analysis(self, df: pd.DataFrame) -> None:
        """渲染分布分析"""
        st.subheader("📋 分布分析")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("无数值列可供分析")
            return
        
        # 指标选择
        selected_col = st.selectbox(
            "选择要分析的指标",
            options=numeric_cols,
            help="选择要进行分布分析的数值指标"
        )
        
        # 分布分析标签页
        dist_tabs = st.tabs(["📊 直方图", "📈 概率密度", "📉 累积分布", "📋 统计检验"])
        
        with dist_tabs[0]:
            self._render_histogram_analysis(df, selected_col)
        
        with dist_tabs[1]:
            self._render_density_analysis(df, selected_col)
        
        with dist_tabs[2]:
            self._render_cumulative_distribution(df, selected_col)
        
        with dist_tabs[3]:
            self._render_statistical_tests(df, selected_col)
    
    def _render_histogram_analysis(self, df: pd.DataFrame, column: str) -> None:
        """渲染直方图分析"""
        col1, col2 = st.columns(2)
        
        with col1:
            # 整体分布
            fig_hist = px.histogram(
                df,
                x=column,
                title=f'{column} 分布直方图',
                nbins=30
            )
            fig_hist.add_vline(x=df[column].mean(), line_dash="dash", line_color="red", annotation_text="均值")
            fig_hist.add_vline(x=df[column].median(), line_dash="dash", line_color="green", annotation_text="中位数")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # 按策略分布
            if 'strategy_type' in df.columns:
                fig_hist_strategy = px.histogram(
                    df,
                    x=column,
                    color='strategy_type',
                    title=f'{column} 按策略分布',
                    nbins=20,
                    opacity=0.7
                )
                st.plotly_chart(fig_hist_strategy, use_container_width=True)
    
    def _render_density_analysis(self, df: pd.DataFrame, column: str) -> None:
        """渲染概率密度分析"""
        from scipy import stats
        
        # 核密度估计
        data = df[column].dropna()
        
        if len(data) < 2:
            st.warning("数据点不足")
            return
        
        # 使用scipy进行核密度估计
        density = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 100)
        density_values = density(x_range)
        
        fig_density = go.Figure()
        
        # 添加密度曲线
        fig_density.add_trace(go.Scatter(
            x=x_range,
            y=density_values,
            mode='lines',
            name='概率密度',
            line=dict(color='blue', width=2)
        ))
        
        # 添加数据点
        fig_density.add_trace(go.Scatter(
            x=data,
            y=np.zeros(len(data)),
            mode='markers',
            name='数据点',
            marker=dict(color='red', size=3),
            yaxis='y2'
        ))
        
        fig_density.update_layout(
            title=f'{column} 概率密度分析',
            xaxis_title=column,
            yaxis_title='密度',
            yaxis2=dict(overlaying='y', side='right', showgrid=False)
        )
        
        st.plotly_chart(fig_density, use_container_width=True)
    
    def _render_cumulative_distribution(self, df: pd.DataFrame, column: str) -> None:
        """渲染累积分布"""
        data = df[column].dropna().sort_values()
        
        if len(data) < 2:
            st.warning("数据点不足")
            return
        
        # 计算累积分布
        y_values = np.arange(1, len(data) + 1) / len(data)
        
        fig_cdf = go.Figure()
        
        fig_cdf.add_trace(go.Scatter(
            x=data,
            y=y_values,
            mode='lines',
            name='累积分布函数',
            line=dict(color='green', width=2)
        ))
        
        # 添加分位数线
        quantiles = [0.25, 0.5, 0.75]
        for q in quantiles:
            value = data.quantile(q)
            fig_cdf.add_vline(
                x=value,
                line_dash="dash",
                line_color="red",
                annotation_text=f"P{int(q*100)}={value:.3f}"
            )
        
        fig_cdf.update_layout(
            title=f'{column} 累积分布函数',
            xaxis_title=column,
            yaxis_title='累积概率'
        )
        
        st.plotly_chart(fig_cdf, use_container_width=True)
    
    def _render_statistical_tests(self, df: pd.DataFrame, column: str) -> None:
        """渲染统计检验结果"""
        from scipy import stats
        
        data = df[column].dropna()
        
        if len(data) < 3:
            st.warning("数据点不足进行统计检验")
            return
        
        # 基本统计量
        st.write("**基本统计量**")
        basic_stats = {
            '数量': len(data),
            '均值': data.mean(),
            '中位数': data.median(),
            '标准差': data.std(),
            '偏度': stats.skew(data),
            '峰度': stats.kurtosis(data),
            '最小值': data.min(),
            '最大值': data.max()
        }
        
        stats_df = pd.DataFrame(list(basic_stats.items()), columns=['统计量', '值'])
        stats_df['值'] = stats_df['值'].round(4)
        st.dataframe(stats_df, hide_index=True)
        
        # 正态性检验
        st.write("**正态性检验**")
        if len(data) >= 8:  # Shapiro-Wilk检验要求
            shapiro_stat, shapiro_p = stats.shapiro(data)
            st.write(f"Shapiro-Wilk检验: 统计量={shapiro_stat:.4f}, p值={shapiro_p:.4f}")
            
            if shapiro_p > 0.05:
                st.success("✅ 数据可能服从正态分布")
            else:
                st.warning("⚠️ 数据可能不服从正态分布")
        
        # 分位数信息
        st.write("**分位数信息**")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = [data.quantile(p/100) for p in percentiles]
        
        percentile_df = pd.DataFrame({
            '分位数': [f'P{p}' for p in percentiles],
            '值': percentile_values
        })
        percentile_df['值'] = percentile_df['值'].round(4)
        st.dataframe(percentile_df, hide_index=True)
    
    def render_correlation_analysis(self, df: pd.DataFrame) -> None:
        """渲染相关性分析"""
        st.subheader("🌐 相关性分析")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("数值列不足进行相关性分析")
            return
        
        # 相关性矩阵
        corr_matrix = df[numeric_cols].corr()
        
        # 相关性分析标签页
        corr_tabs = st.tabs(["🔥 热力图", "🌐 网络图", "📊 强相关对", "📈 散点矩阵"])
        
        with corr_tabs[0]:
            self._render_correlation_heatmap_detailed(corr_matrix)
        
        with corr_tabs[1]:
            self._render_correlation_network(corr_matrix)
        
        with corr_tabs[2]:
            self._render_strong_correlations(corr_matrix)
        
        with corr_tabs[3]:
            self._render_scatter_matrix(df, numeric_cols)
    
    def _render_correlation_heatmap_detailed(self, corr_matrix: pd.DataFrame) -> None:
        """渲染详细相关性热力图"""
        # 创建掩码用于上三角
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        corr_matrix_masked = corr_matrix.mask(mask)
        
        fig = px.imshow(
            corr_matrix_masked,
            title='指标相关性热力图（下三角）',
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        
        # 添加相关系数注释
        annotations = []
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                if not mask[i, j] and not np.isnan(corr_matrix.iloc[i, j]):
                    annotations.append(
                        dict(
                            x=j, y=i,
                            text=str(round(corr_matrix.iloc[i, j], 2)),
                            showarrow=False,
                            font=dict(color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
                        )
                    )
        
        fig.update_layout(annotations=annotations, height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_correlation_network(self, corr_matrix: pd.DataFrame) -> None:
        """渲染相关性网络图"""
        # 只显示强相关关系（绝对值>0.3）
        strong_corr = corr_matrix.abs() > 0.3
        
        # 创建网络图数据
        edges = []
        for i in range(len(corr_matrix.index)):
            for j in range(i+1, len(corr_matrix.columns)):
                if strong_corr.iloc[i, j]:
                    edges.append({
                        'source': corr_matrix.index[i],
                        'target': corr_matrix.columns[j],
                        'weight': abs(corr_matrix.iloc[i, j]),
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        if not edges:
            st.info("没有发现强相关关系（|r| > 0.3）")
            return
        
        # 简化的网络图显示
        st.write("**强相关关系网络**")
        edges_df = pd.DataFrame(edges)
        edges_df['相关性'] = edges_df['correlation'].round(3)
        edges_df['强度'] = edges_df['weight'].round(3)
        
        st.dataframe(
            edges_df[['source', 'target', '相关性', '强度']].rename(columns={
                'source': '指标1',
                'target': '指标2'
            }),
            hide_index=True
        )
    
    def _render_strong_correlations(self, corr_matrix: pd.DataFrame) -> None:
        """渲染强相关对分析"""
        st.write("**强相关关系排序**")
        
        # 提取相关系数（去除对角线和重复）
        correlations = []
        for i in range(len(corr_matrix.index)):
            for j in range(i+1, len(corr_matrix.columns)):
                correlations.append({
                    '指标1': corr_matrix.index[i],
                    '指标2': corr_matrix.columns[j],
                    '相关系数': corr_matrix.iloc[i, j],
                    '绝对值': abs(corr_matrix.iloc[i, j])
                })
        
        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values('绝对值', ascending=False)
        
        # 显示前10个强相关关系
        st.dataframe(
            corr_df.head(10)[['指标1', '指标2', '相关系数']],
            hide_index=True,
            column_config={
                "相关系数": st.column_config.NumberColumn(format="%.4f")
            }
        )
        
        # 相关性强度分布
        fig_corr_dist = px.histogram(
            corr_df,
            x='绝对值',
            title='相关性强度分布',
            labels={'绝对值': '相关系数绝对值', 'count': '频数'}
        )
        st.plotly_chart(fig_corr_dist, use_container_width=True)
    
    def _render_scatter_matrix(self, df: pd.DataFrame, numeric_cols: List[str]) -> None:
        """渲染散点图矩阵"""
        # 限制显示的列数（避免图表过于复杂）
        max_cols = 6
        selected_cols = st.multiselect(
            f"选择要显示的指标（最多{max_cols}个）",
            options=numeric_cols,
            default=numeric_cols[:min(max_cols, len(numeric_cols))],
            max_selections=max_cols
        )
        
        if len(selected_cols) < 2:
            st.warning("请至少选择2个指标")
            return
        
        # 创建散点图矩阵
        fig = px.scatter_matrix(
            df[selected_cols].sample(min(1000, len(df))),  # 限制数据点数量
            title="散点图矩阵",
            opacity=0.6
        )
        
        fig.update_layout(
            height=800,
            width=800
        )
        
        st.plotly_chart(fig, use_container_width=True)