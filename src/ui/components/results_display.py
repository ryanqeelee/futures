"""
结果显示组件

提供套利机会结果的展示和分析功能
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ResultsDisplay:
    """结果显示组件"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def render_results_overview(self, results: List[Dict]) -> None:
        """
        渲染结果概览
        
        Args:
            results: 套利机会结果列表
        """
        if not results:
            st.info("暂无扫描结果")
            return
        
        results_df = pd.DataFrame(results)
        
        # 概览指标
        st.subheader("📊 结果概览")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "发现机会",
                len(results_df),
                help="总共发现的套利机会数量"
            )
        
        with col2:
            if 'profit_margin' in results_df.columns:
                avg_profit = results_df['profit_margin'].mean() * 100
                st.metric(
                    "平均利润率",
                    f"{avg_profit:.2f}%",
                    help="所有机会的平均利润率"
                )
            else:
                st.metric("平均利润率", "N/A")
        
        with col3:
            if 'profit_margin' in results_df.columns:
                max_profit = results_df['profit_margin'].max() * 100
                st.metric(
                    "最高利润率",
                    f"{max_profit:.2f}%",
                    help="单个机会的最高利润率"
                )
            else:
                st.metric("最高利润率", "N/A")
        
        with col4:
            if 'risk_score' in results_df.columns:
                avg_risk = results_df['risk_score'].mean()
                st.metric(
                    "平均风险评分",
                    f"{avg_risk:.2f}",
                    help="平均风险评分（越低越好）"
                )
            else:
                st.metric("平均风险评分", "N/A")
    
    def render_results_table(self, 
                           results: List[Dict],
                           sortable: bool = True,
                           filterable: bool = True) -> None:
        """
        渲染结果表格
        
        Args:
            results: 套利机会结果列表
            sortable: 是否支持排序
            filterable: 是否支持筛选
        """
        if not results:
            return
        
        st.subheader("📋 详细结果")
        
        results_df = pd.DataFrame(results)
        
        # 筛选选项
        if filterable:
            self._render_filter_options(results_df)
        
        # 格式化显示数据
        display_df = self._format_display_data(results_df)
        
        # 显示表格
        st.dataframe(
            display_df,
            width='stretch',
            hide_index=True,
            column_config={
                "利润率": st.column_config.ProgressColumn(
                    "利润率",
                    help="预期利润率",
                    min_value=0,
                    max_value=0.1,
                    format="%.2f%%"
                ),
                "风险评分": st.column_config.ProgressColumn(
                    "风险评分",
                    help="风险评分（0-1，越低越好）",
                    min_value=0,
                    max_value=1,
                    format="%.2f"
                ),
                "置信度": st.column_config.ProgressColumn(
                    "置信度",
                    help="策略置信度",
                    min_value=0,
                    max_value=1,
                    format="%.2f"
                )
            }
        )
        
        # 导出选项
        if st.button("📥 导出结果"):
            csv_data = display_df.to_csv(index=False)
            st.download_button(
                label="下载CSV文件",
                data=csv_data,
                file_name=f"arbitrage_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    def _render_filter_options(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """渲染筛选选项"""
        with st.expander("🔍 筛选选项", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # 策略类型筛选
                if 'strategy_type' in results_df.columns:
                    strategy_types = results_df['strategy_type'].unique().tolist()
                    selected_strategies = st.multiselect(
                        "策略类型",
                        options=strategy_types,
                        default=strategy_types
                    )
                else:
                    selected_strategies = None
            
            with col2:
                # 利润率筛选
                if 'profit_margin' in results_df.columns:
                    min_profit = float(results_df['profit_margin'].min())
                    max_profit = float(results_df['profit_margin'].max())
                    
                    profit_range = st.slider(
                        "利润率范围",
                        min_value=min_profit,
                        max_value=max_profit,
                        value=(min_profit, max_profit),
                        format="%.3f"
                    )
                else:
                    profit_range = None
            
            with col3:
                # 风险评分筛选
                if 'risk_score' in results_df.columns:
                    min_risk = float(results_df['risk_score'].min())
                    max_risk = float(results_df['risk_score'].max())
                    
                    risk_range = st.slider(
                        "风险评分范围",
                        min_value=min_risk,
                        max_value=max_risk,
                        value=(min_risk, max_risk),
                        format="%.2f"
                    )
                else:
                    risk_range = None
            
            # 应用筛选
            filtered_df = results_df.copy()
            
            if selected_strategies and 'strategy_type' in results_df.columns:
                filtered_df = filtered_df[
                    filtered_df['strategy_type'].isin(selected_strategies)
                ]
            
            if profit_range and 'profit_margin' in results_df.columns:
                filtered_df = filtered_df[
                    (filtered_df['profit_margin'] >= profit_range[0]) &
                    (filtered_df['profit_margin'] <= profit_range[1])
                ]
            
            if risk_range and 'risk_score' in results_df.columns:
                filtered_df = filtered_df[
                    (filtered_df['risk_score'] >= risk_range[0]) &
                    (filtered_df['risk_score'] <= risk_range[1])
                ]
            
            return filtered_df
    
    def _format_display_data(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """格式化显示数据"""
        display_df = results_df.copy()
        
        # 格式化列名和数据
        column_mapping = {
            'id': 'ID',
            'strategy_type': '策略类型',
            'profit_margin': '利润率',
            'expected_profit': '预期利润',
            'risk_score': '风险评分',
            'confidence_score': '置信度',
            'instruments': '相关工具',
            'timestamp': '发现时间'
        }
        
        # 重命名列
        display_df = display_df.rename(columns=column_mapping)
        
        # 格式化数值
        if '利润率' in display_df.columns:
            display_df['利润率'] = display_df['利润率'].apply(lambda x: f"{x*100:.2f}%")
        
        if '预期利润' in display_df.columns:
            display_df['预期利润'] = display_df['预期利润'].apply(lambda x: f"{x:.2f}")
        
        if '风险评分' in display_df.columns:
            display_df['风险评分'] = display_df['风险评分'].apply(lambda x: f"{x:.2f}")
        
        if '置信度' in display_df.columns:
            display_df['置信度'] = display_df['置信度'].apply(lambda x: f"{x:.2f}")
        
        if '发现时间' in display_df.columns:
            display_df['发现时间'] = pd.to_datetime(display_df['发现时间']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return display_df
    
    def render_results_charts(self, results: List[Dict]) -> None:
        """
        渲染结果可视化图表
        
        Args:
            results: 套利机会结果列表
        """
        if not results:
            return
        
        results_df = pd.DataFrame(results)
        
        st.subheader("📈 结果可视化")
        
        # 创建图表标签页
        chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs([
            "📊 分布分析", "🎯 风险收益", "⏰ 时间分析", "🏆 策略对比"
        ])
        
        with chart_tab1:
            self._render_distribution_charts(results_df)
        
        with chart_tab2:
            self._render_risk_return_chart(results_df)
        
        with chart_tab3:
            self._render_time_analysis(results_df)
        
        with chart_tab4:
            self._render_strategy_comparison(results_df)
    
    def _render_distribution_charts(self, results_df: pd.DataFrame) -> None:
        """渲染分布分析图表"""
        col1, col2 = st.columns(2)
        
        with col1:
            # 利润率分布
            if 'profit_margin' in results_df.columns:
                fig_profit = px.histogram(
                    results_df,
                    x='profit_margin',
                    title='利润率分布',
                    labels={'profit_margin': '利润率', 'count': '数量'},
                    nbins=20
                )
                fig_profit.update_xaxis(tickformat='.2%')
                st.plotly_chart(fig_profit, width='stretch')
        
        with col2:
            # 风险评分分布
            if 'risk_score' in results_df.columns:
                fig_risk = px.histogram(
                    results_df,
                    x='risk_score',
                    title='风险评分分布',
                    labels={'risk_score': '风险评分', 'count': '数量'},
                    nbins=20
                )
                st.plotly_chart(fig_risk, width='stretch')
        
        # 置信度分布
        if 'confidence_score' in results_df.columns:
            fig_confidence = px.histogram(
                results_df,
                x='confidence_score',
                title='置信度分布',
                labels={'confidence_score': '置信度', 'count': '数量'},
                nbins=20
            )
            st.plotly_chart(fig_confidence, width='stretch')
    
    def _render_risk_return_chart(self, results_df: pd.DataFrame) -> None:
        """渲染风险收益分析图表"""
        if 'risk_score' not in results_df.columns or 'profit_margin' not in results_df.columns:
            st.info("缺少风险或收益数据")
            return
        
        # 散点图
        fig_scatter = px.scatter(
            results_df,
            x='risk_score',
            y='profit_margin',
            color='confidence_score' if 'confidence_score' in results_df.columns else None,
            size='expected_profit' if 'expected_profit' in results_df.columns else None,
            hover_data=['id', 'strategy_type'] if 'id' in results_df.columns else None,
            title='风险-收益分析',
            labels={
                'risk_score': '风险评分',
                'profit_margin': '利润率',
                'confidence_score': '置信度'
            }
        )
        
        fig_scatter.update_yaxis(tickformat='.2%')
        
        # 添加象限分割线
        if len(results_df) > 0:
            risk_median = results_df['risk_score'].median()
            profit_median = results_df['profit_margin'].median()
            
            fig_scatter.add_hline(
                y=profit_median,
                line_dash="dash",
                line_color="gray",
                annotation_text="利润中位数"
            )
            
            fig_scatter.add_vline(
                x=risk_median,
                line_dash="dash",
                line_color="gray",
                annotation_text="风险中位数"
            )
        
        st.plotly_chart(fig_scatter, width='stretch')
        
        # 风险收益四象限分析
        if len(results_df) > 0:
            self._render_quadrant_analysis(results_df)
    
    def _render_quadrant_analysis(self, results_df: pd.DataFrame) -> None:
        """渲染四象限分析"""
        if 'risk_score' not in results_df.columns or 'profit_margin' not in results_df.columns:
            return
        
        risk_median = results_df['risk_score'].median()
        profit_median = results_df['profit_margin'].median()
        
        # 分类机会
        quadrants = {
            '高收益低风险': len(results_df[
                (results_df['profit_margin'] > profit_median) & 
                (results_df['risk_score'] < risk_median)
            ]),
            '高收益高风险': len(results_df[
                (results_df['profit_margin'] > profit_median) & 
                (results_df['risk_score'] >= risk_median)
            ]),
            '低收益低风险': len(results_df[
                (results_df['profit_margin'] <= profit_median) & 
                (results_df['risk_score'] < risk_median)
            ]),
            '低收益高风险': len(results_df[
                (results_df['profit_margin'] <= profit_median) & 
                (results_df['risk_score'] >= risk_median)
            ])
        }
        
        # 创建饼图
        fig_quadrant = px.pie(
            values=list(quadrants.values()),
            names=list(quadrants.keys()),
            title='风险收益四象限分布'
        )
        
        st.plotly_chart(fig_quadrant, width='stretch')
    
    def _render_time_analysis(self, results_df: pd.DataFrame) -> None:
        """渲染时间分析图表"""
        if 'timestamp' not in results_df.columns:
            st.info("缺少时间数据")
            return
        
        # 转换时间列
        results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
        results_df['hour'] = results_df['timestamp'].dt.hour
        results_df['minute'] = results_df['timestamp'].dt.minute
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 按小时分布
            hourly_counts = results_df['hour'].value_counts().sort_index()
            
            fig_hourly = px.bar(
                x=hourly_counts.index,
                y=hourly_counts.values,
                title='机会发现时间分布（按小时）',
                labels={'x': '小时', 'y': '机会数量'}
            )
            
            st.plotly_chart(fig_hourly, width='stretch')
        
        with col2:
            # 时间序列
            if len(results_df) > 1:
                results_sorted = results_df.sort_values('timestamp')
                
                fig_timeline = px.line(
                    results_sorted,
                    x='timestamp',
                    y='profit_margin' if 'profit_margin' in results_df.columns else None,
                    title='利润率时间序列',
                    labels={'timestamp': '时间', 'profit_margin': '利润率'}
                )
                
                fig_timeline.update_yaxis(tickformat='.2%')
                st.plotly_chart(fig_timeline, width='stretch')
    
    def _render_strategy_comparison(self, results_df: pd.DataFrame) -> None:
        """渲染策略对比图表"""
        if 'strategy_type' not in results_df.columns:
            st.info("缺少策略类型数据")
            return
        
        # 策略统计
        strategy_stats = results_df.groupby('strategy_type').agg({
            'profit_margin': ['count', 'mean', 'std'],
            'risk_score': 'mean',
            'confidence_score': 'mean'
        }).round(4)
        
        # 扁平化列名
        strategy_stats.columns = ['机会数量', '平均利润率', '利润率标准差', '平均风险', '平均置信度']
        
        st.write("📊 **策略表现对比**")
        st.dataframe(
            strategy_stats,
            width='stretch'
        )
        
        # 可视化对比
        col1, col2 = st.columns(2)
        
        with col1:
            # 策略机会数量对比
            strategy_counts = results_df['strategy_type'].value_counts()
            
            fig_counts = px.bar(
                x=strategy_counts.values,
                y=strategy_counts.index,
                orientation='h',
                title='各策略发现机会数量',
                labels={'x': '机会数量', 'y': '策略类型'}
            )
            
            st.plotly_chart(fig_counts, width='stretch')
        
        with col2:
            # 策略平均利润率对比
            if 'profit_margin' in results_df.columns:
                avg_profits = results_df.groupby('strategy_type')['profit_margin'].mean().sort_values(ascending=False)
                
                fig_profits = px.bar(
                    x=avg_profits.values,
                    y=avg_profits.index,
                    orientation='h',
                    title='各策略平均利润率',
                    labels={'x': '平均利润率', 'y': '策略类型'}
                )
                
                fig_profits.update_xaxis(tickformat='.2%')
                st.plotly_chart(fig_profits, width='stretch')
    
    def render_detailed_opportunity(self, opportunity: Dict) -> None:
        """
        渲染单个机会的详细信息
        
        Args:
            opportunity: 套利机会详细信息
        """
        st.subheader(f"🎯 机会详情: {opportunity.get('id', 'Unknown')}")
        
        # 基本信息
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("策略类型", opportunity.get('strategy_type', 'N/A'))
            st.metric("利润率", f"{opportunity.get('profit_margin', 0)*100:.2f}%")
        
        with col2:
            st.metric("预期利润", f"{opportunity.get('expected_profit', 0):.2f}")
            st.metric("风险评分", f"{opportunity.get('risk_score', 0):.2f}")
        
        with col3:
            st.metric("置信度", f"{opportunity.get('confidence_score', 0):.2f}")
            st.metric("发现时间", opportunity.get('timestamp', 'N/A'))
        
        # 相关工具
        if 'instruments' in opportunity:
            st.write("**相关工具:**", opportunity['instruments'])
        
        # 交易动作
        if 'actions' in opportunity:
            st.write("**建议交易动作:**")
            for i, action in enumerate(opportunity['actions'], 1):
                st.write(f"{i}. {action}")
        
        # 风险提示
        st.warning("⚠️ 请注意：套利交易存在风险，实际执行前请进行充分的风险评估和资金管理。")
    
    def render_results_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """
        渲染结果摘要
        
        Args:
            results: 套利机会结果列表
            
        Returns:
            Dict: 摘要统计信息
        """
        if not results:
            st.info("暂无结果可供分析")
            return {}
        
        results_df = pd.DataFrame(results)
        
        # 计算摘要统计
        summary = {
            'total_opportunities': len(results_df),
            'avg_profit_margin': results_df['profit_margin'].mean() if 'profit_margin' in results_df.columns else 0,
            'max_profit_margin': results_df['profit_margin'].max() if 'profit_margin' in results_df.columns else 0,
            'avg_risk_score': results_df['risk_score'].mean() if 'risk_score' in results_df.columns else 0,
            'strategy_count': results_df['strategy_type'].nunique() if 'strategy_type' in results_df.columns else 0
        }
        
        # 渲染摘要
        st.markdown("### 📋 扫描摘要")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **发现机会**: {summary['total_opportunities']} 个
            
            **平均利润率**: {summary['avg_profit_margin']*100:.2f}%
            
            **最高利润率**: {summary['max_profit_margin']*100:.2f}%
            """)
        
        with col2:
            st.info(f"""
            **平均风险**: {summary['avg_risk_score']:.2f}
            
            **涉及策略**: {summary['strategy_count']} 种
            
            **扫描时间**: {datetime.now().strftime('%H:%M:%S')}
            """)
        
        return summary