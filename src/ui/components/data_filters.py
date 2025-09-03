"""
数据筛选组件

提供高级数据筛选功能，包括多条件筛选、预设筛选器、
自定义筛选逻辑和筛选历史管理
"""

import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import json

import streamlit as st
import pandas as pd
import numpy as np
from streamlit_tags import st_tags


class FilterOperator(Enum):
    """筛选操作符"""
    EQUALS = "="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    BETWEEN = "between"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


class FilterType(Enum):
    """筛选器类型"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical" 
    TEXT = "text"
    DATETIME = "datetime"
    BOOLEAN = "boolean"


@dataclass
class FilterCondition:
    """筛选条件"""
    column: str
    operator: FilterOperator
    value: Any = None
    value2: Any = None  # 用于between操作
    enabled: bool = True
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'column': self.column,
            'operator': self.operator.value,
            'value': self.value,
            'value2': self.value2,
            'enabled': self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FilterCondition':
        """从字典创建"""
        return cls(
            column=data['column'],
            operator=FilterOperator(data['operator']),
            value=data.get('value'),
            value2=data.get('value2'),
            enabled=data.get('enabled', True)
        )


@dataclass
class FilterPreset:
    """筛选预设"""
    name: str
    description: str
    conditions: List[FilterCondition]
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'name': self.name,
            'description': self.description,
            'conditions': [cond.to_dict() for cond in self.conditions],
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FilterPreset':
        """从字典创建"""
        return cls(
            name=data['name'],
            description=data['description'],
            conditions=[FilterCondition.from_dict(cond) for cond in data['conditions']],
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat()))
        )


class AdvancedDataFilters:
    """高级数据筛选器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """初始化会话状态"""
        if 'filter_conditions' not in st.session_state:
            st.session_state.filter_conditions = []
        
        if 'filter_presets' not in st.session_state:
            st.session_state.filter_presets = self._get_default_presets()
        
        if 'filter_history' not in st.session_state:
            st.session_state.filter_history = []
        
        if 'current_preset' not in st.session_state:
            st.session_state.current_preset = None
        
        if 'filter_logic' not in st.session_state:
            st.session_state.filter_logic = "AND"
    
    def _get_default_presets(self) -> List[FilterPreset]:
        """获取默认预设筛选器"""
        return [
            FilterPreset(
                name="高收益机会",
                description="利润率>2%且风险评分<0.5的机会",
                conditions=[
                    FilterCondition("profit_margin", FilterOperator.GREATER_THAN, 0.02),
                    FilterCondition("risk_score", FilterOperator.LESS_THAN, 0.5)
                ]
            ),
            FilterPreset(
                name="保守投资",
                description="风险评分<0.3且置信度>0.8的机会",
                conditions=[
                    FilterCondition("risk_score", FilterOperator.LESS_THAN, 0.3),
                    FilterCondition("confidence_score", FilterOperator.GREATER_THAN, 0.8)
                ]
            ),
            FilterPreset(
                name="高频交易",
                description="最近1小时内发现的机会",
                conditions=[
                    FilterCondition("timestamp", FilterOperator.GREATER_THAN, 
                                  datetime.now() - timedelta(hours=1))
                ]
            ),
            FilterPreset(
                name="最佳平衡",
                description="夏普比率>1.5的机会",
                conditions=[
                    FilterCondition("sharpe_ratio", FilterOperator.GREATER_THAN, 1.5)
                ]
            )
        ]
    
    def render_filter_interface(self, df: pd.DataFrame) -> pd.DataFrame:
        """渲染筛选界面"""
        if df.empty:
            st.warning("无数据可供筛选")
            return df
        
        st.subheader("🔍 高级数据筛选器")
        
        # 筛选器标签页
        filter_tabs = st.tabs([
            "🎯 快速筛选",
            "⚙️ 自定义筛选", 
            "📋 预设筛选器",
            "📊 筛选历史",
            "🔧 高级设置"
        ])
        
        with filter_tabs[0]:
            filtered_df = self._render_quick_filters(df)
        
        with filter_tabs[1]:
            filtered_df = self._render_custom_filters(df)
        
        with filter_tabs[2]:
            filtered_df = self._render_preset_filters(df)
        
        with filter_tabs[3]:
            self._render_filter_history(df)
            filtered_df = df  # 历史标签页不直接筛选
        
        with filter_tabs[4]:
            filtered_df = self._render_advanced_settings(df)
        
        # 显示筛选结果统计
        self._render_filter_results_summary(df, filtered_df)
        
        return filtered_df
    
    def _render_quick_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """渲染快速筛选器"""
        st.write("**快速筛选选项**")
        
        filtered_df = df.copy()
        
        # 创建快速筛选区域
        quick_col1, quick_col2, quick_col3 = st.columns(3)
        
        with quick_col1:
            # 利润率快速筛选
            if 'profit_margin' in df.columns:
                profit_options = [
                    "全部", ">0%", ">1%", ">2%", ">5%", "定制范围"
                ]
                profit_filter = st.selectbox(
                    "💰 利润率筛选",
                    options=profit_options,
                    help="选择利润率筛选条件"
                )
                
                if profit_filter == ">0%":
                    filtered_df = filtered_df[filtered_df['profit_margin'] > 0]
                elif profit_filter == ">1%":
                    filtered_df = filtered_df[filtered_df['profit_margin'] > 0.01]
                elif profit_filter == ">2%":
                    filtered_df = filtered_df[filtered_df['profit_margin'] > 0.02]
                elif profit_filter == ">5%":
                    filtered_df = filtered_df[filtered_df['profit_margin'] > 0.05]
                elif profit_filter == "定制范围":
                    min_profit = st.number_input(
                        "最小利润率 (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=0.0,
                        step=0.1
                    ) / 100
                    filtered_df = filtered_df[filtered_df['profit_margin'] >= min_profit]
        
        with quick_col2:
            # 风险快速筛选
            if 'risk_score' in df.columns:
                risk_options = [
                    "全部", "<0.1 (极低)", "<0.3 (低)", "<0.5 (中)", "<0.7 (高)", "定制范围"
                ]
                risk_filter = st.selectbox(
                    "⚠️ 风险筛选",
                    options=risk_options,
                    help="选择风险水平筛选条件"
                )
                
                risk_thresholds = {
                    "<0.1 (极低)": 0.1,
                    "<0.3 (低)": 0.3,
                    "<0.5 (中)": 0.5,
                    "<0.7 (高)": 0.7
                }
                
                if risk_filter in risk_thresholds:
                    threshold = risk_thresholds[risk_filter]
                    filtered_df = filtered_df[filtered_df['risk_score'] < threshold]
                elif risk_filter == "定制范围":
                    max_risk = st.number_input(
                        "最大风险评分",
                        min_value=0.0,
                        max_value=1.0,
                        value=1.0,
                        step=0.01
                    )
                    filtered_df = filtered_df[filtered_df['risk_score'] <= max_risk]
        
        with quick_col3:
            # 策略类型快速筛选
            if 'strategy_type' in df.columns:
                strategy_types = ['全部'] + sorted(df['strategy_type'].unique().tolist())
                selected_strategy = st.selectbox(
                    "🎯 策略类型",
                    options=strategy_types,
                    help="选择策略类型筛选"
                )
                
                if selected_strategy != "全部":
                    filtered_df = filtered_df[filtered_df['strategy_type'] == selected_strategy]
        
        # 时间范围快速筛选
        if 'timestamp' in df.columns:
            st.write("**时间范围筛选**")
            time_col1, time_col2 = st.columns(2)
            
            with time_col1:
                time_options = [
                    "全部时间", "最近1小时", "最近6小时", 
                    "最近24小时", "最近7天", "自定义范围"
                ]
                time_filter = st.selectbox(
                    "⏰ 时间范围",
                    options=time_options
                )
            
            with time_col2:
                if time_filter != "全部时间":
                    now = datetime.now()
                    time_thresholds = {
                        "最近1小时": now - timedelta(hours=1),
                        "最近6小时": now - timedelta(hours=6),
                        "最近24小时": now - timedelta(hours=24),
                        "最近7天": now - timedelta(days=7)
                    }
                    
                    if time_filter in time_thresholds:
                        cutoff_time = time_thresholds[time_filter]
                        filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
                        filtered_df = filtered_df[filtered_df['timestamp'] >= cutoff_time]
                    elif time_filter == "自定义范围":
                        date_range = st.date_input(
                            "选择日期范围",
                            value=(datetime.now().date() - timedelta(days=7), datetime.now().date()),
                            help="选择开始和结束日期"
                        )
                        if len(date_range) == 2:
                            start_date, end_date = date_range
                            filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
                            filtered_df = filtered_df[
                                (filtered_df['timestamp'].dt.date >= start_date) &
                                (filtered_df['timestamp'].dt.date <= end_date)
                            ]
        
        return filtered_df
    
    def _render_custom_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """渲染自定义筛选器"""
        st.write("**自定义筛选条件**")
        
        # 筛选逻辑选择
        col1, col2 = st.columns([1, 3])
        with col1:
            logic_option = st.selectbox(
                "条件逻辑",
                options=["AND", "OR"],
                index=0 if st.session_state.filter_logic == "AND" else 1,
                help="多个条件之间的逻辑关系"
            )
            st.session_state.filter_logic = logic_option
        
        # 显示现有筛选条件
        st.write(f"**当前筛选条件 ({len(st.session_state.filter_conditions)} 个)**")
        
        if st.session_state.filter_conditions:
            for i, condition in enumerate(st.session_state.filter_conditions):
                self._render_filter_condition_editor(df, i, condition)
        else:
            st.info("暂无筛选条件，请添加新条件")
        
        # 添加新筛选条件
        st.write("**添加筛选条件**")
        self._render_add_filter_condition(df)
        
        # 筛选条件管理按钮
        manage_col1, manage_col2, manage_col3 = st.columns(3)
        
        with manage_col1:
            if st.button("🗑️ 清除所有条件"):
                st.session_state.filter_conditions = []
                st.rerun()
        
        with manage_col2:
            if st.button("💾 保存为预设"):
                self._show_save_preset_dialog()
        
        with manage_col3:
            if st.button("🔄 重置为默认"):
                st.session_state.filter_conditions = []
                st.session_state.filter_logic = "AND"
                st.rerun()
        
        # 应用筛选条件
        return self._apply_filter_conditions(df, st.session_state.filter_conditions)
    
    def _render_filter_condition_editor(self, df: pd.DataFrame, index: int, condition: FilterCondition):
        """渲染单个筛选条件编辑器"""
        with st.expander(f"条件 {index + 1}: {condition.column} {condition.operator.value}", 
                        expanded=False):
            
            cond_col1, cond_col2, cond_col3, cond_col4 = st.columns([2, 2, 3, 1])
            
            with cond_col1:
                # 列选择
                available_columns = df.columns.tolist()
                current_column_idx = available_columns.index(condition.column) if condition.column in available_columns else 0
                
                new_column = st.selectbox(
                    "字段",
                    options=available_columns,
                    index=current_column_idx,
                    key=f"condition_column_{index}"
                )
                condition.column = new_column
            
            with cond_col2:
                # 操作符选择
                column_type = self._get_column_type(df, condition.column)
                available_operators = self._get_available_operators(column_type)
                
                operator_values = [op.value for op in available_operators]
                current_op_idx = operator_values.index(condition.operator.value) if condition.operator.value in operator_values else 0
                
                new_operator = st.selectbox(
                    "操作符",
                    options=operator_values,
                    index=current_op_idx,
                    key=f"condition_operator_{index}"
                )
                condition.operator = FilterOperator(new_operator)
            
            with cond_col3:
                # 值输入
                self._render_value_input(df, condition, index)
            
            with cond_col4:
                # 启用/禁用和删除按钮
                condition.enabled = st.checkbox(
                    "启用",
                    value=condition.enabled,
                    key=f"condition_enabled_{index}"
                )
                
                if st.button("🗑️", key=f"delete_condition_{index}", help="删除此条件"):
                    st.session_state.filter_conditions.pop(index)
                    st.rerun()
    
    def _render_add_filter_condition(self, df: pd.DataFrame):
        """渲染添加筛选条件界面"""
        add_col1, add_col2, add_col3, add_col4 = st.columns([2, 2, 3, 1])
        
        with add_col1:
            new_column = st.selectbox(
                "选择字段",
                options=df.columns.tolist(),
                key="new_condition_column"
            )
        
        with add_col2:
            column_type = self._get_column_type(df, new_column)
            available_operators = self._get_available_operators(column_type)
            
            new_operator = st.selectbox(
                "选择操作符",
                options=[op.value for op in available_operators],
                key="new_condition_operator"
            )
        
        with add_col3:
            # 根据列类型显示相应的值输入控件
            new_value = self._render_value_input_for_new_condition(df, new_column, new_operator)
        
        with add_col4:
            if st.button("➕ 添加", key="add_new_condition"):
                new_condition = FilterCondition(
                    column=new_column,
                    operator=FilterOperator(new_operator),
                    value=new_value,
                    enabled=True
                )
                st.session_state.filter_conditions.append(new_condition)
                st.rerun()
    
    def _render_value_input(self, df: pd.DataFrame, condition: FilterCondition, index: int):
        """渲染值输入控件"""
        column_type = self._get_column_type(df, condition.column)
        
        if condition.operator == FilterOperator.BETWEEN:
            # Between操作需要两个值
            val_col1, val_col2 = st.columns(2)
            with val_col1:
                condition.value = self._render_single_value_input(
                    df, condition.column, column_type, condition.value, 
                    f"condition_value1_{index}", "最小值"
                )
            with val_col2:
                condition.value2 = self._render_single_value_input(
                    df, condition.column, column_type, condition.value2,
                    f"condition_value2_{index}", "最大值"
                )
        elif condition.operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
            # IN操作需要多个值
            if column_type == FilterType.CATEGORICAL:
                unique_values = df[condition.column].unique().tolist()
                condition.value = st.multiselect(
                    "选择值",
                    options=unique_values,
                    default=condition.value if isinstance(condition.value, list) else [],
                    key=f"condition_multi_value_{index}"
                )
            else:
                # 使用标签输入
                condition.value = st_tags(
                    label="输入值（按回车分隔）",
                    text="输入值并按回车...",
                    value=condition.value if isinstance(condition.value, list) else [],
                    key=f"condition_tags_{index}"
                )
        elif condition.operator in [FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL]:
            # NULL操作不需要值
            st.write("(无需输入值)")
        else:
            # 单值操作
            condition.value = self._render_single_value_input(
                df, condition.column, column_type, condition.value,
                f"condition_single_value_{index}", "值"
            )
    
    def _render_single_value_input(self, df: pd.DataFrame, column: str, column_type: FilterType, 
                                 current_value: Any, key: str, label: str = "值") -> Any:
        """渲染单值输入控件"""
        if column_type == FilterType.NUMERICAL:
            min_val = float(df[column].min())
            max_val = float(df[column].max())
            default_val = float(current_value) if current_value is not None else min_val
            
            return st.number_input(
                label,
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                key=key
            )
        
        elif column_type == FilterType.CATEGORICAL:
            unique_values = df[column].unique().tolist()
            default_idx = unique_values.index(current_value) if current_value in unique_values else 0
            
            return st.selectbox(
                label,
                options=unique_values,
                index=default_idx,
                key=key
            )
        
        elif column_type == FilterType.DATETIME:
            if current_value:
                default_date = pd.to_datetime(current_value).date()
            else:
                default_date = datetime.now().date()
            
            selected_date = st.date_input(
                label,
                value=default_date,
                key=key
            )
            return datetime.combine(selected_date, datetime.min.time())
        
        elif column_type == FilterType.BOOLEAN:
            return st.checkbox(
                label,
                value=bool(current_value) if current_value is not None else False,
                key=key
            )
        
        else:  # TEXT
            return st.text_input(
                label,
                value=str(current_value) if current_value is not None else "",
                key=key
            )
    
    def _render_value_input_for_new_condition(self, df: pd.DataFrame, column: str, operator: str) -> Any:
        """为新条件渲染值输入"""
        column_type = self._get_column_type(df, column)
        
        if operator == "between":
            st.write("将在添加后设置范围值")
            return None
        elif operator in ["in", "not_in"]:
            if column_type == FilterType.CATEGORICAL:
                unique_values = df[column].unique().tolist()
                return st.multiselect(
                    "选择值",
                    options=unique_values,
                    key="new_condition_multi_value"
                )
            else:
                return st_tags(
                    label="输入值（按回车分隔）",
                    text="输入值并按回车...",
                    key="new_condition_tags"
                )
        elif operator in ["is_null", "is_not_null"]:
            return None
        else:
            return self._render_single_value_input(
                df, column, column_type, None,
                "new_condition_value", "值"
            )
    
    def _get_column_type(self, df: pd.DataFrame, column: str) -> FilterType:
        """获取列的数据类型"""
        if column not in df.columns:
            return FilterType.TEXT
        
        dtype = df[column].dtype
        
        if pd.api.types.is_numeric_dtype(dtype):
            return FilterType.NUMERICAL
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return FilterType.DATETIME
        elif pd.api.types.is_bool_dtype(dtype):
            return FilterType.BOOLEAN
        elif df[column].nunique() <= 20:  # 唯一值较少，视为分类
            return FilterType.CATEGORICAL
        else:
            return FilterType.TEXT
    
    def _get_available_operators(self, column_type: FilterType) -> List[FilterOperator]:
        """获取列类型可用的操作符"""
        base_operators = [FilterOperator.EQUALS, FilterOperator.NOT_EQUALS, 
                         FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL]
        
        if column_type == FilterType.NUMERICAL:
            return base_operators + [
                FilterOperator.GREATER_THAN, FilterOperator.GREATER_EQUAL,
                FilterOperator.LESS_THAN, FilterOperator.LESS_EQUAL,
                FilterOperator.BETWEEN, FilterOperator.IN, FilterOperator.NOT_IN
            ]
        elif column_type == FilterType.CATEGORICAL:
            return base_operators + [FilterOperator.IN, FilterOperator.NOT_IN]
        elif column_type == FilterType.TEXT:
            return base_operators + [
                FilterOperator.CONTAINS, FilterOperator.NOT_CONTAINS,
                FilterOperator.STARTS_WITH, FilterOperator.ENDS_WITH,
                FilterOperator.IN, FilterOperator.NOT_IN
            ]
        elif column_type == FilterType.DATETIME:
            return base_operators + [
                FilterOperator.GREATER_THAN, FilterOperator.GREATER_EQUAL,
                FilterOperator.LESS_THAN, FilterOperator.LESS_EQUAL,
                FilterOperator.BETWEEN
            ]
        elif column_type == FilterType.BOOLEAN:
            return base_operators
        
        return base_operators
    
    def _apply_filter_conditions(self, df: pd.DataFrame, conditions: List[FilterCondition]) -> pd.DataFrame:
        """应用筛选条件"""
        if not conditions:
            return df
        
        # 只应用启用的条件
        enabled_conditions = [cond for cond in conditions if cond.enabled]
        
        if not enabled_conditions:
            return df
        
        filtered_df = df.copy()
        condition_results = []
        
        for condition in enabled_conditions:
            try:
                mask = self._apply_single_condition(filtered_df, condition)
                condition_results.append(mask)
            except Exception as e:
                st.error(f"筛选条件错误 - {condition.column} {condition.operator.value}: {str(e)}")
                continue
        
        if not condition_results:
            return filtered_df
        
        # 根据逻辑操作符合并条件
        if st.session_state.filter_logic == "AND":
            final_mask = condition_results[0]
            for mask in condition_results[1:]:
                final_mask = final_mask & mask
        else:  # OR
            final_mask = condition_results[0]
            for mask in condition_results[1:]:
                final_mask = final_mask | mask
        
        return filtered_df[final_mask]
    
    def _apply_single_condition(self, df: pd.DataFrame, condition: FilterCondition) -> pd.Series:
        """应用单个筛选条件"""
        column = condition.column
        operator = condition.operator
        value = condition.value
        value2 = condition.value2
        
        if column not in df.columns:
            raise ValueError(f"列 '{column}' 不存在")
        
        series = df[column]
        
        if operator == FilterOperator.EQUALS:
            return series == value
        elif operator == FilterOperator.NOT_EQUALS:
            return series != value
        elif operator == FilterOperator.GREATER_THAN:
            return series > value
        elif operator == FilterOperator.GREATER_EQUAL:
            return series >= value
        elif operator == FilterOperator.LESS_THAN:
            return series < value
        elif operator == FilterOperator.LESS_EQUAL:
            return series <= value
        elif operator == FilterOperator.BETWEEN:
            return (series >= value) & (series <= value2)
        elif operator == FilterOperator.IN:
            return series.isin(value if isinstance(value, list) else [value])
        elif operator == FilterOperator.NOT_IN:
            return ~series.isin(value if isinstance(value, list) else [value])
        elif operator == FilterOperator.CONTAINS:
            return series.astype(str).str.contains(str(value), na=False, case=False)
        elif operator == FilterOperator.NOT_CONTAINS:
            return ~series.astype(str).str.contains(str(value), na=False, case=False)
        elif operator == FilterOperator.STARTS_WITH:
            return series.astype(str).str.startswith(str(value), na=False)
        elif operator == FilterOperator.ENDS_WITH:
            return series.astype(str).str.endswith(str(value), na=False)
        elif operator == FilterOperator.IS_NULL:
            return series.isna()
        elif operator == FilterOperator.IS_NOT_NULL:
            return series.notna()
        else:
            raise ValueError(f"不支持的操作符: {operator.value}")
    
    def _render_preset_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """渲染预设筛选器"""
        st.write("**预设筛选器**")
        
        # 预设筛选器选择
        preset_names = ["无"] + [preset.name for preset in st.session_state.filter_presets]
        
        current_preset_idx = 0
        if st.session_state.current_preset:
            try:
                current_preset_idx = preset_names.index(st.session_state.current_preset)
            except ValueError:
                pass
        
        selected_preset_name = st.selectbox(
            "选择预设筛选器",
            options=preset_names,
            index=current_preset_idx,
            help="选择一个预设的筛选条件组合"
        )
        
        if selected_preset_name == "无":
            st.session_state.current_preset = None
            return df
        
        # 找到选中的预设
        selected_preset = None
        for preset in st.session_state.filter_presets:
            if preset.name == selected_preset_name:
                selected_preset = preset
                break
        
        if not selected_preset:
            st.error("预设筛选器不存在")
            return df
        
        st.session_state.current_preset = selected_preset_name
        
        # 显示预设详情
        with st.expander(f"预设详情: {selected_preset.name}", expanded=True):
            st.write(f"**描述**: {selected_preset.description}")
            st.write(f"**创建时间**: {selected_preset.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**条件数量**: {len(selected_preset.conditions)}")
            
            # 显示条件列表
            if selected_preset.conditions:
                st.write("**筛选条件**:")
                for i, condition in enumerate(selected_preset.conditions):
                    condition_text = f"{i+1}. {condition.column} {condition.operator.value}"
                    if condition.value is not None:
                        if condition.operator == FilterOperator.BETWEEN:
                            condition_text += f" {condition.value} 和 {condition.value2}"
                        else:
                            condition_text += f" {condition.value}"
                    
                    st.write(condition_text)
        
        # 预设管理按钮
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        
        with preset_col1:
            if st.button("📝 编辑预设"):
                # 将预设条件加载到自定义筛选器
                st.session_state.filter_conditions = selected_preset.conditions.copy()
                st.success(f"预设 '{selected_preset.name}' 已加载到自定义筛选器")
        
        with preset_col2:
            if st.button("🗑️ 删除预设"):
                st.session_state.filter_presets = [
                    p for p in st.session_state.filter_presets if p.name != selected_preset.name
                ]
                st.session_state.current_preset = None
                st.rerun()
        
        with preset_col3:
            if st.button("📋 复制预设"):
                new_name = f"{selected_preset.name}_副本"
                new_preset = FilterPreset(
                    name=new_name,
                    description=f"复制自: {selected_preset.description}",
                    conditions=selected_preset.conditions.copy()
                )
                st.session_state.filter_presets.append(new_preset)
                st.success(f"预设已复制为 '{new_name}'")
        
        # 应用预设筛选
        return self._apply_filter_conditions(df, selected_preset.conditions)
    
    def _show_save_preset_dialog(self):
        """显示保存预设对话框"""
        if not st.session_state.filter_conditions:
            st.warning("没有筛选条件可保存")
            return
        
        with st.form("save_preset_form"):
            preset_name = st.text_input(
                "预设名称",
                placeholder="输入预设名称..."
            )
            
            preset_description = st.text_area(
                "预设描述",
                placeholder="描述这个预设的用途..."
            )
            
            if st.form_submit_button("💾 保存预设"):
                if not preset_name:
                    st.error("请输入预设名称")
                    return
                
                # 检查名称是否重复
                existing_names = [p.name for p in st.session_state.filter_presets]
                if preset_name in existing_names:
                    st.error("预设名称已存在")
                    return
                
                # 创建新预设
                new_preset = FilterPreset(
                    name=preset_name,
                    description=preset_description or "用户自定义预设",
                    conditions=st.session_state.filter_conditions.copy()
                )
                
                st.session_state.filter_presets.append(new_preset)
                st.success(f"预设 '{preset_name}' 保存成功！")
    
    def _render_filter_history(self, df: pd.DataFrame):
        """渲染筛选历史"""
        st.write("**筛选历史**")
        
        if not st.session_state.filter_history:
            st.info("暂无筛选历史")
            return
        
        # 显示历史记录
        for i, history_item in enumerate(reversed(st.session_state.filter_history[-10:])):  # 只显示最近10条
            with st.expander(f"历史 {i+1}: {history_item['timestamp']}", expanded=False):
                st.write(f"**筛选结果**: {history_item['result_count']} 条记录")
                st.write(f"**筛选条件数量**: {len(history_item['conditions'])}")
                st.write(f"**逻辑操作**: {history_item['logic']}")
                
                if st.button(f"🔄 恢复此筛选", key=f"restore_history_{i}"):
                    st.session_state.filter_conditions = [
                        FilterCondition.from_dict(cond) for cond in history_item['conditions']
                    ]
                    st.session_state.filter_logic = history_item['logic']
                    st.success("筛选条件已恢复")
                    st.rerun()
        
        # 清除历史按钮
        if st.button("🗑️ 清除筛选历史"):
            st.session_state.filter_history = []
            st.success("筛选历史已清除")
    
    def _render_advanced_settings(self, df: pd.DataFrame) -> pd.DataFrame:
        """渲染高级设置"""
        st.write("**高级筛选设置**")
        
        # SQL查询界面
        with st.expander("💻 SQL查询界面", expanded=False):
            st.write("**高级用户可以使用SQL语法进行筛选**")
            
            # 显示表结构
            st.write("**表结构信息**:")
            schema_info = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                non_null_count = df[col].count()
                unique_count = df[col].nunique()
                
                schema_info.append({
                    '列名': col,
                    '数据类型': dtype,
                    '非空数量': non_null_count,
                    '唯一值数量': unique_count
                })
            
            schema_df = pd.DataFrame(schema_info)
            st.dataframe(schema_df, hide_index=True)
            
            # SQL查询输入
            sql_query = st.text_area(
                "SQL WHERE子句",
                placeholder="例如: profit_margin > 0.02 AND risk_score < 0.5",
                help="输入WHERE子句内容，不需要包含'WHERE'关键字"
            )
            
            if st.button("执行SQL查询"):
                if sql_query.strip():
                    try:
                        # 使用pandas.query方法执行查询
                        filtered_df = df.query(sql_query)
                        st.success(f"查询成功，返回 {len(filtered_df)} 条记录")
                        return filtered_df
                    except Exception as e:
                        st.error(f"SQL查询错误: {str(e)}")
                        return df
                else:
                    st.warning("请输入SQL查询")
        
        # 正则表达式筛选
        with st.expander("🔤 正则表达式筛选", expanded=False):
            text_columns = [col for col in df.columns if df[col].dtype == 'object']
            
            if not text_columns:
                st.info("没有文本列可用于正则表达式筛选")
            else:
                regex_col1, regex_col2 = st.columns(2)
                
                with regex_col1:
                    regex_column = st.selectbox(
                        "选择文本列",
                        options=text_columns
                    )
                
                with regex_col2:
                    regex_pattern = st.text_input(
                        "正则表达式",
                        placeholder="例如: ^[A-Z].*"
                    )
                
                if st.button("应用正则表达式"):
                    if regex_pattern:
                        try:
                            mask = df[regex_column].astype(str).str.contains(
                                regex_pattern, 
                                na=False, 
                                regex=True,
                                case=False
                            )
                            filtered_df = df[mask]
                            st.success(f"正则表达式筛选成功，返回 {len(filtered_df)} 条记录")
                            return filtered_df
                        except Exception as e:
                            st.error(f"正则表达式错误: {str(e)}")
                    else:
                        st.warning("请输入正则表达式")
        
        # 统计筛选
        with st.expander("📊 统计筛选", expanded=False):
            st.write("基于统计分布进行筛选")
            
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_columns:
                st.info("没有数值列可用于统计筛选")
            else:
                stat_col = st.selectbox(
                    "选择数值列",
                    options=numeric_columns
                )
                
                stat_method = st.selectbox(
                    "统计方法",
                    options=["异常值检测", "分位数筛选", "标准差筛选"]
                )
                
                if stat_method == "异常值检测":
                    # 使用IQR方法检测异常值
                    Q1 = df[stat_col].quantile(0.25)
                    Q3 = df[stat_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_option = st.selectbox(
                        "异常值处理",
                        options=["排除异常值", "只保留异常值"]
                    )
                    
                    if st.button("应用异常值筛选"):
                        if outlier_option == "排除异常值":
                            mask = (df[stat_col] >= lower_bound) & (df[stat_col] <= upper_bound)
                        else:
                            mask = (df[stat_col] < lower_bound) | (df[stat_col] > upper_bound)
                        
                        filtered_df = df[mask]
                        st.success(f"异常值筛选完成，返回 {len(filtered_df)} 条记录")
                        return filtered_df
                
                elif stat_method == "分位数筛选":
                    lower_percentile = st.slider(
                        "下分位数 (%)",
                        min_value=0,
                        max_value=50,
                        value=10
                    )
                    
                    upper_percentile = st.slider(
                        "上分位数 (%)",
                        min_value=50,
                        max_value=100,
                        value=90
                    )
                    
                    if st.button("应用分位数筛选"):
                        lower_value = df[stat_col].quantile(lower_percentile / 100)
                        upper_value = df[stat_col].quantile(upper_percentile / 100)
                        
                        mask = (df[stat_col] >= lower_value) & (df[stat_col] <= upper_value)
                        filtered_df = df[mask]
                        st.success(f"分位数筛选完成，返回 {len(filtered_df)} 条记录")
                        return filtered_df
                
                elif stat_method == "标准差筛选":
                    std_multiplier = st.slider(
                        "标准差倍数",
                        min_value=1.0,
                        max_value=3.0,
                        value=2.0,
                        step=0.1
                    )
                    
                    if st.button("应用标准差筛选"):
                        mean_val = df[stat_col].mean()
                        std_val = df[stat_col].std()
                        lower_bound = mean_val - std_multiplier * std_val
                        upper_bound = mean_val + std_multiplier * std_val
                        
                        mask = (df[stat_col] >= lower_bound) & (df[stat_col] <= upper_bound)
                        filtered_df = df[mask]
                        st.success(f"标准差筛选完成，返回 {len(filtered_df)} 条记录")
                        return filtered_df
        
        return df
    
    def _render_filter_results_summary(self, original_df: pd.DataFrame, filtered_df: pd.DataFrame):
        """渲染筛选结果摘要"""
        st.markdown("---")
        st.subheader("📊 筛选结果摘要")
        
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        
        with summary_col1:
            st.metric(
                "原始记录数",
                f"{len(original_df):,}",
                help="筛选前的总记录数"
            )
        
        with summary_col2:
            st.metric(
                "筛选后记录数",
                f"{len(filtered_df):,}",
                help="筛选后剩余的记录数"
            )
        
        with summary_col3:
            retention_rate = len(filtered_df) / len(original_df) * 100 if len(original_df) > 0 else 0
            st.metric(
                "保留率",
                f"{retention_rate:.1f}%",
                help="筛选后保留的数据比例"
            )
        
        with summary_col4:
            filtered_count = len(original_df) - len(filtered_df)
            st.metric(
                "过滤记录数",
                f"{filtered_count:,}",
                help="被筛选掉的记录数"
            )
        
        # 保存筛选历史
        if len(st.session_state.filter_conditions) > 0:
            history_item = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'conditions': [cond.to_dict() for cond in st.session_state.filter_conditions],
                'logic': st.session_state.filter_logic,
                'original_count': len(original_df),
                'result_count': len(filtered_df),
                'retention_rate': retention_rate
            }
            
            # 避免重复添加相同的筛选历史
            if not st.session_state.filter_history or st.session_state.filter_history[-1]['timestamp'] != history_item['timestamp']:
                st.session_state.filter_history.append(history_item)
                
                # 限制历史记录数量
                if len(st.session_state.filter_history) > 50:
                    st.session_state.filter_history = st.session_state.filter_history[-50:]
    
    def export_filter_config(self) -> str:
        """导出筛选配置为JSON"""
        config = {
            'conditions': [cond.to_dict() for cond in st.session_state.filter_conditions],
            'presets': [preset.to_dict() for preset in st.session_state.filter_presets],
            'logic': st.session_state.filter_logic,
            'exported_at': datetime.now().isoformat()
        }
        return json.dumps(config, indent=2, ensure_ascii=False)
    
    def import_filter_config(self, config_json: str) -> bool:
        """从JSON导入筛选配置"""
        try:
            config = json.loads(config_json)
            
            # 导入筛选条件
            if 'conditions' in config:
                st.session_state.filter_conditions = [
                    FilterCondition.from_dict(cond) for cond in config['conditions']
                ]
            
            # 导入预设
            if 'presets' in config:
                st.session_state.filter_presets = [
                    FilterPreset.from_dict(preset) for preset in config['presets']
                ]
            
            # 导入逻辑操作符
            if 'logic' in config:
                st.session_state.filter_logic = config['logic']
            
            return True
        
        except Exception as e:
            st.error(f"导入配置失败: {str(e)}")
            return False