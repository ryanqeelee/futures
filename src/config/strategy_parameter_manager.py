"""
Strategy Parameter Management System

提供策略参数的定义、验证、预设配置和动态管理功能
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
from pathlib import Path

from .models import (
    StrategyType, StrategyConfig, ParameterDefinition, 
    ParameterConstraint, StrategyParameterSet
)


class StrategyParameterManager:
    """策略参数管理器"""
    
    def __init__(self, config_dir: Path = None):
        self.config_dir = config_dir or Path("config/strategy_presets")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 预定义的策略参数集
        self._parameter_sets = self._initialize_parameter_definitions()
        self._preset_configs = self._load_preset_configurations()
    
    def _initialize_parameter_definitions(self) -> Dict[StrategyType, StrategyParameterSet]:
        """初始化策略参数定义"""
        parameter_sets = {}
        
        # 定价套利策略参数
        parameter_sets[StrategyType.PRICING_ARBITRAGE] = StrategyParameterSet(
            strategy_type=StrategyType.PRICING_ARBITRAGE,
            parameter_definitions=[
                ParameterDefinition(
                    name="price_deviation_threshold",
                    display_name="价格偏差阈值 (%)",
                    description="理论价格与市场价格的最小偏差百分比",
                    parameter_type="float",
                    default_value=2.0,
                    constraint=ParameterConstraint(min_value=0.5, max_value=10.0, step=0.1),
                    category="pricing"
                ),
                ParameterDefinition(
                    name="min_volume_threshold",
                    display_name="最小成交量",
                    description="期权合约的最小成交量要求",
                    parameter_type="int",
                    default_value=50,
                    constraint=ParameterConstraint(min_value=1, max_value=1000, step=1),
                    category="liquidity"
                ),
                ParameterDefinition(
                    name="volatility_range",
                    display_name="波动率范围",
                    description="隐含波动率的合理范围 (min, max)",
                    parameter_type="str",
                    default_value="0.1,0.8",
                    constraint=ParameterConstraint(required=True),
                    category="pricing",
                    advanced=True
                )
            ]
        )
        
        # 看跌看涨平价策略参数  
        parameter_sets[StrategyType.PUT_CALL_PARITY] = StrategyParameterSet(
            strategy_type=StrategyType.PUT_CALL_PARITY,
            parameter_definitions=[
                ParameterDefinition(
                    name="parity_tolerance",
                    display_name="平价偏差容忍度 (%)",
                    description="看跌看涨平价公式的偏差容忍度",
                    parameter_type="float",
                    default_value=1.5,
                    constraint=ParameterConstraint(min_value=0.1, max_value=5.0, step=0.1),
                    category="arbitrage"
                ),
                ParameterDefinition(
                    name="min_strike_gap",
                    display_name="最小行权价间隔",
                    description="同一到期日期权的最小行权价间隔",
                    parameter_type="float",
                    default_value=0.05,
                    constraint=ParameterConstraint(min_value=0.01, max_value=0.5, step=0.01),
                    category="filtering"
                ),
                ParameterDefinition(
                    name="interest_rate",
                    display_name="无风险利率 (%)",
                    description="用于平价计算的无风险利率",
                    parameter_type="float",
                    default_value=2.5,
                    constraint=ParameterConstraint(min_value=0.0, max_value=10.0, step=0.1),
                    category="pricing",
                    advanced=True
                )
            ]
        )
        
        # 波动率套利策略参数
        parameter_sets[StrategyType.VOLATILITY_ARBITRAGE] = StrategyParameterSet(
            strategy_type=StrategyType.VOLATILITY_ARBITRAGE,
            parameter_definitions=[
                ParameterDefinition(
                    name="volatility_spread_threshold",
                    display_name="波动率价差阈值 (%)",
                    description="隐含波动率与历史波动率的最小价差",
                    parameter_type="float",
                    default_value=3.0,
                    constraint=ParameterConstraint(min_value=1.0, max_value=10.0, step=0.1),
                    category="volatility"
                ),
                ParameterDefinition(
                    name="lookback_period",
                    display_name="历史波动率回看期 (天)",
                    description="计算历史波动率的时间窗口",
                    parameter_type="int",
                    default_value=20,
                    constraint=ParameterConstraint(min_value=5, max_value=252, step=1),
                    category="volatility"
                ),
                ParameterDefinition(
                    name="delta_neutral",
                    display_name="Delta中性调整",
                    description="是否启用Delta中性头寸调整",
                    parameter_type="bool",
                    default_value=True,
                    category="risk_management"
                )
            ]
        )
        
        # 日历价差策略参数
        parameter_sets[StrategyType.CALENDAR_SPREAD] = StrategyParameterSet(
            strategy_type=StrategyType.CALENDAR_SPREAD,
            parameter_definitions=[
                ParameterDefinition(
                    name="time_decay_threshold",
                    display_name="时间价值衰减阈值",
                    description="预期的最小时间价值衰减收益",
                    parameter_type="float",
                    default_value=0.02,
                    constraint=ParameterConstraint(min_value=0.005, max_value=0.1, step=0.005),
                    category="time_decay"
                ),
                ParameterDefinition(
                    name="maturity_spread_range",
                    display_name="到期日差异范围 (天)",
                    description="近月和远月合约的到期日差异范围",
                    parameter_type="str",
                    default_value="30,90",
                    constraint=ParameterConstraint(required=True),
                    category="maturity"
                ),
                ParameterDefinition(
                    name="moneyness_range",
                    display_name="价性范围",
                    description="期权价性的可接受范围 (ITM/ATM/OTM)",
                    parameter_type="choice",
                    default_value="ATM",
                    constraint=ParameterConstraint(choices=["ITM", "ATM", "OTM", "ALL"]),
                    category="filtering"
                )
            ]
        )
        
        # 价格预测交易策略参数
        parameter_sets[StrategyType.PREDICTION_BASED] = StrategyParameterSet(
            strategy_type=StrategyType.PREDICTION_BASED,
            parameter_definitions=[
                ParameterDefinition(
                    name="prediction_confidence_threshold",
                    display_name="预测置信度阈值 (%)",
                    description="AI预测模型的最小置信度要求",
                    parameter_type="float",
                    default_value=70.0,
                    constraint=ParameterConstraint(min_value=50.0, max_value=95.0, step=1.0),
                    category="prediction"
                ),
                ParameterDefinition(
                    name="prediction_horizon",
                    display_name="预测时间跨度 (天)",
                    description="AI模型的预测时间范围",
                    parameter_type="int",
                    default_value=5,
                    constraint=ParameterConstraint(min_value=1, max_value=30, step=1),
                    category="prediction"
                ),
                ParameterDefinition(
                    name="model_type",
                    display_name="预测模型类型",
                    description="使用的机器学习模型类型",
                    parameter_type="choice",
                    default_value="lstm",
                    constraint=ParameterConstraint(choices=["lstm", "transformer", "ensemble"]),
                    category="model",
                    advanced=True
                ),
                ParameterDefinition(
                    name="feature_importance_threshold",
                    display_name="特征重要性阈值",
                    description="特征选择的重要性阈值",
                    parameter_type="float",
                    default_value=0.05,
                    constraint=ParameterConstraint(min_value=0.01, max_value=0.2, step=0.01),
                    category="model",
                    advanced=True
                )
            ]
        )
        
        return parameter_sets
    
    def _load_preset_configurations(self) -> Dict[StrategyType, Dict[str, Dict[str, Any]]]:
        """加载预设配置"""
        preset_configs = {}
        
        # 为每种策略定义预设配置
        preset_configs[StrategyType.PRICING_ARBITRAGE] = {
            "保守型": {
                "price_deviation_threshold": 3.0,
                "min_volume_threshold": 100,
                "volatility_range": "0.15,0.6"
            },
            "平衡型": {
                "price_deviation_threshold": 2.0,
                "min_volume_threshold": 50,
                "volatility_range": "0.1,0.8"
            },
            "激进型": {
                "price_deviation_threshold": 1.0,
                "min_volume_threshold": 20,
                "volatility_range": "0.05,1.0"
            }
        }
        
        preset_configs[StrategyType.PUT_CALL_PARITY] = {
            "保守型": {
                "parity_tolerance": 2.0,
                "min_strike_gap": 0.1,
                "interest_rate": 2.5
            },
            "平衡型": {
                "parity_tolerance": 1.5,
                "min_strike_gap": 0.05,
                "interest_rate": 2.5
            },
            "激进型": {
                "parity_tolerance": 1.0,
                "min_strike_gap": 0.02,
                "interest_rate": 2.5
            }
        }
        
        preset_configs[StrategyType.VOLATILITY_ARBITRAGE] = {
            "保守型": {
                "volatility_spread_threshold": 5.0,
                "lookback_period": 30,
                "delta_neutral": True
            },
            "平衡型": {
                "volatility_spread_threshold": 3.0,
                "lookback_period": 20,
                "delta_neutral": True
            },
            "激进型": {
                "volatility_spread_threshold": 2.0,
                "lookback_period": 10,
                "delta_neutral": False
            }
        }
        
        preset_configs[StrategyType.CALENDAR_SPREAD] = {
            "保守型": {
                "time_decay_threshold": 0.03,
                "maturity_spread_range": "60,120",
                "moneyness_range": "ATM"
            },
            "平衡型": {
                "time_decay_threshold": 0.02,
                "maturity_spread_range": "30,90",
                "moneyness_range": "ATM"
            },
            "激进型": {
                "time_decay_threshold": 0.01,
                "maturity_spread_range": "15,60",
                "moneyness_range": "ALL"
            }
        }
        
        preset_configs[StrategyType.PREDICTION_BASED] = {
            "保守型": {
                "prediction_confidence_threshold": 80.0,
                "prediction_horizon": 10,
                "model_type": "ensemble",
                "feature_importance_threshold": 0.1
            },
            "平衡型": {
                "prediction_confidence_threshold": 70.0,
                "prediction_horizon": 5,
                "model_type": "lstm",
                "feature_importance_threshold": 0.05
            },
            "激进型": {
                "prediction_confidence_threshold": 60.0,
                "prediction_horizon": 3,
                "model_type": "transformer",
                "feature_importance_threshold": 0.02
            }
        }
        
        return preset_configs
    
    def get_strategy_parameter_definitions(self, strategy_type: StrategyType) -> List[ParameterDefinition]:
        """获取策略的参数定义"""
        if strategy_type in self._parameter_sets:
            return self._parameter_sets[strategy_type].parameter_definitions
        return []
    
    def get_preset_configurations(self, strategy_type: StrategyType) -> Dict[str, Dict[str, Any]]:
        """获取策略的预设配置"""
        return self._preset_configs.get(strategy_type, {})
    
    def create_strategy_config(
        self, 
        strategy_type: StrategyType, 
        preset_name: Optional[str] = None,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> StrategyConfig:
        """创建策略配置"""
        config = StrategyConfig(
            type=strategy_type,
            enabled=True,
            priority=1
        )
        
        # 使用预设配置
        if preset_name and strategy_type in self._preset_configs:
            presets = self._preset_configs[strategy_type]
            if preset_name in presets:
                config.parameters = presets[preset_name].copy()
                config.preset_name = preset_name
        
        # 覆盖自定义参数
        if custom_params:
            config.parameters.update(custom_params)
            config.custom_config = True
        
        config.last_modified = datetime.now()
        return config
    
    def validate_parameters(self, strategy_type: StrategyType, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证策略参数"""
        errors = []
        
        if strategy_type not in self._parameter_sets:
            errors.append(f"未知的策略类型: {strategy_type}")
            return False, errors
        
        param_defs = {p.name: p for p in self._parameter_sets[strategy_type].parameter_definitions}
        
        for param_name, value in parameters.items():
            if param_name not in param_defs:
                errors.append(f"未知参数: {param_name}")
                continue
            
            param_def = param_defs[param_name]
            constraint = param_def.constraint
            
            if constraint:
                # 检查数值范围
                if param_def.parameter_type in ["float", "int"]:
                    if constraint.min_value is not None and value < constraint.min_value:
                        errors.append(f"{param_def.display_name}: 值 {value} 小于最小值 {constraint.min_value}")
                    if constraint.max_value is not None and value > constraint.max_value:
                        errors.append(f"{param_def.display_name}: 值 {value} 大于最大值 {constraint.max_value}")
                
                # 检查选择列表
                if constraint.choices and value not in constraint.choices:
                    errors.append(f"{param_def.display_name}: 值 {value} 不在允许的选择列表中")
        
        # 检查必需参数
        for param_def in param_defs.values():
            if param_def.constraint and param_def.constraint.required:
                if param_def.name not in parameters:
                    errors.append(f"缺少必需参数: {param_def.display_name}")
        
        return len(errors) == 0, errors
    
    def save_custom_preset(self, strategy_type: StrategyType, preset_name: str, parameters: Dict[str, Any]) -> bool:
        """保存自定义预设配置"""
        try:
            preset_file = self.config_dir / f"{strategy_type.value}_presets.json"
            
            # 加载现有预设
            presets = {}
            if preset_file.exists():
                with open(preset_file, 'r', encoding='utf-8') as f:
                    presets = json.load(f)
            
            # 添加新预设
            presets[preset_name] = parameters
            
            # 保存到文件
            with open(preset_file, 'w', encoding='utf-8') as f:
                json.dump(presets, f, indent=2, ensure_ascii=False)
            
            # 更新内存中的预设配置
            if strategy_type not in self._preset_configs:
                self._preset_configs[strategy_type] = {}
            self._preset_configs[strategy_type][preset_name] = parameters
            
            return True
        except Exception:
            return False
    
    def get_parameter_categories(self, strategy_type: StrategyType) -> List[str]:
        """获取策略参数的分类"""
        if strategy_type not in self._parameter_sets:
            return []
        
        categories = set()
        for param_def in self._parameter_sets[strategy_type].parameter_definitions:
            categories.add(param_def.category)
        
        return sorted(list(categories))