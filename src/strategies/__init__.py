"""
Strategy module for arbitrage strategy implementations.
Implements various arbitrage strategies with plugin architecture.
"""

# New interface-based imports
from .base import (
    BaseStrategy, StrategyResult, StrategyParameters, OptionData,
    TradingAction, ActionType, RiskMetrics, RiskLevel, StrategyRegistry,
    OptionType
)
from .pricing_arbitrage import PricingArbitrageStrategy, PricingArbitrageParameters
from .legacy_integration import LegacyIntegrationStrategy

# Import Chinese strategy files to ensure registration
try:
    from .看跌看涨平价策略 import PutCallParityStrategy
    from .波动率套利策略 import VolatilityArbitrageStrategy
    _chinese_strategies_available = True
except ImportError as e:
    print(f"Warning: Could not import Chinese strategies: {e}")
    _chinese_strategies_available = False

# Legacy compatibility imports (if available)
try:
    from .base_strategy import IArbitrageStrategy, BaseStrategy as LegacyBaseStrategy
    from .strategy_manager import StrategyManager
    _has_legacy_strategies = True
except ImportError:
    _has_legacy_strategies = False

__all__ = [
    # New interface classes
    'BaseStrategy', 'StrategyResult', 'StrategyParameters', 'OptionData',
    'TradingAction', 'ActionType', 'RiskMetrics', 'RiskLevel', 'OptionType',
    
    # Registry
    'StrategyRegistry',
    
    # Strategy implementations
    'PricingArbitrageStrategy', 'PricingArbitrageParameters',
    'LegacyIntegrationStrategy'
]

# Add Chinese strategies to exports if available
if _chinese_strategies_available:
    __all__.extend(['PutCallParityStrategy', 'VolatilityArbitrageStrategy'])

# Add legacy exports if available
if _has_legacy_strategies:
    __all__.extend([
        'IArbitrageStrategy', 'LegacyBaseStrategy', 'StrategyManager'
    ])