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

# Add legacy exports if available
if _has_legacy_strategies:
    __all__.extend([
        'IArbitrageStrategy', 'LegacyBaseStrategy', 'StrategyManager'
    ])