"""
Configuration package for the options arbitrage system.
Provides type-safe configuration management with Pydantic models.
"""

from .models import (
    SystemConfig, DataSourceConfig, StrategyConfig, RiskConfig,
    CacheConfig, MonitoringConfig, ArbitrageOpportunity,
    DataSourceType, StrategyType, LogLevel, ValidationError
)
from .manager import ConfigManager, get_config_manager, get_config, ConfigurationError

# Backward compatibility - can be removed later
try:
    from .settings import Settings
    from .strategy_config import StrategyConfig as LegacyStrategyConfig
    from .risk_config import RiskConfig as LegacyRiskConfig
    _has_legacy = True
except ImportError:
    _has_legacy = False

__all__ = [
    # New Pydantic-based models
    'SystemConfig', 'DataSourceConfig', 'StrategyConfig', 'RiskConfig',
    'CacheConfig', 'MonitoringConfig', 'ArbitrageOpportunity', 'ValidationError',
    
    # Enums
    'DataSourceType', 'StrategyType', 'LogLevel',
    
    # Manager
    'ConfigManager', 'get_config_manager', 'get_config', 'ConfigurationError'
]

# Add legacy exports if available
if _has_legacy:
    __all__.extend(['Settings', 'LegacyStrategyConfig', 'LegacyRiskConfig'])