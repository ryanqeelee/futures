"""
Plugin Configuration System for Strategy Management.
插件配置系统，用于策略管理

This module provides comprehensive configuration management for the plugin system,
including strategy-specific parameters, plugin loading preferences, and runtime configuration.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

from .models import StrategyType, StrategyConfig, PluginConfig


@dataclass
class DefaultStrategyConfigs:
    """Default configurations for built-in strategies."""
    
    @staticmethod
    def get_pricing_arbitrage_config() -> StrategyConfig:
        """Get default configuration for pricing arbitrage strategy."""
        return StrategyConfig(
            type=StrategyType.PRICING_ARBITRAGE,
            enabled=True,
            priority=1,
            min_profit_threshold=0.02,  # 2% minimum profit
            max_risk_tolerance=0.15,    # 15% max risk
            parameters={
                'min_price_deviation': 0.05,
                'max_price_deviation': 0.5,
                'require_theoretical_price': True,
                'min_implied_volatility': 0.01,
                'max_implied_volatility': 2.0,
            }
        )
    
    @staticmethod
    def get_volatility_arbitrage_config() -> StrategyConfig:
        """Get default configuration for volatility arbitrage strategy."""
        return StrategyConfig(
            type=StrategyType.VOLATILITY_ARBITRAGE,
            enabled=True,
            priority=2,
            min_profit_threshold=0.03,  # 3% minimum profit
            max_risk_tolerance=0.25,    # 25% max risk (higher for vol strategies)
            parameters={
                'min_iv_spread': 0.05,
                'max_iv_level': 1.0,
                'min_iv_level': 0.02,
                'use_historical_volatility': True,
                'historical_volatility_window': 30,
                'min_hv_iv_spread': 0.03,
                'enable_calendar_spreads': True,
                'min_calendar_days': 7,
                'max_calendar_days': 90,
                'enable_surface_analysis': True,
                'min_surface_points': 5,
                'skew_threshold': 0.02,
                'volatility_mean_reversion_period': 20,
                'confidence_threshold': 0.7
            }
        )
    
    @staticmethod
    def get_put_call_parity_config() -> StrategyConfig:
        """Get default configuration for put-call parity strategy."""
        return StrategyConfig(
            type=StrategyType.PUT_CALL_PARITY,
            enabled=True,
            priority=3,
            min_profit_threshold=0.01,  # 1% minimum profit (arbitrage should be lower threshold)
            max_risk_tolerance=0.1,     # 10% max risk (should be very low for arbitrage)
            parameters={
                'min_parity_deviation': 0.02,
                'max_parity_deviation': 0.5,
                'risk_free_rate': 0.03,
                'dividend_yield': 0.0,
                'exact_strike_match': True,
                'max_strike_difference': 0.01,
                'exact_expiry_match': True,
                'max_expiry_difference': 1,
                'include_transaction_costs': True,
                'transaction_cost_rate': 0.005,
                'max_position_size': 10,
                'prefer_liquid_options': True,
                'avoid_deep_itm_puts': True,
                'max_itm_amount': 0.1
            }
        )


@dataclass 
class PluginLoadingConfig:
    """Configuration for plugin loading behavior."""
    
    enabled_strategies: List[StrategyType] = field(default_factory=lambda: [
        StrategyType.PRICING_ARBITRAGE,
        StrategyType.VOLATILITY_ARBITRAGE,
        StrategyType.PUT_CALL_PARITY
    ])
    
    strategy_priorities: Dict[StrategyType, int] = field(default_factory=lambda: {
        StrategyType.PUT_CALL_PARITY: 1,      # Highest priority (risk-free arbitrage)
        StrategyType.PRICING_ARBITRAGE: 2,    # Medium priority  
        StrategyType.VOLATILITY_ARBITRAGE: 3  # Lower priority (more complex)
    })
    
    # Plugin file mapping
    strategy_file_mapping: Dict[StrategyType, str] = field(default_factory=lambda: {
        StrategyType.PRICING_ARBITRAGE: "pricing_arbitrage.py",
        StrategyType.VOLATILITY_ARBITRAGE: "波动率套利策略.py",
        StrategyType.PUT_CALL_PARITY: "看跌看涨平价策略.py"
    })
    
    # Loading preferences
    load_order: List[StrategyType] = field(default_factory=lambda: [
        StrategyType.PUT_CALL_PARITY,      # Load arbitrage strategies first
        StrategyType.PRICING_ARBITRAGE,
        StrategyType.VOLATILITY_ARBITRAGE
    ])
    
    # Error handling
    continue_on_load_error: bool = True
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    
    # Development settings
    auto_reload_in_dev: bool = True
    validate_plugins_on_load: bool = True
    enable_plugin_metrics: bool = True


class PluginConfigurationManager:
    """
    Comprehensive plugin configuration manager.
    
    Handles loading, validation, and management of plugin configurations
    with support for different environments and runtime updates.
    """
    
    def __init__(self, base_config_dir: Optional[Path] = None):
        """
        Initialize the plugin configuration manager.
        
        Args:
            base_config_dir: Base directory for configuration files
        """
        self.base_config_dir = base_config_dir or Path("config")
        self.runtime_overrides: Dict[str, Any] = {}  # Initialize first
        self.plugin_config = self._load_plugin_config()
        self.loading_config = PluginLoadingConfig()
        self.strategy_configs = self._initialize_strategy_configs()
        self.last_updated = datetime.now()
    
    def _load_plugin_config(self) -> PluginConfig:
        """Load plugin configuration from file or use defaults."""
        config_file = self.base_config_dir / "plugin_config.yaml"
        
        if config_file.exists():
            # In a real implementation, you would load from YAML/JSON
            # For now, return default configuration
            pass
        
        return PluginConfig()
    
    def _initialize_strategy_configs(self) -> Dict[StrategyType, StrategyConfig]:
        """Initialize strategy configurations with defaults."""
        configs = {}
        
        # Load default configurations
        configs[StrategyType.PRICING_ARBITRAGE] = DefaultStrategyConfigs.get_pricing_arbitrage_config()
        configs[StrategyType.VOLATILITY_ARBITRAGE] = DefaultStrategyConfigs.get_volatility_arbitrage_config()
        configs[StrategyType.PUT_CALL_PARITY] = DefaultStrategyConfigs.get_put_call_parity_config()
        
        # Apply any custom overrides from config files
        self._apply_config_overrides(configs)
        
        return configs
    
    def _apply_config_overrides(self, configs: Dict[StrategyType, StrategyConfig]) -> None:
        """Apply configuration overrides from files or environment."""
        # In a real implementation, this would load overrides from:
        # - Environment variables
        # - Configuration files (YAML/JSON)
        # - Database settings
        # - Runtime API calls
        
        # Apply runtime overrides
        for strategy_type, overrides in self.runtime_overrides.items():
            if hasattr(StrategyType, strategy_type) and strategy_type in configs:
                config = configs[strategy_type]
                for key, value in overrides.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                    elif key in config.parameters:
                        config.parameters[key] = value
    
    def get_strategy_config(self, strategy_type: StrategyType) -> Optional[StrategyConfig]:
        """
        Get configuration for a specific strategy.
        
        Args:
            strategy_type: Type of strategy to get config for
            
        Returns:
            Strategy configuration or None if not found
        """
        return self.strategy_configs.get(strategy_type)
    
    def update_strategy_config(
        self, 
        strategy_type: StrategyType, 
        config_updates: Dict[str, Any]
    ) -> bool:
        """
        Update strategy configuration at runtime.
        
        Args:
            strategy_type: Strategy to update
            config_updates: Configuration updates to apply
            
        Returns:
            True if update was successful
        """
        try:
            if strategy_type not in self.strategy_configs:
                return False
            
            config = self.strategy_configs[strategy_type]
            
            # Update main config fields
            for key, value in config_updates.items():
                if key == 'parameters':
                    # Update parameters dict
                    if isinstance(value, dict):
                        config.parameters.update(value)
                elif hasattr(config, key):
                    setattr(config, key, value)
            
            self.last_updated = datetime.now()
            return True
            
        except Exception as e:
            print(f"Error updating strategy config: {e}")
            return False
    
    def get_plugin_loading_config(self) -> PluginLoadingConfig:
        """Get plugin loading configuration."""
        return self.loading_config
    
    def get_enabled_strategies(self) -> List[StrategyType]:
        """Get list of enabled strategies in priority order."""
        enabled = []
        
        # Get strategies in priority order
        for strategy_type in self.loading_config.load_order:
            if (strategy_type in self.loading_config.enabled_strategies and 
                strategy_type in self.strategy_configs and
                self.strategy_configs[strategy_type].enabled):
                enabled.append(strategy_type)
        
        # Add any enabled strategies not in load_order
        for strategy_type in self.loading_config.enabled_strategies:
            if (strategy_type not in enabled and 
                strategy_type in self.strategy_configs and
                self.strategy_configs[strategy_type].enabled):
                enabled.append(strategy_type)
        
        return enabled
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """
        Validate all plugin configurations.
        
        Returns:
            Dictionary mapping configuration areas to validation errors
        """
        errors = {
            'plugin_config': [],
            'strategy_configs': [],
            'loading_config': []
        }
        
        # Validate plugin config
        if not self.plugin_config.plugin_directories:
            errors['plugin_config'].append("No plugin directories specified")
        
        for directory in self.plugin_config.plugin_directories:
            plugin_path = Path(directory)
            if not plugin_path.exists():
                errors['plugin_config'].append(f"Plugin directory does not exist: {directory}")
        
        # Validate strategy configs
        for strategy_type, config in self.strategy_configs.items():
            if config.min_profit_threshold < 0:
                errors['strategy_configs'].append(f"{strategy_type}: Negative profit threshold")
            
            if config.max_risk_tolerance < 0 or config.max_risk_tolerance > 1:
                errors['strategy_configs'].append(f"{strategy_type}: Invalid risk tolerance")
        
        # Validate loading config
        for strategy_type in self.loading_config.enabled_strategies:
            if strategy_type not in self.strategy_configs:
                errors['loading_config'].append(f"Enabled strategy {strategy_type} has no configuration")
        
        return errors
    
    def get_plugin_directories(self) -> List[Path]:
        """Get list of plugin directories as Path objects."""
        return [Path(directory) for directory in self.plugin_config.plugin_directories]
    
    def export_configuration(self, format: str = "dict") -> Union[Dict[str, Any], str]:
        """
        Export current configuration.
        
        Args:
            format: Export format ('dict', 'json', 'yaml')
            
        Returns:
            Configuration in requested format
        """
        config_dict = {
            'plugin_config': {
                'enabled': self.plugin_config.enabled,
                'plugin_directories': self.plugin_config.plugin_directories,
                'auto_reload': self.plugin_config.auto_reload,
                'enable_hot_reload': self.plugin_config.enable_hot_reload,
                'max_load_retries': self.plugin_config.max_load_retries,
                'validate_on_load': self.plugin_config.validate_on_load,
                'parallel_loading': self.plugin_config.parallel_loading
            },
            'loading_config': {
                'enabled_strategies': [s.value for s in self.loading_config.enabled_strategies],
                'strategy_priorities': {s.value: p for s, p in self.loading_config.strategy_priorities.items()},
                'load_order': [s.value for s in self.loading_config.load_order]
            },
            'strategy_configs': {}
        }
        
        # Export strategy configs
        for strategy_type, config in self.strategy_configs.items():
            config_dict['strategy_configs'][strategy_type.value] = {
                'enabled': config.enabled,
                'priority': config.priority,
                'min_profit_threshold': config.min_profit_threshold,
                'max_risk_tolerance': config.max_risk_tolerance,
                'parameters': config.parameters
            }
        
        if format == "dict":
            return config_dict
        elif format == "json":
            import json
            return json.dumps(config_dict, indent=2, default=str)
        elif format == "yaml":
            try:
                import yaml
                return yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
            except ImportError:
                return str(config_dict)  # Fallback to string representation
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_runtime_metrics(self) -> Dict[str, Any]:
        """Get runtime metrics for plugin configuration."""
        return {
            'last_updated': self.last_updated.isoformat(),
            'total_strategies': len(self.strategy_configs),
            'enabled_strategies': len(self.get_enabled_strategies()),
            'plugin_directories': len(self.plugin_config.plugin_directories),
            'runtime_overrides': len(self.runtime_overrides),
            'configuration_errors': sum(len(errors) for errors in self.validate_configuration().values())
        }
    
    def reload_configuration(self) -> bool:
        """Reload configuration from files."""
        try:
            self.plugin_config = self._load_plugin_config()
            self.strategy_configs = self._initialize_strategy_configs()
            self.last_updated = datetime.now()
            return True
        except Exception as e:
            print(f"Error reloading configuration: {e}")
            return False


# Global configuration manager instance
_config_manager: Optional[PluginConfigurationManager] = None


def get_plugin_config_manager() -> PluginConfigurationManager:
    """Get or create the global plugin configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = PluginConfigurationManager()
    return _config_manager


def initialize_plugin_config(base_config_dir: Optional[Path] = None) -> PluginConfigurationManager:
    """
    Initialize the plugin configuration system.
    
    Args:
        base_config_dir: Base directory for configuration files
        
    Returns:
        Initialized plugin configuration manager
    """
    global _config_manager
    _config_manager = PluginConfigurationManager(base_config_dir)
    return _config_manager