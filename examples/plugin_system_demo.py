#!/usr/bin/env python3
"""
Plugin System Demo Script
插件系统演示脚本

This script demonstrates the complete plugin loading system functionality including:
- Plugin discovery and loading
- Strategy configuration management
- Hot-reload capabilities
- Error handling and recovery
- Performance monitoring
- Health checks

Usage:
    python examples/plugin_system_demo.py
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.plugin_manager import PluginManager, PluginManagerConfig
from config.plugin_config import PluginConfigurationManager, initialize_plugin_config
from config.models import StrategyType
from strategies.base import OptionData, OptionType


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DemoOptionData:
    """Demo option data for testing strategies."""
    
    def __init__(self, code: str, underlying: str, option_type: str, strike: float, 
                 market_price: float, iv: float = 0.2):
        self.code = code
        self.name = f"{underlying} {option_type}{strike}"
        self.underlying = underlying
        self.option_type = OptionType.CALL if option_type == 'C' else OptionType.PUT
        self.strike_price = strike
        self.market_price = market_price
        self.bid_price = market_price * 0.95
        self.ask_price = market_price * 1.05
        self.volume = 1000
        self.open_interest = 5000
        self.implied_volatility = iv
        self.theoretical_price = market_price * 1.02
        self.expiry_date = datetime.now().replace(month=12, day=31)  # End of year
        self.delta = 0.5
        self.gamma = 0.02
        self.theta = -0.01
        self.vega = 0.1
    
    @property
    def days_to_expiry(self) -> int:
        return (self.expiry_date - datetime.now()).days
    
    @property
    def time_to_expiry(self) -> float:
        return self.days_to_expiry / 365.0
    
    @property
    def mid_price(self) -> float:
        return (self.bid_price + self.ask_price) / 2
    
    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price
    
    @property
    def spread_pct(self) -> float:
        return self.spread / self.mid_price if self.mid_price > 0 else 0


def create_sample_options_data() -> List[DemoOptionData]:
    """Create sample options data for demonstration."""
    options = []
    
    # Create options for different underlyings
    underlyings = ["AAPL", "TSLA", "SPY"]
    
    for underlying in underlyings:
        base_price = {"AAPL": 150, "TSLA": 200, "SPY": 400}[underlying]
        
        # Create options with different strikes and types
        for strike_offset in [-20, -10, 0, 10, 20]:
            strike = base_price + strike_offset
            
            # Call options
            call_price = max(base_price - strike, 0) + 5  # Intrinsic + time value
            options.append(DemoOptionData(
                code=f"{underlying}{strike}C",
                underlying=underlying,
                option_type="C",
                strike=strike,
                market_price=call_price,
                iv=0.2 + strike_offset * 0.001  # Volatility skew
            ))
            
            # Put options
            put_price = max(strike - base_price, 0) + 3  # Intrinsic + time value
            options.append(DemoOptionData(
                code=f"{underlying}{strike}P", 
                underlying=underlying,
                option_type="P",
                strike=strike,
                market_price=put_price,
                iv=0.25 + abs(strike_offset) * 0.001  # Volatility smile
            ))
    
    return options


async def demonstrate_plugin_discovery():
    """Demonstrate plugin discovery and loading."""
    logger.info("=== Plugin Discovery and Loading Demo ===")
    
    # Initialize configuration manager
    config_manager = initialize_plugin_config()
    
    # Setup plugin manager
    plugin_config = PluginManagerConfig(
        plugin_directories=['src/strategies'],
        auto_reload=True,
        enable_hot_reload=True,
        validate_on_load=True,
        parallel_loading=True
    )
    
    plugin_manager = PluginManager(plugin_config)
    
    try:
        # Initialize plugin system
        logger.info("Initializing plugin system...")
        await plugin_manager.initialize()
        
        # Show discovered plugins
        plugins = plugin_manager.list_plugins()
        logger.info(f"Discovered {len(plugins)} plugins:")
        
        for plugin_name, plugin_info in plugins.items():
            logger.info(f"  - {plugin_name}: {plugin_info.strategy_type.value}")
            logger.info(f"    File: {plugin_info.file_path}")
            logger.info(f"    Priority: {plugin_info.priority}")
            logger.info(f"    Enabled: {plugin_info.is_enabled}")
        
        # Show loading statistics
        stats = plugin_manager.get_load_statistics()
        logger.info(f"\nLoading Statistics:")
        logger.info(f"  Total loads: {stats['total_loads']}")
        logger.info(f"  Successful: {stats['successful_loads']}")
        logger.info(f"  Failed: {stats['failed_loads']}")
        
        return plugin_manager, config_manager
        
    except Exception as e:
        logger.error(f"Error during plugin discovery: {e}")
        await plugin_manager.shutdown()
        raise


async def demonstrate_strategy_execution(plugin_manager: PluginManager):
    """Demonstrate strategy execution."""
    logger.info("\n=== Strategy Execution Demo ===")
    
    # Get loaded strategies
    strategies = plugin_manager.get_strategies()
    logger.info(f"Loaded {len(strategies)} strategies")
    
    # Create sample data
    options_data = create_sample_options_data()
    logger.info(f"Created {len(options_data)} sample options")
    
    # Execute each strategy
    for strategy_name, strategy in strategies.items():
        logger.info(f"\nExecuting strategy: {strategy_name}")
        logger.info(f"Strategy type: {strategy.strategy_type.value}")
        
        try:
            # Execute strategy
            start_time = datetime.now()
            result = strategy.scan_opportunities(options_data)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Display results
            logger.info(f"Execution time: {execution_time:.3f}s")
            logger.info(f"Success: {result.success}")
            logger.info(f"Opportunities found: {len(result.opportunities)}")
            
            # Show first few opportunities
            for i, opportunity in enumerate(result.opportunities[:3]):
                logger.info(f"  Opportunity {i+1}:")
                logger.info(f"    Expected profit: ${opportunity.expected_profit:.2f}")
                logger.info(f"    Profit margin: {opportunity.profit_margin*100:.2f}%")
                logger.info(f"    Risk score: {opportunity.risk_score:.3f}")
                logger.info(f"    Confidence: {opportunity.confidence_score:.3f}")
                logger.info(f"    Instruments: {', '.join(opportunity.instruments)}")
            
            if len(result.opportunities) > 3:
                logger.info(f"    ... and {len(result.opportunities) - 3} more")
            
        except Exception as e:
            logger.error(f"Error executing strategy {strategy_name}: {e}")


async def demonstrate_configuration_management(config_manager: PluginConfigurationManager):
    """Demonstrate configuration management."""
    logger.info("\n=== Configuration Management Demo ===")
    
    # Show current configurations
    enabled_strategies = config_manager.get_enabled_strategies()
    logger.info(f"Enabled strategies: {[s.value for s in enabled_strategies]}")
    
    # Show strategy-specific configurations
    for strategy_type in enabled_strategies:
        config = config_manager.get_strategy_config(strategy_type)
        if config:
            logger.info(f"\n{strategy_type.value} configuration:")
            logger.info(f"  Enabled: {config.enabled}")
            logger.info(f"  Priority: {config.priority}")
            logger.info(f"  Min profit threshold: {config.min_profit_threshold}")
            logger.info(f"  Max risk tolerance: {config.max_risk_tolerance}")
            logger.info(f"  Parameters: {len(config.parameters)} custom parameters")
    
    # Demonstrate configuration update
    logger.info("\nUpdating pricing arbitrage configuration...")
    success = config_manager.update_strategy_config(
        StrategyType.PRICING_ARBITRAGE,
        {
            'min_profit_threshold': 0.03,
            'parameters': {'min_price_deviation': 0.06}
        }
    )
    logger.info(f"Configuration update success: {success}")
    
    # Validate configuration
    errors = config_manager.validate_configuration()
    error_count = sum(len(error_list) for error_list in errors.values())
    logger.info(f"Configuration validation errors: {error_count}")
    
    if error_count > 0:
        for section, error_list in errors.items():
            if error_list:
                logger.warning(f"  {section}: {', '.join(error_list)}")


async def demonstrate_health_monitoring(plugin_manager: PluginManager):
    """Demonstrate health monitoring and diagnostics."""
    logger.info("\n=== Health Monitoring Demo ===")
    
    # Perform health check
    health_info = await plugin_manager.health_check()
    
    logger.info(f"Overall system status: {health_info['status']}")
    logger.info(f"Hot-reload active: {health_info['hot_reload_active']}")
    logger.info(f"Plugin directories: {len(health_info['plugin_directories'])}")
    
    # Show plugin-specific health
    plugins_health = health_info['plugins']
    logger.info(f"\nPlugin health status ({len(plugins_health)} plugins):")
    
    for plugin_name, health in plugins_health.items():
        logger.info(f"  {plugin_name}:")
        logger.info(f"    Enabled: {health['enabled']}")
        logger.info(f"    Load count: {health['load_count']}")
        logger.info(f"    Error count: {health['error_count']}")
        logger.info(f"    Instance creation: {health['instance_creation']}")
        
        if health['last_error']:
            logger.warning(f"    Last error: {health['last_error']}")
    
    # Show statistics
    stats = health_info['statistics']
    logger.info(f"\nSystem statistics:")
    logger.info(f"  Total plugins: {stats['total_plugins']}")
    logger.info(f"  Enabled plugins: {stats['enabled_plugins']}")
    logger.info(f"  Plugins with errors: {stats['plugins_with_errors']}")
    logger.info(f"  Reload count: {stats['reload_count']}")


async def demonstrate_hot_reload(plugin_manager: PluginManager):
    """Demonstrate hot-reload functionality."""
    logger.info("\n=== Hot-Reload Demo ===")
    
    plugins_before = plugin_manager.list_plugins()
    logger.info(f"Plugins loaded before reload: {len(plugins_before)}")
    
    # Try to reload a plugin manually
    plugin_names = list(plugins_before.keys())
    if plugin_names:
        plugin_to_reload = plugin_names[0]
        logger.info(f"Attempting to reload plugin: {plugin_to_reload}")
        
        success = await plugin_manager.reload_plugin(plugin_to_reload)
        logger.info(f"Reload success: {success}")
        
        # Check statistics after reload
        stats = plugin_manager.get_load_statistics()
        logger.info(f"Reload count: {stats['reload_count']}")
    
    plugins_after = plugin_manager.list_plugins()
    logger.info(f"Plugins loaded after reload: {len(plugins_after)}")


async def demonstrate_performance_metrics(config_manager: PluginConfigurationManager):
    """Demonstrate performance metrics collection."""
    logger.info("\n=== Performance Metrics Demo ===")
    
    # Get runtime metrics
    metrics = config_manager.get_runtime_metrics()
    
    logger.info("Runtime metrics:")
    logger.info(f"  Last updated: {metrics['last_updated']}")
    logger.info(f"  Total strategies: {metrics['total_strategies']}")
    logger.info(f"  Enabled strategies: {metrics['enabled_strategies']}")
    logger.info(f"  Plugin directories: {metrics['plugin_directories']}")
    logger.info(f"  Runtime overrides: {metrics['runtime_overrides']}")
    logger.info(f"  Configuration errors: {metrics['configuration_errors']}")


async def demonstrate_configuration_export(config_manager: PluginConfigurationManager):
    """Demonstrate configuration export functionality."""
    logger.info("\n=== Configuration Export Demo ===")
    
    # Export as dictionary
    config_dict = config_manager.export_configuration("dict")
    logger.info(f"Configuration exported as dict: {len(config_dict)} sections")
    
    # Export as JSON
    config_json = config_manager.export_configuration("json")
    logger.info(f"Configuration exported as JSON: {len(config_json)} characters")
    
    # Show sample of JSON export (first 200 characters)
    logger.info(f"JSON sample: {config_json[:200]}...")
    
    try:
        # Export as YAML if available
        config_yaml = config_manager.export_configuration("yaml")
        logger.info(f"Configuration exported as YAML: {len(config_yaml)} characters")
    except (ImportError, ValueError):
        logger.info("YAML export not available (yaml package not installed)")


async def main():
    """Main demonstration function."""
    logger.info("Starting Plugin System Demo")
    logger.info("=" * 50)
    
    plugin_manager = None
    config_manager = None
    
    try:
        # 1. Plugin Discovery and Loading
        plugin_manager, config_manager = await demonstrate_plugin_discovery()
        
        # 2. Strategy Execution
        await demonstrate_strategy_execution(plugin_manager)
        
        # 3. Configuration Management
        await demonstrate_configuration_management(config_manager)
        
        # 4. Health Monitoring
        await demonstrate_health_monitoring(plugin_manager)
        
        # 5. Hot-Reload Demo
        await demonstrate_hot_reload(plugin_manager)
        
        # 6. Performance Metrics
        await demonstrate_performance_metrics(config_manager)
        
        # 7. Configuration Export
        await demonstrate_configuration_export(config_manager)
        
        logger.info("\n" + "=" * 50)
        logger.info("Plugin System Demo Completed Successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        if plugin_manager:
            logger.info("Shutting down plugin manager...")
            await plugin_manager.shutdown()
            logger.info("Plugin manager shutdown complete")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())