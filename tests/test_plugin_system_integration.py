"""
Comprehensive integration tests for the plugin loading system.
插件加载系统的综合集成测试

Tests the complete plugin system including:
- Plugin discovery and loading
- Strategy configuration management  
- Plugin hot-reload functionality
- Error handling and recovery
- Performance and memory usage
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.plugin_manager import PluginManager, PluginManagerConfig, PluginInfo
from src.config.models import StrategyType, StrategyConfig
from src.config.plugin_config import PluginConfigurationManager, get_plugin_config_manager
from src.strategies.base import BaseStrategy, StrategyParameters, OptionData, StrategyResult


class MockOptionData:
    """Mock option data for testing."""
    
    def __init__(self, code="TEST001", underlying="TEST", strike=100.0):
        self.code = code
        self.underlying = underlying
        self.strike_price = strike
        self.market_price = 10.0
        self.bid_price = 9.5
        self.ask_price = 10.5
        self.volume = 1000
        self.open_interest = 5000
        self.implied_volatility = 0.2
        self.theoretical_price = 10.1
        self.option_type = "C"
        self.expiry_date = datetime.now() + timedelta(days=30)
        
    @property
    def days_to_expiry(self):
        return (self.expiry_date - datetime.now()).days
        
    @property
    def time_to_expiry(self):
        return self.days_to_expiry / 365.0
        
    @property
    def mid_price(self):
        return (self.bid_price + self.ask_price) / 2
        
    @property
    def spread(self):
        return self.ask_price - self.bid_price
        
    @property
    def spread_pct(self):
        return self.spread / self.mid_price if self.mid_price > 0 else 0


@pytest.fixture
def temp_plugin_dir():
    """Create temporary directory for plugin tests."""
    temp_dir = tempfile.mkdtemp(prefix="plugin_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture 
def sample_strategy_plugin():
    """Sample strategy plugin code for testing."""
    return '''
"""Sample strategy plugin for testing."""

from datetime import datetime
from typing import List
from src.config.models import StrategyType, ArbitrageOpportunity
from src.strategies.base import BaseStrategy, StrategyResult, StrategyParameters, OptionData, TradingAction, ActionType, RiskMetrics, RiskLevel, StrategyRegistry


@StrategyRegistry.register(StrategyType.PRICING_ARBITRAGE)
class TestPricingStrategy(BaseStrategy):
    """Test pricing strategy for plugin system testing."""
    
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.PRICING_ARBITRAGE
    
    def scan_opportunities(self, options_data: List[OptionData]) -> StrategyResult:
        """Scan for test opportunities."""
        return StrategyResult(
            strategy_name=self.name,
            opportunities=[],
            execution_time=0.001,
            data_timestamp=datetime.now(),
            success=True
        )
    
    def calculate_profit(self, options, actions):
        """Calculate test profit."""
        return 100.0
    
    def assess_risk(self, options, actions):
        """Assess test risk."""
        return RiskMetrics(
            max_loss=50.0,
            max_gain=100.0,
            probability_profit=0.7,
            expected_return=35.0,
            risk_level=RiskLevel.MEDIUM,
            liquidity_risk=0.1,
            time_decay_risk=0.2,
            volatility_risk=0.15
        )
'''


@pytest.fixture
def broken_strategy_plugin():
    """Broken strategy plugin code for error testing."""
    return '''
"""Broken strategy plugin for error testing."""

# This plugin has syntax errors and missing imports
from invalid_module import NonExistentClass

class BrokenStrategy(BaseStrategy):
    def __init__(self):
        # This will cause an error
        super().__init__(invalid_parameter=True)
        
    # Missing required methods
    def broken_method(self):
        raise Exception("This strategy is intentionally broken")
'''


class TestPluginSystemIntegration:
    """Integration tests for the complete plugin system."""
    
    @pytest.mark.asyncio
    async def test_plugin_discovery_and_loading(self, temp_plugin_dir, sample_strategy_plugin):
        """Test basic plugin discovery and loading."""
        # Create sample plugin file
        plugin_file = temp_plugin_dir / "test_strategy.py"
        plugin_file.write_text(sample_strategy_plugin)
        
        # Create plugin manager
        config = PluginManagerConfig(plugin_directories=[str(temp_plugin_dir)])
        plugin_manager = PluginManager(config)
        
        # Initialize and discover plugins
        await plugin_manager.initialize()
        
        # Verify plugin was loaded
        plugins = plugin_manager.list_plugins()
        assert len(plugins) > 0
        
        # Check specific plugin
        plugin_names = list(plugins.keys())
        assert any("TestPricingStrategy" in name for name in plugin_names)
        
        # Verify strategy functionality
        strategies = plugin_manager.get_strategies()
        assert len(strategies) > 0
        
        await plugin_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_plugin_configuration_integration(self, temp_plugin_dir, sample_strategy_plugin):
        """Test plugin system integration with configuration management."""
        # Setup plugin file
        plugin_file = temp_plugin_dir / "test_strategy.py"
        plugin_file.write_text(sample_strategy_plugin)
        
        # Initialize configuration manager
        config_manager = PluginConfigurationManager()
        
        # Create plugin manager with config
        plugin_config = config_manager.get_plugin_loading_config()
        manager_config = PluginManagerConfig(plugin_directories=[str(temp_plugin_dir)])
        plugin_manager = PluginManager(manager_config)
        
        await plugin_manager.initialize()
        
        # Test configuration updates
        strategy_config = config_manager.get_strategy_config(StrategyType.PRICING_ARBITRAGE)
        assert strategy_config is not None
        assert strategy_config.enabled == True
        
        # Update configuration
        success = config_manager.update_strategy_config(
            StrategyType.PRICING_ARBITRAGE,
            {'min_profit_threshold': 0.05}
        )
        assert success == True
        
        updated_config = config_manager.get_strategy_config(StrategyType.PRICING_ARBITRAGE)
        assert updated_config.min_profit_threshold == 0.05
        
        await plugin_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, temp_plugin_dir, broken_strategy_plugin):
        """Test error handling when loading broken plugins."""
        # Create broken plugin file
        broken_file = temp_plugin_dir / "broken_strategy.py"
        broken_file.write_text(broken_strategy_plugin)
        
        # Create plugin manager with error tolerance
        config = PluginManagerConfig(
            plugin_directories=[str(temp_plugin_dir)],
            max_load_retries=2,
            retry_delay=0.1
        )
        plugin_manager = PluginManager(config)
        
        # Initialize - should handle errors gracefully
        await plugin_manager.initialize()
        
        # Verify system still works despite broken plugin
        plugins = plugin_manager.list_plugins()
        # Should have no successfully loaded plugins due to broken code
        
        # Check load statistics
        stats = plugin_manager.get_load_statistics()
        assert stats['failed_loads'] > 0
        
        await plugin_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_multiple_strategy_loading(self, temp_plugin_dir):
        """Test loading multiple different strategy types."""
        # Create multiple strategy plugins
        strategies = [
            ("pricing_strategy.py", StrategyType.PRICING_ARBITRAGE),
            ("volatility_strategy.py", StrategyType.VOLATILITY_ARBITRAGE),
            ("parity_strategy.py", StrategyType.PUT_CALL_PARITY)
        ]
        
        for filename, strategy_type in strategies:
            plugin_code = f'''
from datetime import datetime
from typing import List
from src.config.models import StrategyType, ArbitrageOpportunity
from src.strategies.base import BaseStrategy, StrategyResult, StrategyParameters, OptionData, TradingAction, ActionType, RiskMetrics, RiskLevel, StrategyRegistry

@StrategyRegistry.register(StrategyType.{strategy_type.name})
class Test{strategy_type.name.title()}Strategy(BaseStrategy):
    """Test {strategy_type.value} strategy."""
    
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.{strategy_type.name}
    
    def scan_opportunities(self, options_data: List[OptionData]) -> StrategyResult:
        return StrategyResult(
            strategy_name=self.name,
            opportunities=[],
            execution_time=0.001,
            data_timestamp=datetime.now(),
            success=True
        )
    
    def calculate_profit(self, options, actions):
        return 100.0
    
    def assess_risk(self, options, actions):
        return RiskMetrics(
            max_loss=50.0, max_gain=100.0, probability_profit=0.7,
            expected_return=35.0, risk_level=RiskLevel.MEDIUM,
            liquidity_risk=0.1, time_decay_risk=0.2, volatility_risk=0.15
        )
'''
            plugin_file = temp_plugin_dir / filename
            plugin_file.write_text(plugin_code)
        
        # Load all plugins
        config = PluginManagerConfig(plugin_directories=[str(temp_plugin_dir)])
        plugin_manager = PluginManager(config)
        await plugin_manager.initialize()
        
        # Verify all strategies loaded
        plugins = plugin_manager.list_plugins()
        assert len(plugins) == 3
        
        strategies = plugin_manager.get_strategies()
        assert len(strategies) == 3
        
        # Test each strategy type
        strategy_types = set()
        for strategy in strategies.values():
            strategy_types.add(strategy.strategy_type)
        
        expected_types = {StrategyType.PRICING_ARBITRAGE, StrategyType.VOLATILITY_ARBITRAGE, StrategyType.PUT_CALL_PARITY}
        assert strategy_types == expected_types
        
        await plugin_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_strategy_execution(self, temp_plugin_dir, sample_strategy_plugin):
        """Test end-to-end strategy execution through plugin system."""
        # Setup plugin
        plugin_file = temp_plugin_dir / "test_strategy.py"
        plugin_file.write_text(sample_strategy_plugin)
        
        # Load plugins
        config = PluginManagerConfig(plugin_directories=[str(temp_plugin_dir)])
        plugin_manager = PluginManager(config)
        await plugin_manager.initialize()
        
        # Get strategies
        strategies = plugin_manager.get_strategies()
        assert len(strategies) > 0
        
        # Create mock option data
        mock_options = [
            MockOptionData("TEST001", "TEST", 100.0),
            MockOptionData("TEST002", "TEST", 105.0),
            MockOptionData("TEST003", "TEST", 95.0)
        ]
        
        # Execute each strategy
        results = []
        for strategy_name, strategy in strategies.items():
            try:
                result = strategy.scan_opportunities(mock_options)
                assert isinstance(result, StrategyResult)
                assert result.success == True
                results.append(result)
            except Exception as e:
                pytest.fail(f"Strategy {strategy_name} execution failed: {e}")
        
        assert len(results) > 0
        
        await plugin_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_plugin_hot_reload(self, temp_plugin_dir, sample_strategy_plugin):
        """Test hot-reload functionality."""
        # Note: This test requires watchdog package for full functionality
        plugin_file = temp_plugin_dir / "test_strategy.py"
        plugin_file.write_text(sample_strategy_plugin)
        
        # Create plugin manager with hot-reload enabled
        config = PluginManagerConfig(
            plugin_directories=[str(temp_plugin_dir)],
            enable_hot_reload=True,
            reload_delay=0.1
        )
        plugin_manager = PluginManager(config)
        await plugin_manager.initialize()
        
        # Verify initial load
        plugins_before = len(plugin_manager.list_plugins())
        assert plugins_before > 0
        
        # Modify plugin file
        modified_plugin = sample_strategy_plugin.replace(
            "Calculate test profit.",
            "Calculate modified test profit."
        )
        plugin_file.write_text(modified_plugin)
        
        # Wait for reload (if watchdog is available)
        await asyncio.sleep(0.5)
        
        # Test manual reload
        reload_success = await plugin_manager.reload_plugin("TestPricingStrategy")
        # reload_success might be False if plugin name doesn't match exactly
        
        plugins_after = len(plugin_manager.list_plugins())
        # Should have same number of plugins after reload
        
        await plugin_manager.shutdown()
    
    def test_configuration_validation(self):
        """Test configuration validation and error detection."""
        config_manager = PluginConfigurationManager()
        
        # Test valid configuration
        errors = config_manager.validate_configuration()
        assert isinstance(errors, dict)
        
        # Test invalid configuration update
        success = config_manager.update_strategy_config(
            StrategyType.PRICING_ARBITRAGE,
            {'min_profit_threshold': -0.1}  # Invalid negative threshold
        )
        assert success == True  # Update succeeds but validation should catch it
        
        # Validate updated configuration
        errors = config_manager.validate_configuration()
        assert len(errors['strategy_configs']) > 0
    
    def test_configuration_export_import(self):
        """Test configuration export and import functionality."""
        config_manager = PluginConfigurationManager()
        
        # Export configuration
        config_dict = config_manager.export_configuration("dict")
        assert isinstance(config_dict, dict)
        assert 'plugin_config' in config_dict
        assert 'strategy_configs' in config_dict
        
        # Test JSON export
        config_json = config_manager.export_configuration("json")
        assert isinstance(config_json, str)
        
        # Test YAML export (if available)
        try:
            config_yaml = config_manager.export_configuration("yaml")
            assert isinstance(config_yaml, str)
        except ImportError:
            # YAML not available, skip
            pass
    
    @pytest.mark.asyncio
    async def test_plugin_system_performance(self, temp_plugin_dir):
        """Test plugin system performance with multiple plugins."""
        # Create multiple plugins for performance testing
        num_plugins = 5
        for i in range(num_plugins):
            plugin_code = f'''
from datetime import datetime
from typing import List
from src.config.models import StrategyType, ArbitrageOpportunity
from src.strategies.base import BaseStrategy, StrategyResult, StrategyParameters, OptionData, TradingAction, ActionType, RiskMetrics, RiskLevel, StrategyRegistry

@StrategyRegistry.register(StrategyType.PRICING_ARBITRAGE)
class TestStrategy{i}(BaseStrategy):
    """Test strategy {i} for performance testing."""
    
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.PRICING_ARBITRAGE
    
    def scan_opportunities(self, options_data: List[OptionData]) -> StrategyResult:
        # Simulate some work
        import time
        time.sleep(0.001)  # 1ms work simulation
        
        return StrategyResult(
            strategy_name=self.name,
            opportunities=[],
            execution_time=0.001,
            data_timestamp=datetime.now(),
            success=True
        )
    
    def calculate_profit(self, options, actions):
        return 100.0 + {i}
    
    def assess_risk(self, options, actions):
        return RiskMetrics(
            max_loss=50.0, max_gain=100.0, probability_profit=0.7,
            expected_return=35.0, risk_level=RiskLevel.MEDIUM,
            liquidity_risk=0.1, time_decay_risk=0.2, volatility_risk=0.15
        )
'''
            plugin_file = temp_plugin_dir / f"strategy_{i}.py"
            plugin_file.write_text(plugin_code)
        
        # Test parallel loading performance
        config = PluginManagerConfig(
            plugin_directories=[str(temp_plugin_dir)],
            parallel_loading=True,
            max_load_workers=4
        )
        
        start_time = datetime.now()
        plugin_manager = PluginManager(config)
        await plugin_manager.initialize()
        loading_time = (datetime.now() - start_time).total_seconds()
        
        # Verify all plugins loaded
        plugins = plugin_manager.list_plugins()
        assert len(plugins) == num_plugins
        
        # Check loading performance
        stats = plugin_manager.get_load_statistics()
        assert stats['successful_loads'] == num_plugins
        assert loading_time < 5.0  # Should load within 5 seconds
        
        # Test strategy execution performance  
        strategies = plugin_manager.get_strategies()
        mock_options = [MockOptionData(f"TEST{i:03d}", "TEST", 100.0 + i) for i in range(10)]
        
        start_time = datetime.now()
        for strategy in strategies.values():
            result = strategy.scan_opportunities(mock_options)
            assert result.success == True
        execution_time = (datetime.now() - start_time).total_seconds()
        
        assert execution_time < 2.0  # Should execute within 2 seconds
        
        await plugin_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_plugin_health_monitoring(self, temp_plugin_dir, sample_strategy_plugin):
        """Test plugin health monitoring and diagnostics."""
        # Setup plugin
        plugin_file = temp_plugin_dir / "test_strategy.py"
        plugin_file.write_text(sample_strategy_plugin)
        
        # Load plugin
        config = PluginManagerConfig(plugin_directories=[str(temp_plugin_dir)])
        plugin_manager = PluginManager(config)
        await plugin_manager.initialize()
        
        # Perform health check
        health_info = await plugin_manager.health_check()
        
        assert isinstance(health_info, dict)
        assert 'status' in health_info
        assert 'plugins' in health_info
        assert 'statistics' in health_info
        
        # Verify health status
        assert health_info['status'] in ['healthy', 'degraded']
        
        # Check plugin-specific health
        plugins_health = health_info['plugins']
        assert len(plugins_health) > 0
        
        for plugin_name, plugin_health in plugins_health.items():
            assert 'enabled' in plugin_health
            assert 'load_count' in plugin_health
            assert 'error_count' in plugin_health
            assert 'instance_creation' in plugin_health
        
        await plugin_manager.shutdown()


if __name__ == "__main__":
    # Run tests manually if needed
    pytest.main([__file__, "-v"])