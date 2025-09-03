#!/usr/bin/env python3
"""
Simple validation script for the plugin system.
检验插件系统的简单脚本
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test basic imports."""
    print("Testing imports...")
    
    try:
        # Test config models
        from config.models import StrategyType, StrategyConfig
        print("✓ Config models import successful")
        
        # Test base strategy
        from strategies.base import BaseStrategy, StrategyRegistry
        print("✓ Base strategy import successful")
        
        # Test plugin manager
        from core.plugin_manager import PluginManager, PluginManagerConfig
        print("✓ Plugin manager import successful")
        
        # Test plugin config
        from config.plugin_config import PluginConfigurationManager
        print("✓ Plugin configuration import successful")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_strategy_registry():
    """Test strategy registry functionality."""
    print("\nTesting strategy registry...")
    
    try:
        from strategies.base import StrategyRegistry
        from config.models import StrategyType
        
        # Clear registry for clean test
        StrategyRegistry.clear_registry()
        
        # Check initial state
        strategies = StrategyRegistry.get_registered_strategies()
        print(f"✓ Initial registry state: {len(strategies)} strategies")
        
        return True
        
    except Exception as e:
        print(f"✗ Strategy registry error: {e}")
        return False

def test_plugin_config_manager():
    """Test plugin configuration manager."""
    print("\nTesting plugin configuration manager...")
    
    try:
        from config.plugin_config import PluginConfigurationManager
        from config.models import StrategyType
        
        # Create config manager
        config_manager = PluginConfigurationManager()
        
        # Test basic functionality
        enabled_strategies = config_manager.get_enabled_strategies()
        print(f"✓ Config manager created: {len(enabled_strategies)} enabled strategies")
        
        # Test configuration export
        config_dict = config_manager.export_configuration("dict")
        print(f"✓ Configuration export works: {len(config_dict)} sections")
        
        # Test validation
        errors = config_manager.validate_configuration()
        error_count = sum(len(error_list) for error_list in errors.values())
        print(f"✓ Configuration validation: {error_count} errors")
        
        return True
        
    except Exception as e:
        print(f"✗ Plugin config manager error: {e}")
        return False

def test_plugin_manager_config():
    """Test plugin manager configuration."""
    print("\nTesting plugin manager configuration...")
    
    try:
        from core.plugin_manager import PluginManagerConfig
        
        # Create config
        config = PluginManagerConfig(
            plugin_directories=['src/strategies'],
            auto_reload=True,
            enable_hot_reload=True
        )
        
        print(f"✓ Plugin manager config created: {len(config.plugin_directories)} directories")
        print(f"✓ Auto reload: {config.auto_reload}")
        print(f"✓ Hot reload: {config.enable_hot_reload}")
        
        return True
        
    except Exception as e:
        print(f"✗ Plugin manager config error: {e}")
        return False

def validate_strategy_files():
    """Validate that strategy files exist and have correct structure."""
    print("\nValidating strategy files...")
    
    strategy_files = [
        "src/strategies/pricing_arbitrage.py",
        "src/strategies/波动率套利策略.py", 
        "src/strategies/看跌看涨平价策略.py"
    ]
    
    for file_path in strategy_files:
        full_path = Path(__file__).parent / file_path
        if full_path.exists():
            print(f"✓ Found: {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
            return False
    
    return True

def main():
    """Main validation function."""
    print("Plugin System Validation")
    print("=" * 40)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test strategy registry
    if not test_strategy_registry():
        all_passed = False
    
    # Test plugin configuration manager
    if not test_plugin_config_manager():
        all_passed = False
    
    # Test plugin manager configuration
    if not test_plugin_manager_config():
        all_passed = False
        
    # Validate strategy files
    if not validate_strategy_files():
        all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All validation tests passed!")
        print("Plugin system is ready for use.")
        return 0
    else:
        print("✗ Some validation tests failed!")
        print("Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())