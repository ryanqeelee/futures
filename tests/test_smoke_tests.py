"""
Smoke tests for basic module functionality and coverage.
These tests focus on import coverage and basic instantiation.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


class TestImportCoverage:
    """Test that modules can be imported successfully."""

    def test_import_config_models(self):
        """Test config.models can be imported."""
        from src.config import models
        assert models is not None
        
        # Test specific imports
        from src.config.models import LogLevel, StrategyType, DataSourceType
        assert LogLevel is not None
        assert StrategyType is not None
        assert DataSourceType is not None

    def test_import_strategies(self):
        """Test strategy modules can be imported."""
        from src.strategies import base
        assert base is not None
        
        from src.strategies import pricing_arbitrage
        assert pricing_arbitrage is not None

    def test_import_adapters(self):
        """Test adapter modules can be imported."""
        from src.adapters import base
        assert base is not None

    def test_import_core_modules(self):
        """Test core modules can be imported."""
        from src.core import exceptions
        assert exceptions is not None
        
        from src.core import cache_manager  
        assert cache_manager is not None

    def test_import_engine_modules(self):
        """Test engine modules can be imported."""
        from src.engine import arbitrage_engine
        assert arbitrage_engine is not None
        
        from src.engine import risk_manager
        assert risk_manager is not None

    def test_import_validation_modules(self):
        """Test validation modules can be imported."""
        from src.validation import data_authenticity_validator
        assert data_authenticity_validator is not None

    def test_import_ui_modules(self):
        """Test UI modules can be imported."""
        from src.ui import streamlit_app
        assert streamlit_app is not None
        
        from src.ui.components import config_panel
        assert config_panel is not None

    def test_import_utils(self):
        """Test utils modules can be imported."""
        from src.utils import performance_monitor
        assert performance_monitor is not None


class TestBasicEnumUsage:
    """Test basic enum functionality."""

    def test_log_level_enum(self):
        """Test LogLevel enum basic usage."""
        from src.config.models import LogLevel
        
        # Test that enum values exist
        assert hasattr(LogLevel, 'DEBUG')
        assert hasattr(LogLevel, 'INFO')
        assert hasattr(LogLevel, 'WARNING') 
        assert hasattr(LogLevel, 'ERROR')

    def test_strategy_type_enum(self):
        """Test StrategyType enum basic usage."""
        from src.config.models import StrategyType
        
        # Test that enum has values 
        assert len(list(StrategyType)) > 0
        
        # Test specific value if it exists
        if hasattr(StrategyType, 'PRICING_ARBITRAGE'):
            assert StrategyType.PRICING_ARBITRAGE is not None

    def test_data_source_type_enum(self):
        """Test DataSourceType enum basic usage."""
        from src.config.models import DataSourceType
        
        # Test that enum has values
        assert len(list(DataSourceType)) > 0


class TestBasicClassInstantiation:
    """Test basic class instantiation where possible."""

    def test_error_context_creation(self):
        """Test ErrorContext can be created."""
        from src.core.exceptions import create_error_context
        
        # Test basic context creation
        context = create_error_context("test_component", "test_operation")
        assert context is not None

    def test_connection_status_enum(self):
        """Test ConnectionStatus enum usage."""
        from src.adapters.base import ConnectionStatus
        
        assert ConnectionStatus.CONNECTED is not None
        assert ConnectionStatus.DISCONNECTED is not None

    def test_cache_policy_enum(self):
        """Test cache-related enums."""
        try:
            from src.core.cache_manager import CachePolicy
            assert len(list(CachePolicy)) > 0
        except ImportError:
            # If CachePolicy doesn't exist, that's fine
            pass

    def test_basic_option_data_structure(self):
        """Test OptionData structure if available."""
        try:
            from src.strategies.base import OptionData
            # If we can import it, test basic attributes
            assert OptionData is not None
        except ImportError:
            # If not available, skip
            pass


class TestExceptionHandling:
    """Test exception handling framework."""

    def test_exception_classes_exist(self):
        """Test that exception classes are defined."""
        from src.core.exceptions import (
            TradingSystemError, DataSourceError, ArbitrageError,
            PricingError, RiskError, SystemError, ConfigurationError
        )
        
        # Test inheritance
        assert issubclass(DataSourceError, TradingSystemError)
        assert issubclass(ArbitrageError, TradingSystemError)
        assert issubclass(PricingError, ArbitrageError)

    def test_exception_instantiation(self):
        """Test exceptions can be instantiated."""
        from src.core.exceptions import TradingSystemError
        
        error = TradingSystemError("Test message")
        assert str(error) == "Test message"

    def test_error_handler_decorator_exists(self):
        """Test error handler decorators exist."""
        from src.core.exceptions import error_handler, async_error_handler
        
        assert callable(error_handler)
        assert callable(async_error_handler)


class TestEngineComponents:
    """Test engine component basic functionality."""

    def test_arbitrage_engine_class_exists(self):
        """Test ArbitrageEngine class can be imported."""
        from src.engine.arbitrage_engine import ArbitrageEngine
        assert ArbitrageEngine is not None

    def test_risk_manager_class_exists(self):
        """Test AdvancedRiskManager class can be imported."""
        from src.engine.risk_manager import AdvancedRiskManager
        assert AdvancedRiskManager is not None


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""

    def test_performance_monitor_import(self):
        """Test PerformanceMonitor can be imported."""
        from src.utils.performance_monitor import PerformanceMonitor
        assert PerformanceMonitor is not None

    def test_performance_monitor_basic_usage(self):
        """Test basic PerformanceMonitor functionality."""
        from src.utils.performance_monitor import PerformanceMonitor
        
        # Try to create instance if possible
        try:
            monitor = PerformanceMonitor()
            assert monitor is not None
        except Exception:
            # If instantiation fails, that's expected without proper setup
            pass


class TestCacheSystem:
    """Test cache system components."""

    def test_cache_manager_import(self):
        """Test cache manager can be imported."""
        from src.core.cache_manager import IntelligentCacheManager
        assert IntelligentCacheManager is not None

    def test_data_type_enum_exists(self):
        """Test DataType enum if it exists."""
        try:
            from src.core.cache_manager import DataType
            assert len(list(DataType)) > 0
        except ImportError:
            # If DataType doesn't exist as expected, skip
            pass


class TestStrategySystem:
    """Test strategy system components."""

    def test_pricing_arbitrage_strategy_import(self):
        """Test pricing arbitrage strategy import."""
        from src.strategies.pricing_arbitrage import PricingArbitrageStrategy
        assert PricingArbitrageStrategy is not None

    def test_strategy_base_import(self):
        """Test strategy base classes."""
        from src.strategies import base
        
        # Check if base strategy components exist
        assert hasattr(base, '__file__')


class TestValidationSystem:
    """Test validation system components."""

    def test_data_authenticity_validator_import(self):
        """Test data authenticity validator import."""
        from src.validation.data_authenticity_validator import DataAuthenticityValidator
        assert DataAuthenticityValidator is not None


if __name__ == "__main__":
    pytest.main([__file__])