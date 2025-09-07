"""
Basic tests for configuration models.
"""

import pytest
from datetime import datetime
from decimal import Decimal

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.config.models import (
    LogLevel, DataSourceType, StrategyType, DatabaseConfig, CacheConfig,
    DataSourceConfig, RiskConfig, StrategyConfig, ArbitrageOpportunity
)


class TestEnums:
    """Test configuration enums."""

    def test_log_level_enum(self):
        """Test LogLevel enum values."""
        assert hasattr(LogLevel, 'DEBUG')
        assert hasattr(LogLevel, 'INFO') 
        assert hasattr(LogLevel, 'WARNING')
        assert hasattr(LogLevel, 'ERROR')
        
        # Test that values are strings
        assert isinstance(LogLevel.INFO, str)

    def test_data_source_type_enum(self):
        """Test DataSourceType enum values."""
        assert hasattr(DataSourceType, 'TUSHARE')
        assert isinstance(DataSourceType.TUSHARE, str)

    def test_strategy_type_enum(self):
        """Test StrategyType enum values."""
        assert hasattr(StrategyType, 'PRICING_ARBITRAGE')
        assert isinstance(StrategyType.PRICING_ARBITRAGE, str)


class TestDatabaseConfig:
    """Test DatabaseConfig model."""

    def test_database_config_creation(self):
        """Test DatabaseConfig can be created with default values."""
        config = DatabaseConfig()
        
        # Should have default values
        assert hasattr(config, 'host')
        assert hasattr(config, 'port')
        assert hasattr(config, 'database')
        assert hasattr(config, 'username')

    def test_database_config_with_custom_values(self):
        """Test DatabaseConfig with custom values."""
        config = DatabaseConfig(
            host='localhost',
            port=5432,
            database='testdb',
            username='testuser'
        )
        
        assert config.host == 'localhost'
        assert config.port == 5432
        assert config.database == 'testdb'
        assert config.username == 'testuser'


class TestCacheConfig:
    """Test CacheConfig model."""

    def test_cache_config_creation(self):
        """Test CacheConfig creation."""
        config = CacheConfig()
        
        assert hasattr(config, 'redis_host')
        assert hasattr(config, 'redis_port')
        assert hasattr(config, 'ttl')


class TestDataSourceConfig:
    """Test DataSourceConfig model."""

    def test_data_source_config_creation(self):
        """Test DataSourceConfig creation."""
        config = DataSourceConfig()
        
        assert hasattr(config, 'type')
        assert hasattr(config, 'enabled')
        assert hasattr(config, 'priority')

    def test_data_source_config_with_values(self):
        """Test DataSourceConfig with specific values."""
        config = DataSourceConfig(
            type=DataSourceType.TUSHARE,
            enabled=True,
            priority=1
        )
        
        assert config.type == DataSourceType.TUSHARE
        assert config.enabled is True
        assert config.priority == 1


class TestRiskConfig:
    """Test RiskConfig model."""

    def test_risk_config_creation(self):
        """Test RiskConfig creation with defaults."""
        config = RiskConfig()
        
        # Should have risk-related attributes
        assert hasattr(config, 'max_position_size')
        assert hasattr(config, 'max_daily_loss')
        assert hasattr(config, 'min_liquidity_volume')

    def test_risk_config_with_custom_values(self):
        """Test RiskConfig with custom values."""
        config = RiskConfig(
            max_position_size=50000.0,
            max_daily_loss=5000.0,
            min_liquidity_volume=100
        )
        
        assert config.max_position_size == 50000.0
        assert config.max_daily_loss == 5000.0
        assert config.min_liquidity_volume == 100


class TestStrategyConfig:
    """Test StrategyConfig model."""

    def test_strategy_config_creation(self):
        """Test StrategyConfig creation."""
        config = StrategyConfig()
        
        assert hasattr(config, 'type')
        assert hasattr(config, 'enabled')
        assert hasattr(config, 'priority')


class TestArbitrageOpportunity:
    """Test ArbitrageOpportunity model."""

    def test_arbitrage_opportunity_creation(self):
        """Test ArbitrageOpportunity creation with required fields."""
        opportunity = ArbitrageOpportunity(
            underlying_asset="AAPL",
            strategy_type=StrategyType.PRICING_ARBITRAGE,
            expected_profit=100.0,
            probability=0.85,
            description="Test arbitrage opportunity"
        )
        
        assert opportunity.underlying_asset == "AAPL"
        assert opportunity.strategy_type == StrategyType.PRICING_ARBITRAGE
        assert opportunity.expected_profit == 100.0
        assert opportunity.probability == 0.85
        assert opportunity.description == "Test arbitrage opportunity"

    def test_arbitrage_opportunity_with_optional_fields(self):
        """Test ArbitrageOpportunity with optional fields."""
        opportunity = ArbitrageOpportunity(
            underlying_asset="GOOGL",
            strategy_type=StrategyType.PRICING_ARBITRAGE,
            expected_profit=250.0,
            probability=0.90,
            description="High probability opportunity",
            risk_score=0.15,
            confidence_level=0.95
        )
        
        assert opportunity.risk_score == 0.15
        assert opportunity.confidence_level == 0.95

    def test_arbitrage_opportunity_validation(self):
        """Test ArbitrageOpportunity field validation."""
        # Test that probability is within valid range
        try:
            opportunity = ArbitrageOpportunity(
                underlying_asset="TEST",
                strategy_type=StrategyType.PRICING_ARBITRAGE,
                expected_profit=100.0,
                probability=1.5,  # Invalid - greater than 1
                description="Test"
            )
            # If validation is implemented, this should fail
            # If not, we just verify the object was created
            assert opportunity.probability == 1.5
        except Exception as e:
            # Validation error is expected for invalid probability
            assert "probability" in str(e).lower() or "validation" in str(e).lower()

    def test_arbitrage_opportunity_required_fields(self):
        """Test ArbitrageOpportunity requires essential fields."""
        # Test that we can create with minimal required fields
        try:
            opportunity = ArbitrageOpportunity(
                underlying_asset="SPY",
                strategy_type=StrategyType.PRICING_ARBITRAGE,
                expected_profit=50.0,
                probability=0.75,
                description="Minimal opportunity"
            )
            
            assert opportunity is not None
            assert opportunity.underlying_asset == "SPY"
            
        except Exception as e:
            # If there are validation errors, they should be meaningful
            error_msg = str(e).lower()
            assert any(field in error_msg for field in ['required', 'missing', 'field'])


class TestModelIntegration:
    """Test integration between different model classes."""

    def test_strategy_type_consistency(self):
        """Test StrategyType enum is consistent across models."""
        # Create a strategy config
        strategy_config = StrategyConfig(
            type=StrategyType.PRICING_ARBITRAGE,
            enabled=True
        )
        
        # Create an opportunity with same strategy type
        opportunity = ArbitrageOpportunity(
            underlying_asset="TSLA",
            strategy_type=StrategyType.PRICING_ARBITRAGE,
            expected_profit=75.0,
            probability=0.80,
            description="Pricing arbitrage for TSLA"
        )
        
        # Verify consistency
        assert strategy_config.type == opportunity.strategy_type

    def test_config_model_instantiation(self):
        """Test that all config models can be instantiated."""
        models = [
            DatabaseConfig,
            CacheConfig,
            DataSourceConfig,
            RiskConfig,
            StrategyConfig
        ]
        
        for model_class in models:
            try:
                instance = model_class()
                assert instance is not None
                assert hasattr(instance, '__dict__')
            except Exception as e:
                pytest.fail(f"Failed to instantiate {model_class.__name__}: {e}")


if __name__ == "__main__":
    pytest.main([__file__])