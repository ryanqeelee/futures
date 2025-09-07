"""
Basic tests for pricing arbitrage strategy.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.strategies.pricing_arbitrage import PricingArbitrageStrategy
from src.strategies.base import OptionData, ArbitrageOpportunity, StrategyType


class TestPricingArbitrageStrategy:
    """Test the pricing arbitrage strategy implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logger = MagicMock()
        self.strategy = PricingArbitrageStrategy(self.logger)

    def test_strategy_initialization(self):
        """Test that strategy initializes correctly."""
        assert self.strategy.name == "定价套利策略"
        assert self.strategy.strategy_type == StrategyType.PRICING_ARBITRAGE
        assert self.strategy.logger == self.logger

    def test_create_sample_option_data(self):
        """Test creation of sample option data."""
        # Create basic option data
        option = OptionData(
            symbol="TSLA",
            underlying_symbol="TSLA",
            expiry_date=datetime.now() + timedelta(days=30),
            strike_price=150.0,
            option_type="call",
            market_price=10.0,
            implied_volatility=0.3,
            volume=100,
            open_interest=500
        )
        
        assert option.symbol == "TSLA"
        assert option.strike_price == 150.0
        assert option.market_price == 10.0

    def test_empty_data_handling(self):
        """Test strategy handles empty data gracefully."""
        empty_data = []
        
        # Should not crash with empty data
        try:
            result = self.strategy.find_opportunities(empty_data)
            # Strategy should return empty list or handle gracefully
            assert isinstance(result, list)
        except Exception as e:
            # If it throws an exception, it should be a meaningful one
            assert "empty" in str(e).lower() or "no data" in str(e).lower()

    def test_single_option_data(self):
        """Test strategy with single option (should find no arbitrage)."""
        single_option = [
            OptionData(
                symbol="SPY_Call_400",
                underlying_symbol="SPY", 
                expiry_date=datetime.now() + timedelta(days=15),
                strike_price=400.0,
                option_type="call",
                market_price=5.0,
                implied_volatility=0.25,
                volume=1000,
                open_interest=2000
            )
        ]
        
        # Single option shouldn't create arbitrage opportunities
        result = self.strategy.find_opportunities(single_option)
        assert isinstance(result, list)

    def test_strategy_name_and_type_properties(self):
        """Test strategy name and type are properly set."""
        assert hasattr(self.strategy, 'name')
        assert hasattr(self.strategy, 'strategy_type')
        assert isinstance(self.strategy.name, str)
        assert len(self.strategy.name) > 0

    def test_logger_integration(self):
        """Test that logger is properly integrated."""
        assert self.strategy.logger is not None
        
        # Test that logger is used (indirectly by checking it's available)
        assert hasattr(self.strategy, 'logger')

    def test_strategy_with_mock_data(self):
        """Test strategy with realistic mock data."""
        # Create a set of related options that might have pricing discrepancies
        options = [
            OptionData(
                symbol="AAPL_Call_150",
                underlying_symbol="AAPL",
                expiry_date=datetime.now() + timedelta(days=30),
                strike_price=150.0,
                option_type="call", 
                market_price=8.0,
                implied_volatility=0.25,
                volume=500,
                open_interest=1000
            ),
            OptionData(
                symbol="AAPL_Call_155",
                underlying_symbol="AAPL",
                expiry_date=datetime.now() + timedelta(days=30),
                strike_price=155.0,
                option_type="call",
                market_price=12.0,  # Potentially overpriced
                implied_volatility=0.30,
                volume=300,
                open_interest=800
            ),
            OptionData(
                symbol="AAPL_Put_150", 
                underlying_symbol="AAPL",
                expiry_date=datetime.now() + timedelta(days=30),
                strike_price=150.0,
                option_type="put",
                market_price=7.0,
                implied_volatility=0.28,
                volume=400,
                open_interest=900
            )
        ]
        
        # Test that strategy can process multiple options without crashing
        try:
            result = self.strategy.find_opportunities(options)
            assert isinstance(result, list)
            # Each result should be an ArbitrageOpportunity if any are found
            for opp in result:
                assert hasattr(opp, 'strategy_type') or isinstance(opp, dict)
        except Exception as e:
            # If there's an error, it should be a meaningful trading-related error
            error_msg = str(e).lower()
            # Allow common trading-related errors
            allowed_errors = ['insufficient data', 'calculation error', 'invalid input', 'no opportunities']
            assert any(allowed in error_msg for allowed in allowed_errors), f"Unexpected error: {e}"

    def test_strategy_configuration(self):
        """Test strategy can be configured."""
        # Test that strategy has configurable parameters
        assert hasattr(self.strategy, '__dict__')
        
        # Strategy should have some internal state or configuration
        assert len(vars(self.strategy)) > 0

    def test_strategy_type_enum(self):
        """Test strategy type enum is correctly assigned."""
        from src.strategies.base import StrategyType
        
        # Verify the enum value exists and is properly assigned
        assert hasattr(StrategyType, 'PRICING_ARBITRAGE')
        assert self.strategy.strategy_type == StrategyType.PRICING_ARBITRAGE


if __name__ == "__main__":
    pytest.main([__file__])