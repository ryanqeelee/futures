"""
Test package for ArbitrageEngine.

This package contains comprehensive test suites for:
- ArbitrageEngine core functionality
- AdvancedRiskManager components
- PerformanceMonitor features
- Integration testing
- Performance benchmarking

Run all tests with:
    pytest src/engine/tests/ -v

Run specific test categories:
    pytest src/engine/tests/test_arbitrage_engine.py -v
    pytest src/engine/tests/test_risk_manager.py -v
"""

# Test configuration
TEST_DATA_DIR = "test_data"
MOCK_CONFIG_FILE = "test_config.json"

# Performance test thresholds
PERFORMANCE_THRESHOLDS = {
    'max_scan_time': 5.0,        # seconds
    'max_memory_usage': 1.0,     # GB
    'min_cache_hit_rate': 0.7,   # 70%
    'max_error_rate': 0.01,      # 1%
}

# Test data generators
def generate_mock_options(count=10):
    """Generate mock option data for testing."""
    from datetime import datetime, timedelta
    from src.strategies.base import OptionData, OptionType
    
    options = []
    for i in range(count):
        option = OptionData(
            code=f"TEST{i:03d}{'C' if i % 2 == 0 else 'P'}",
            name=f"Test Option {i}",
            underlying=f"TEST{i//2:03d}",
            option_type=OptionType.CALL if i % 2 == 0 else OptionType.PUT,
            strike_price=100.0 + i * 5,
            expiry_date=datetime.now() + timedelta(days=30 + i),
            market_price=5.0 + i * 0.5,
            bid_price=4.9 + i * 0.5,
            ask_price=5.1 + i * 0.5,
            volume=1000 + i * 100,
            open_interest=500 + i * 50,
            implied_volatility=0.25 + i * 0.01,
            theoretical_price=4.95 + i * 0.5
        )
        options.append(option)
    
    return options

def generate_mock_opportunities(count=5):
    """Generate mock arbitrage opportunities for testing."""
    from datetime import datetime
    from src.config.models import ArbitrageOpportunity, StrategyType
    
    opportunities = []
    for i in range(count):
        opp = ArbitrageOpportunity(
            id=f"mock_opp_{i}",
            strategy_type=StrategyType.PRICING_ARBITRAGE,
            instruments=[f"TEST{i}C", f"TEST{i}P"],
            underlying=f"TEST{i}",
            expected_profit=100.0 + i * 50,
            profit_margin=0.05 + i * 0.01,
            confidence_score=0.7 + i * 0.05,
            max_loss=20.0 + i * 5,
            risk_score=0.1 + i * 0.05,
            days_to_expiry=30 + i * 5,
            market_prices={f"TEST{i}C": 5.0, f"TEST{i}P": 4.0},
            volumes={f"TEST{i}C": 1000, f"TEST{i}P": 800},
            actions=[],
            data_source="mock"
        )
        opportunities.append(opp)
    
    return opportunities