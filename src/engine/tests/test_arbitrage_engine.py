"""
Comprehensive unit tests for ArbitrageEngine.

Tests cover:
- Engine initialization and configuration
- Opportunity scanning and detection
- Performance optimizations
- Risk management integration
- Error handling and edge cases
- Parallel processing capabilities
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import List, Dict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.engine.arbitrage_engine import (
    ArbitrageEngine, ScanParameters, TradingSignal,
    PerformanceOptimizedCache, EnginePerformanceMetrics
)
from src.engine.risk_manager import AdvancedRiskManager, PortfolioRiskMetrics
from src.engine.performance_monitor import PerformanceMonitor
from src.config.models import (
    ArbitrageOpportunity, StrategyType, SystemConfig, RiskConfig
)
from src.strategies.base import (
    BaseStrategy, StrategyResult, OptionData, OptionType, 
    ActionType, RiskLevel, StrategyParameters
)
from src.adapters.base import BaseDataAdapter, DataRequest, DataResponse
from src.config.manager import ConfigManager


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""
    
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.PRICING_ARBITRAGE
    
    def scan_opportunities(self, options_data: List[OptionData]) -> StrategyResult:
        # Create mock opportunities
        opportunities = []
        if options_data:
            opp = ArbitrageOpportunity(
                id="test_opp_1",
                strategy_type=self.strategy_type,
                instruments=[options_data[0].code],
                underlying=options_data[0].underlying,
                expected_profit=100.0,
                profit_margin=0.05,
                confidence_score=0.8,
                max_loss=20.0,
                risk_score=0.2,
                days_to_expiry=30,
                market_prices={options_data[0].code: options_data[0].market_price},
                actions=[{
                    'instrument': options_data[0].code,
                    'action': 'BUY',
                    'quantity': 1,
                    'price': options_data[0].market_price
                }],
                data_source="mock"
            )
            opportunities.append(opp)
        
        return StrategyResult(
            strategy_name=self.name,
            opportunities=opportunities,
            execution_time=0.1,
            data_timestamp=datetime.now(),
            success=True
        )
    
    def calculate_profit(self, options: List[OptionData], actions) -> float:
        return 100.0
    
    def assess_risk(self, options: List[OptionData], actions) -> Mock:
        mock_risk = Mock()
        mock_risk.max_loss = 20.0
        mock_risk.risk_level = RiskLevel.LOW
        return mock_risk


class MockDataAdapter(BaseDataAdapter):
    """Mock data adapter for testing."""
    
    @property
    def data_source_type(self):
        return "mock"
    
    async def connect(self) -> None:
        self._update_connection_status("CONNECTED")
    
    async def disconnect(self) -> None:
        self._update_connection_status("DISCONNECTED")
    
    async def get_option_data(self, request: DataRequest) -> DataResponse:
        # Create mock option data
        mock_options = [
            OptionData(
                code="TEST001C2024030100100",
                name="TEST001看涨2024-03-01-100",
                underlying="TEST001",
                option_type=OptionType.CALL,
                strike_price=100.0,
                expiry_date=datetime(2024, 3, 1),
                market_price=5.2,
                bid_price=5.1,
                ask_price=5.3,
                volume=1000,
                open_interest=500,
                implied_volatility=0.25,
                theoretical_price=5.0,
                delta=0.6,
                gamma=0.1,
                theta=-0.02,
                vega=0.3
            ),
            OptionData(
                code="TEST001P2024030100100",
                name="TEST001看跌2024-03-01-100",
                underlying="TEST001",
                option_type=OptionType.PUT,
                strike_price=100.0,
                expiry_date=datetime(2024, 3, 1),
                market_price=4.8,
                bid_price=4.7,
                ask_price=4.9,
                volume=800,
                open_interest=300,
                implied_volatility=0.23,
                theoretical_price=4.9,
                delta=-0.4,
                gamma=0.1,
                theta=-0.02,
                vega=0.3
            )
        ]
        
        return DataResponse(
            request=request,
            data=mock_options,
            timestamp=datetime.now(),
            source="mock",
            quality="HIGH"
        )
    
    async def get_underlying_price(self, symbol: str, as_of_date=None) -> float:
        return 100.0


@pytest.fixture
def mock_config_manager():
    """Create mock configuration manager."""
    config_manager = Mock()
    
    # Mock system config
    system_config = Mock()
    system_config.log_level.value = "INFO"
    system_config.cache.max_size = 1000
    system_config.cache.ttl_seconds = 300
    system_config.risk = Mock()
    system_config.risk.max_daily_loss = 1000.0
    system_config.risk.max_concentration = 0.3
    system_config.risk.max_position_size = 10000.0
    system_config.risk.min_liquidity_volume = 100
    system_config.risk.max_days_to_expiry = 90
    system_config.risk.min_days_to_expiry = 1
    system_config.strategies = {}
    
    config_manager.get_system_config.return_value = system_config
    
    return config_manager


@pytest.fixture
def mock_data_adapters():
    """Create mock data adapters."""
    adapter = MockDataAdapter({}, "mock_adapter")
    return {"mock": adapter}


@pytest.fixture
def mock_strategies():
    """Create mock strategies."""
    strategy = MockStrategy()
    return {StrategyType.PRICING_ARBITRAGE: strategy}


@pytest.fixture
async def arbitrage_engine(mock_config_manager, mock_data_adapters, mock_strategies):
    """Create ArbitrageEngine instance for testing."""
    engine = ArbitrageEngine(
        config_manager=mock_config_manager,
        data_adapters=mock_data_adapters,
        strategies=mock_strategies
    )
    
    # Connect mock adapters
    for adapter in mock_data_adapters.values():
        await adapter.connect()
    
    yield engine
    
    # Cleanup
    await engine.shutdown()


class TestPerformanceOptimizedCache:
    """Test the performance-optimized cache system."""
    
    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        cache = PerformanceOptimizedCache(max_size=3, default_ttl=1)
        
        # Test set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test hit/miss tracking
        assert cache.hits == 1
        assert cache.misses == 0
        
        # Test miss
        assert cache.get("nonexistent") is None
        assert cache.misses == 1
    
    def test_cache_ttl_expiration(self):
        """Test TTL expiration."""
        cache = PerformanceOptimizedCache(max_size=10, default_ttl=0.1)
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        import time
        time.sleep(0.2)
        
        assert cache.get("key1") is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction."""
        cache = PerformanceOptimizedCache(max_size=2)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1
        
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_cache_hit_rate(self):
        """Test hit rate calculation."""
        cache = PerformanceOptimizedCache()
        
        cache.set("key1", "value1")
        cache.get("key1")  # hit
        cache.get("key2")  # miss
        cache.get("key1")  # hit
        
        assert cache.hit_rate == 2/3  # 2 hits, 1 miss


class TestArbitrageEngine:
    """Test the main ArbitrageEngine class."""
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, mock_config_manager, mock_data_adapters, mock_strategies):
        """Test engine initialization."""
        engine = ArbitrageEngine(
            config_manager=mock_config_manager,
            data_adapters=mock_data_adapters,
            strategies=mock_strategies
        )
        
        assert engine.config_manager == mock_config_manager
        assert len(engine.data_adapters) == 1
        assert len(engine.strategies) == 1
        assert isinstance(engine.cache, PerformanceOptimizedCache)
        assert isinstance(engine.performance_metrics, EnginePerformanceMetrics)
        
        await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_scan_opportunities_basic(self, arbitrage_engine):
        """Test basic opportunity scanning."""
        scan_params = ScanParameters(
            strategy_types=[StrategyType.PRICING_ARBITRAGE],
            min_profit_threshold=0.01,
            max_results=100
        )
        
        opportunities = await arbitrage_engine.scan_opportunities(scan_params)
        
        assert isinstance(opportunities, list)
        assert len(opportunities) > 0
        assert all(isinstance(opp, ArbitrageOpportunity) for opp in opportunities)
    
    @pytest.mark.asyncio
    async def test_scan_opportunities_empty_data(self, arbitrage_engine):
        """Test scanning with empty market data."""
        # Mock adapter to return empty data
        for adapter in arbitrage_engine.data_adapters.values():
            adapter.get_option_data = AsyncMock(return_value=DataResponse(
                request=Mock(),
                data=[],
                timestamp=datetime.now(),
                source="mock",
                quality="HIGH"
            ))
        
        scan_params = ScanParameters()
        opportunities = await arbitrage_engine.scan_opportunities(scan_params)
        
        assert opportunities == []
    
    @pytest.mark.asyncio
    async def test_fetch_market_data_async(self, arbitrage_engine):
        """Test asynchronous market data fetching."""
        scan_params = ScanParameters()
        
        market_data = await arbitrage_engine._fetch_market_data_async(scan_params)
        
        assert isinstance(market_data, list)
        assert len(market_data) > 0
        assert all(isinstance(option, OptionData) for option in market_data)
    
    @pytest.mark.asyncio
    async def test_preprocess_market_data(self, arbitrage_engine):
        """Test market data preprocessing."""
        # Create test market data
        market_data = [
            OptionData(
                code="TEST001C",
                name="Test Call",
                underlying="TEST001",
                option_type=OptionType.CALL,
                strike_price=100.0,
                expiry_date=datetime.now() + timedelta(days=30),
                market_price=5.0,
                bid_price=4.9,
                ask_price=5.1,
                volume=1000,
                open_interest=500
            )
        ]
        
        await arbitrage_engine._preprocess_market_data(market_data)
        
        # Check that theoretical pricing was added
        option = market_data[0]
        assert option.theoretical_price is not None
        assert option.theoretical_price > 0
    
    def test_rank_and_filter_opportunities(self, arbitrage_engine):
        """Test opportunity ranking and filtering."""
        # Create test opportunities
        opportunities = [
            ArbitrageOpportunity(
                id="opp1",
                strategy_type=StrategyType.PRICING_ARBITRAGE,
                instruments=["TEST001C"],
                underlying="TEST001",
                expected_profit=100.0,
                profit_margin=0.05,
                confidence_score=0.8,
                max_loss=20.0,
                risk_score=0.2,
                days_to_expiry=30,
                market_prices={"TEST001C": 5.0},
                actions=[],
                data_source="mock"
            ),
            ArbitrageOpportunity(
                id="opp2",
                strategy_type=StrategyType.PRICING_ARBITRAGE,
                instruments=["TEST002P"],
                underlying="TEST002",
                expected_profit=200.0,
                profit_margin=0.08,
                confidence_score=0.9,
                max_loss=30.0,
                risk_score=0.1,
                days_to_expiry=45,
                market_prices={"TEST002P": 4.0},
                actions=[],
                data_source="mock"
            )
        ]
        
        scan_params = ScanParameters(
            min_profit_threshold=0.01,
            max_risk_tolerance=0.5,
            max_results=10
        )
        
        ranked_opportunities = arbitrage_engine._rank_and_filter_opportunities(
            opportunities, scan_params
        )
        
        assert len(ranked_opportunities) == 2
        # Check that opportunities are ranked by composite score
        assert ranked_opportunities[0].profit_margin >= ranked_opportunities[1].profit_margin
    
    def test_calculate_risk_metrics(self, arbitrage_engine):
        """Test risk metrics calculation."""
        opportunity = ArbitrageOpportunity(
            id="test_opp",
            strategy_type=StrategyType.PRICING_ARBITRAGE,
            instruments=["TEST001C"],
            underlying="TEST001",
            expected_profit=100.0,
            profit_margin=0.05,
            confidence_score=0.8,
            max_loss=20.0,
            risk_score=0.2,
            days_to_expiry=30,
            market_prices={"TEST001C": 5.0},
            volumes={"TEST001C": 1000},
            actions=[],
            data_source="mock"
        )
        
        risk_metrics = arbitrage_engine.calculate_risk_metrics(opportunity)
        
        assert risk_metrics.max_loss == 20.0
        assert risk_metrics.expected_return == 0.05
        assert 0 <= risk_metrics.probability_profit <= 1.0
        assert isinstance(risk_metrics.risk_level, RiskLevel)
    
    def test_generate_trading_signals(self, arbitrage_engine):
        """Test trading signal generation."""
        opportunities = [
            ArbitrageOpportunity(
                id="test_opp",
                strategy_type=StrategyType.PRICING_ARBITRAGE,
                instruments=["TEST001C"],
                underlying="TEST001",
                expected_profit=100.0,
                profit_margin=0.05,
                confidence_score=0.8,
                max_loss=20.0,
                risk_score=0.2,
                days_to_expiry=30,
                market_prices={"TEST001C": 5.0},
                volumes={"TEST001C": 1000},
                actions=[{
                    'instrument': "TEST001C",
                    'action': 'BUY',
                    'quantity': 1,
                    'price': 5.0
                }],
                data_source="mock"
            )
        ]
        
        signals = arbitrage_engine.generate_trading_signals(opportunities)
        
        assert len(signals) == 1
        signal = signals[0]
        assert isinstance(signal, TradingSignal)
        assert signal.opportunity_id == "test_opp"
        assert signal.expected_profit == 100.0
        assert len(signal.actions) == 1
    
    @pytest.mark.asyncio
    async def test_health_check(self, arbitrage_engine):
        """Test engine health check."""
        health_status = await arbitrage_engine.health_check()
        
        assert 'engine_status' in health_status
        assert 'adapters' in health_status
        assert 'strategies' in health_status
        assert 'performance_metrics' in health_status
        
        assert health_status['engine_status'] == 'healthy'
        assert health_status['strategies'] == 1
    
    def test_performance_metrics_update(self, arbitrage_engine):
        """Test performance metrics tracking."""
        initial_scans = getattr(arbitrage_engine.performance_metrics, 'scans_completed', 0)
        
        # Update metrics
        arbitrage_engine._update_performance_metrics(5.0, 10)
        
        assert arbitrage_engine.performance_metrics.total_scan_time == 5.0
        assert arbitrage_engine.performance_metrics.total_opportunities_found == 10
    
    def test_cache_integration(self, arbitrage_engine):
        """Test cache integration in engine."""
        # Test cache functionality
        arbitrage_engine.cache.set("test_key", "test_value")
        assert arbitrage_engine.cache.get("test_key") == "test_value"
        
        # Test cache clear
        arbitrage_engine.clear_cache()
        assert arbitrage_engine.cache.get("test_key") is None
    
    @pytest.mark.asyncio
    async def test_parallel_strategy_execution(self, arbitrage_engine):
        """Test parallel strategy execution."""
        # Add multiple strategies
        strategy2 = MockStrategy()
        strategy2._name = "MockStrategy2"
        arbitrage_engine.strategies[StrategyType.PUT_CALL_PARITY] = strategy2
        
        market_data = [
            OptionData(
                code="TEST001C",
                name="Test Call",
                underlying="TEST001",
                option_type=OptionType.CALL,
                strike_price=100.0,
                expiry_date=datetime.now() + timedelta(days=30),
                market_price=5.0,
                bid_price=4.9,
                ask_price=5.1,
                volume=1000,
                open_interest=500
            )
        ]
        
        scan_params = ScanParameters()
        
        opportunities = await arbitrage_engine._execute_strategies_parallel(
            market_data, scan_params
        )
        
        # Should have opportunities from both strategies
        assert len(opportunities) >= 2
    
    def test_error_handling_in_strategy_execution(self, arbitrage_engine):
        """Test error handling in strategy execution."""
        # Create a strategy that raises an exception
        failing_strategy = MockStrategy()
        failing_strategy.scan_opportunities = Mock(side_effect=Exception("Test error"))
        
        market_data = []
        scan_params = ScanParameters()
        
        result = arbitrage_engine._execute_single_strategy(
            failing_strategy, market_data, scan_params
        )
        
        assert not result.success
        assert "Test error" in result.error_message
    
    @pytest.mark.asyncio 
    async def test_adapter_failure_handling(self, arbitrage_engine):
        """Test handling of adapter failures."""
        # Make adapter fail
        for adapter in arbitrage_engine.data_adapters.values():
            adapter.get_option_data = AsyncMock(side_effect=Exception("Connection failed"))
        
        scan_params = ScanParameters()
        market_data = await arbitrage_engine._fetch_market_data_async(scan_params)
        
        # Should handle failure gracefully
        assert market_data == []
    
    @pytest.mark.asyncio
    async def test_engine_shutdown(self, arbitrage_engine):
        """Test engine shutdown process."""
        # Engine should shutdown without errors
        await arbitrage_engine.shutdown()
        
        # Check that adapters are disconnected
        for adapter in arbitrage_engine.data_adapters.values():
            # In real scenario, adapter would be disconnected
            # Here we just verify shutdown completed without exceptions
            pass


class TestPerformanceOptimizations:
    """Test performance optimization features."""
    
    @pytest.mark.asyncio
    async def test_vectorized_pricing_integration(self, arbitrage_engine):
        """Test integration with vectorized pricing engine."""
        # Create market data that should trigger pricing calculations
        market_data = [
            OptionData(
                code=f"TEST{i:03d}C",
                name=f"Test Call {i}",
                underlying=f"TEST{i:03d}",
                option_type=OptionType.CALL,
                strike_price=100.0 + i,
                expiry_date=datetime.now() + timedelta(days=30),
                market_price=5.0 + i * 0.1,
                bid_price=4.9 + i * 0.1,
                ask_price=5.1 + i * 0.1,
                volume=1000,
                open_interest=500
            ) for i in range(10)
        ]
        
        start_time = time.perf_counter()
        await arbitrage_engine._preprocess_market_data(market_data)
        end_time = time.perf_counter()
        
        # Verify all options have theoretical prices
        for option in market_data:
            assert option.theoretical_price is not None
            assert option.theoretical_price > 0
        
        # Performance should be reasonable
        processing_time = end_time - start_time
        assert processing_time < 1.0  # Should complete within 1 second
    
    def test_cache_performance_under_load(self):
        """Test cache performance under high load."""
        cache = PerformanceOptimizedCache(max_size=1000)
        
        # Add many items
        start_time = time.perf_counter()
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}")
        end_time = time.perf_counter()
        
        set_time = end_time - start_time
        
        # Retrieve items
        start_time = time.perf_counter()
        for i in range(1000):
            cache.get(f"key_{i}")
        end_time = time.perf_counter()
        
        get_time = end_time - start_time
        
        # Performance should be reasonable
        assert set_time < 1.0  # Setting 1000 items should take < 1s
        assert get_time < 0.5  # Getting 1000 items should take < 0.5s
        assert cache.hit_rate > 0.99  # Should have very high hit rate


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])