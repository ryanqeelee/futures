"""
Unit tests for the Advanced Risk Manager.

Tests cover:
- VaR calculations (Historical, Parametric, Monte Carlo)
- Maximum drawdown analysis
- Sharpe ratio calculations
- Portfolio risk assessment
- Anomaly detection
- Risk limit validation
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.engine.risk_manager import (
    AdvancedRiskManager, VaRResult, DrawdownResult, SharpeRatioResult,
    PortfolioRiskMetrics
)
from src.config.models import ArbitrageOpportunity, StrategyType, RiskConfig
from src.strategies.base import RiskLevel


@pytest.fixture
def risk_config():
    """Create a test risk configuration."""
    return RiskConfig(
        max_position_size=10000.0,
        max_daily_loss=1000.0,
        min_liquidity_volume=100,
        max_concentration=0.3,
        max_days_to_expiry=90,
        min_days_to_expiry=1
    )


@pytest.fixture
def risk_manager(risk_config):
    """Create an AdvancedRiskManager instance."""
    return AdvancedRiskManager(risk_config)


@pytest.fixture
def sample_returns():
    """Create sample return data for testing."""
    np.random.seed(42)  # For reproducible results
    returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
    return returns


@pytest.fixture
def sample_opportunities():
    """Create sample arbitrage opportunities."""
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
            volumes={"TEST001C": 1000},
            actions=[],
            data_source="mock"
        ),
        ArbitrageOpportunity(
            id="opp2",
            strategy_type=StrategyType.PUT_CALL_PARITY,
            instruments=["TEST002P"],
            underlying="TEST002",
            expected_profit=150.0,
            profit_margin=0.07,
            confidence_score=0.9,
            max_loss=25.0,
            risk_score=0.15,
            days_to_expiry=45,
            market_prices={"TEST002P": 4.0},
            volumes={"TEST002P": 800},
            actions=[],
            data_source="mock"
        ),
        ArbitrageOpportunity(
            id="opp3",
            strategy_type=StrategyType.PRICING_ARBITRAGE,
            instruments=["TEST001P"],
            underlying="TEST001", # Same underlying as opp1
            expected_profit=80.0,
            profit_margin=0.04,
            confidence_score=0.7,
            max_loss=15.0,
            risk_score=0.25,
            days_to_expiry=20,
            market_prices={"TEST001P": 3.5},
            volumes={"TEST001P": 500},
            actions=[],
            data_source="mock"
        )
    ]
    return opportunities


class TestVaRCalculations:
    """Test Value at Risk calculations."""
    
    def test_historical_var_basic(self, risk_manager, sample_returns):
        """Test basic historical VaR calculation."""
        var_result = risk_manager.calculate_var(
            sample_returns, 
            confidence_level=0.95, 
            method='historical'
        )
        
        assert isinstance(var_result, VaRResult)
        assert var_result.confidence_level == 0.95
        assert var_result.method == 'historical'
        assert var_result.var_1_day > 0
        assert var_result.var_5_day > var_result.var_1_day  # Multi-day VaR should be higher
        assert var_result.var_10_day > var_result.var_5_day
    
    def test_parametric_var_basic(self, risk_manager, sample_returns):
        """Test parametric VaR calculation."""
        var_result = risk_manager.calculate_var(
            sample_returns,
            confidence_level=0.99,
            method='parametric'
        )
        
        assert isinstance(var_result, VaRResult)
        assert var_result.confidence_level == 0.99
        assert var_result.method == 'parametric'
        assert var_result.var_1_day > 0
    
    def test_monte_carlo_var_basic(self, risk_manager, sample_returns):
        """Test Monte Carlo VaR calculation."""
        var_result = risk_manager.calculate_var(
            sample_returns,
            confidence_level=0.95,
            method='monte_carlo'
        )
        
        assert isinstance(var_result, VaRResult)
        assert var_result.confidence_level == 0.95
        assert var_result.method == 'monte_carlo'
        assert var_result.var_1_day > 0
    
    def test_var_empty_returns(self, risk_manager):
        """Test VaR calculation with empty returns."""
        empty_returns = np.array([])
        
        var_result = risk_manager.calculate_var(empty_returns)
        
        assert var_result.var_1_day == 0
        assert var_result.var_5_day == 0
        assert var_result.var_10_day == 0
    
    def test_var_confidence_levels(self, risk_manager, sample_returns):
        """Test VaR at different confidence levels."""
        var_95 = risk_manager.calculate_var(sample_returns, confidence_level=0.95)
        var_99 = risk_manager.calculate_var(sample_returns, confidence_level=0.99)
        
        # Higher confidence level should give higher VaR
        assert var_99.var_1_day >= var_95.var_1_day
    
    def test_historical_var_implementation(self, risk_manager):
        """Test historical VaR implementation details."""
        # Create known distribution
        returns = np.array([-0.05, -0.03, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
        
        var_90 = risk_manager._historical_var(returns, 0.90)
        
        # With 90% confidence, we expect to capture the worst 10% (1 out of 10 values)
        # The worst value is -0.05, so VaR should be 0.05
        assert abs(var_90 - 0.05) < 0.001
    
    def test_parametric_var_implementation(self, risk_manager):
        """Test parametric VaR implementation."""
        # Create returns with known mean and std
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)  # mean=0.1%, std=2%
        
        var_result = risk_manager._parametric_var(returns, 0.95)
        
        # For normal distribution, 95% VaR ≈ mean - 1.645 * std
        expected_var = abs(0.001 - 1.645 * 0.02)  # Approximately 0.032
        
        # Allow some tolerance due to sampling variation
        assert abs(var_result - expected_var) < 0.01


class TestDrawdownAnalysis:
    """Test maximum drawdown analysis."""
    
    def test_drawdown_basic(self, risk_manager):
        """Test basic drawdown calculation."""
        # Create price series with known drawdown
        prices = np.array([100, 110, 120, 90, 80, 100, 110])
        
        drawdown_result = risk_manager.calculate_maximum_drawdown(prices)
        
        assert isinstance(drawdown_result, DrawdownResult)
        assert drawdown_result.max_drawdown > 0
        assert drawdown_result.max_drawdown_pct > 0
        assert drawdown_result.recovery_date is not None  # Should recover to peak
    
    def test_drawdown_no_recovery(self, risk_manager):
        """Test drawdown with no recovery."""
        # Price series that never recovers
        prices = np.array([100, 110, 120, 90, 80, 70])
        
        drawdown_result = risk_manager.calculate_maximum_drawdown(prices)
        
        assert drawdown_result.max_drawdown > 0
        assert drawdown_result.recovery_date is None  # No recovery
        assert drawdown_result.current_drawdown > 0  # Still in drawdown
    
    def test_drawdown_empty_prices(self, risk_manager):
        """Test drawdown with empty price series."""
        empty_prices = np.array([])
        
        drawdown_result = risk_manager.calculate_maximum_drawdown(empty_prices)
        
        assert drawdown_result.max_drawdown == 0
        assert drawdown_result.max_drawdown_pct == 0
    
    def test_drawdown_monotonic_increase(self, risk_manager):
        """Test drawdown with monotonically increasing prices."""
        prices = np.array([100, 105, 110, 115, 120])
        
        drawdown_result = risk_manager.calculate_maximum_drawdown(prices)
        
        # Should have minimal or zero drawdown
        assert drawdown_result.max_drawdown_pct <= 0.01  # Very small drawdown


class TestSharpeRatio:
    """Test Sharpe ratio calculations."""
    
    def test_sharpe_ratio_basic(self, risk_manager, sample_returns):
        """Test basic Sharpe ratio calculation."""
        sharpe_result = risk_manager.calculate_sharpe_ratio(sample_returns)
        
        assert isinstance(sharpe_result, SharpeRatioResult)
        assert sharpe_result.period_days == len(sample_returns)
        assert sharpe_result.risk_free_rate == 0.02  # Default
        assert isinstance(sharpe_result.sharpe_ratio, float)
    
    def test_sharpe_ratio_positive_returns(self, risk_manager):
        """Test Sharpe ratio with consistently positive returns."""
        # Create positive returns
        positive_returns = np.full(252, 0.001)  # 0.1% daily return
        
        sharpe_result = risk_manager.calculate_sharpe_ratio(positive_returns)
        
        # Should have positive Sharpe ratio
        assert sharpe_result.sharpe_ratio > 0
        assert sharpe_result.annualized_return > 0
        assert sharpe_result.annualized_volatility >= 0
    
    def test_sharpe_ratio_empty_returns(self, risk_manager):
        """Test Sharpe ratio with empty returns."""
        empty_returns = np.array([])
        
        sharpe_result = risk_manager.calculate_sharpe_ratio(empty_returns)
        
        assert sharpe_result.sharpe_ratio == 0
        assert sharpe_result.annualized_return == 0
        assert sharpe_result.annualized_volatility == 0
    
    def test_sharpe_ratio_custom_risk_free_rate(self, risk_manager, sample_returns):
        """Test Sharpe ratio with custom risk-free rate."""
        rf_rate = 0.05  # 5% risk-free rate
        
        sharpe_result = risk_manager.calculate_sharpe_ratio(sample_returns, rf_rate)
        
        assert sharpe_result.risk_free_rate == rf_rate


class TestAnomalyDetection:
    """Test anomaly detection using Z-score."""
    
    def test_zscore_anomaly_detection_basic(self, risk_manager):
        """Test basic Z-score anomaly detection."""
        # Create data with clear outliers
        values = np.array([1, 2, 3, 2, 1, 100, 2, 3, 1])  # 100 is an outlier
        
        anomalies = risk_manager.detect_anomalies_zscore(values, threshold=2.0)
        
        assert len(anomalies) >= 1
        # The outlier (100) should be detected
        assert any(values[idx] == 100 for idx, z_score in anomalies)
    
    def test_zscore_no_anomalies(self, risk_manager):
        """Test Z-score with no anomalies."""
        # Create normal data
        np.random.seed(42)
        values = np.random.normal(0, 1, 100)
        
        anomalies = risk_manager.detect_anomalies_zscore(values, threshold=3.0)
        
        # With threshold=3.0, should have very few or no anomalies
        assert len(anomalies) <= 5  # Very few false positives expected
    
    def test_zscore_empty_data(self, risk_manager):
        """Test Z-score with empty data."""
        empty_values = np.array([])
        
        anomalies = risk_manager.detect_anomalies_zscore(empty_values)
        
        assert len(anomalies) == 0
    
    def test_zscore_identical_values(self, risk_manager):
        """Test Z-score with identical values."""
        identical_values = np.full(10, 5.0)  # All values are 5.0
        
        anomalies = risk_manager.detect_anomalies_zscore(identical_values)
        
        assert len(anomalies) == 0  # No anomalies in identical values


class TestPortfolioRiskAssessment:
    """Test comprehensive portfolio risk assessment."""
    
    def test_portfolio_risk_basic(self, risk_manager, sample_opportunities, sample_returns):
        """Test basic portfolio risk assessment."""
        portfolio_metrics = risk_manager.assess_portfolio_risk(
            sample_opportunities,
            sample_returns,
            current_portfolio_value=100000.0
        )
        
        assert isinstance(portfolio_metrics, PortfolioRiskMetrics)
        assert portfolio_metrics.total_exposure > 0
        assert 0 <= portfolio_metrics.concentration_risk <= 1
        assert 0 <= portfolio_metrics.correlation_risk <= 1
        assert 0 <= portfolio_metrics.liquidity_risk <= 1
        assert 0 <= portfolio_metrics.time_decay_risk <= 1
        assert 0 <= portfolio_metrics.overall_risk_score <= 1
        assert isinstance(portfolio_metrics.risk_level, RiskLevel)
    
    def test_concentration_risk_calculation(self, risk_manager, sample_opportunities):
        """Test concentration risk calculation."""
        concentration_risk = risk_manager._calculate_concentration_risk(sample_opportunities)
        
        assert 0 <= concentration_risk <= 1
        # With multiple underlyings, concentration should be moderate
        assert concentration_risk < 1.0
    
    def test_correlation_risk_calculation(self, risk_manager, sample_opportunities):
        """Test correlation risk calculation."""
        correlation_risk = risk_manager._calculate_correlation_risk(sample_opportunities)
        
        assert 0 <= correlation_risk <= 1
        # With multiple strategy types, correlation risk should be moderate
        assert correlation_risk < 1.0
    
    def test_liquidity_risk_calculation(self, risk_manager, sample_opportunities):
        """Test liquidity risk calculation."""
        liquidity_risk = risk_manager._calculate_liquidity_risk(sample_opportunities)
        
        assert 0 <= liquidity_risk <= 1
        # All sample opportunities have reasonable volume
        assert liquidity_risk < 0.8
    
    def test_time_decay_risk_calculation(self, risk_manager, sample_opportunities):
        """Test time decay risk calculation."""
        time_risk = risk_manager._calculate_time_decay_risk(sample_opportunities)
        
        assert 0 <= time_risk <= 1
        # Mix of expiry dates should give moderate time risk
        assert time_risk < 1.0
    
    def test_portfolio_risk_empty_opportunities(self, risk_manager):
        """Test portfolio risk with empty opportunities."""
        portfolio_metrics = risk_manager.assess_portfolio_risk([])
        
        assert portfolio_metrics.total_exposure == 0
        assert portfolio_metrics.risk_level == RiskLevel.CRITICAL


class TestPositionSizing:
    """Test position sizing calculations."""
    
    def test_position_sizing_basic(self, risk_manager):
        """Test basic position sizing."""
        opportunity = ArbitrageOpportunity(
            id="test_opp",
            strategy_type=StrategyType.PRICING_ARBITRAGE,
            instruments=["TEST001C"],
            underlying="TEST001",
            expected_profit=100.0,
            profit_margin=0.05,
            confidence_score=0.8,
            max_loss=20.0,
            risk_score=0.02,  # 2% risk
            days_to_expiry=30,
            market_prices={"TEST001C": 5.0},
            volumes={"TEST001C": 1000},
            actions=[],
            data_source="mock"
        )
        
        position_size = risk_manager.calculate_position_size(
            opportunity,
            portfolio_value=100000.0,
            risk_budget=0.02
        )
        
        assert position_size > 0
        assert position_size <= 100000.0  # Should not exceed portfolio value
        assert position_size <= risk_manager.config.max_position_size
    
    def test_position_sizing_zero_risk(self, risk_manager):
        """Test position sizing with zero risk."""
        opportunity = ArbitrageOpportunity(
            id="test_opp",
            strategy_type=StrategyType.PRICING_ARBITRAGE,
            instruments=["TEST001C"],
            underlying="TEST001",
            expected_profit=100.0,
            profit_margin=0.05,
            confidence_score=0.8,
            max_loss=0.0,
            risk_score=0.0,  # No risk
            days_to_expiry=30,
            market_prices={"TEST001C": 5.0},
            volumes={"TEST001C": 1000},
            actions=[],
            data_source="mock"
        )
        
        position_size = risk_manager.calculate_position_size(opportunity, 100000.0)
        
        assert position_size == 0  # Should not allocate to zero-risk scenario
    
    def test_position_sizing_constraints(self, risk_manager):
        """Test position sizing respects all constraints."""
        opportunity = ArbitrageOpportunity(
            id="test_opp",
            strategy_type=StrategyType.PRICING_ARBITRAGE,
            instruments=["TEST001C"],
            underlying="TEST001",
            expected_profit=100.0,
            profit_margin=0.1,  # High profit
            confidence_score=0.95,  # High confidence
            max_loss=10.0,
            risk_score=0.01,  # Low risk
            days_to_expiry=30,
            market_prices={"TEST001C": 5.0},
            volumes={"TEST001C": 1000},
            actions=[],
            data_source="mock"
        )
        
        position_size = risk_manager.calculate_position_size(
            opportunity,
            portfolio_value=100000.0,
            risk_budget=0.01  # 1% risk budget
        )
        
        # Should respect configuration constraints
        assert position_size <= risk_manager.config.max_position_size
        assert position_size <= 100000.0 * risk_manager.config.max_concentration


class TestRiskLimitValidation:
    """Test risk limit validation."""
    
    def test_validate_risk_limits_pass(self, risk_manager, sample_opportunities):
        """Test risk limit validation that should pass."""
        is_valid, violations = risk_manager.validate_risk_limits(sample_opportunities)
        
        assert is_valid
        assert len(violations) == 0
    
    def test_validate_risk_limits_max_loss_violation(self, risk_manager):
        """Test risk limit validation with max loss violation."""
        # Create opportunity that exceeds max daily loss
        high_loss_opp = ArbitrageOpportunity(
            id="high_loss",
            strategy_type=StrategyType.PRICING_ARBITRAGE,
            instruments=["TEST001C"],
            underlying="TEST001",
            expected_profit=100.0,
            profit_margin=0.05,
            confidence_score=0.8,
            max_loss=2000.0,  # Exceeds max_daily_loss of 1000
            risk_score=0.2,
            days_to_expiry=30,
            market_prices={"TEST001C": 5.0},
            volumes={"TEST001C": 1000},
            actions=[],
            data_source="mock"
        )
        
        is_valid, violations = risk_manager.validate_risk_limits([high_loss_opp])
        
        assert not is_valid
        assert len(violations) > 0
        assert any("max daily loss" in v.lower() for v in violations)
    
    def test_validate_risk_limits_expiry_violation(self, risk_manager):
        """Test risk limit validation with expiry violation."""
        # Create opportunity with invalid expiry
        invalid_expiry_opp = ArbitrageOpportunity(
            id="invalid_expiry",
            strategy_type=StrategyType.PRICING_ARBITRAGE,
            instruments=["TEST001C"],
            underlying="TEST001",
            expected_profit=100.0,
            profit_margin=0.05,
            confidence_score=0.8,
            max_loss=20.0,
            risk_score=0.2,
            days_to_expiry=120,  # Exceeds max_days_to_expiry of 90
            market_prices={"TEST001C": 5.0},
            volumes={"TEST001C": 1000},
            actions=[],
            data_source="mock"
        )
        
        is_valid, violations = risk_manager.validate_risk_limits([invalid_expiry_opp])
        
        assert not is_valid
        assert len(violations) > 0
        assert any("expiry" in v.lower() for v in violations)
    
    def test_validate_risk_limits_liquidity_violation(self, risk_manager):
        """Test risk limit validation with liquidity violation."""
        # Create opportunity with insufficient liquidity
        low_liquidity_opp = ArbitrageOpportunity(
            id="low_liquidity",
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
            volumes={"TEST001C": 50},  # Below min_liquidity_volume of 100
            actions=[],
            data_source="mock"
        )
        
        is_valid, violations = risk_manager.validate_risk_limits([low_liquidity_opp])
        
        assert not is_valid
        assert len(violations) > 0
        assert any("liquidity" in v.lower() for v in violations)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])