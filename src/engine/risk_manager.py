"""
Advanced Risk Management System for Arbitrage Engine.

This module implements comprehensive risk management with:
- VaR (Value at Risk) calculations
- Maximum drawdown analysis
- Sharpe ratio computations
- Z-score anomaly detection
- Dynamic position sizing
- Portfolio-level risk assessment
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
from collections import deque
import logging

from ..config.models import ArbitrageOpportunity, RiskConfig
from ..strategies.base import RiskMetrics, RiskLevel, OptionData

# Import unified exception framework
from src.core.exceptions import (
    TradingSystemError, RiskError, SystemError,
    error_handler, create_error_context
)


@dataclass
class VaRResult:
    """Value at Risk calculation result."""
    var_1_day: float
    var_5_day: float
    var_10_day: float
    confidence_level: float
    method: str
    timestamp: datetime


@dataclass
class DrawdownResult:
    """Maximum drawdown analysis result."""
    max_drawdown: float
    max_drawdown_pct: float
    peak_date: datetime
    trough_date: datetime
    recovery_date: Optional[datetime]
    current_drawdown: float


@dataclass
class SharpeRatioResult:
    """Sharpe ratio calculation result."""
    sharpe_ratio: float
    annualized_return: float
    annualized_volatility: float
    risk_free_rate: float
    period_days: int


@dataclass
class PortfolioRiskMetrics:
    """Comprehensive portfolio risk metrics."""
    total_exposure: float
    concentration_risk: float
    correlation_risk: float
    liquidity_risk: float
    time_decay_risk: float
    var_metrics: VaRResult
    drawdown_metrics: DrawdownResult
    sharpe_metrics: Optional[SharpeRatioResult]
    overall_risk_score: float
    risk_level: RiskLevel


class AdvancedRiskManager:
    """
    Advanced risk management system with comprehensive analytics.
    
    Features:
    - Multiple VaR calculation methods (Historical, Parametric, Monte Carlo)
    - Dynamic risk thresholds based on market conditions
    - Portfolio correlation analysis
    - Real-time risk monitoring
    - Stress testing capabilities
    """
    
    def __init__(self, risk_config: RiskConfig):
        """
        Initialize the risk manager.
        
        Args:
            risk_config: Risk management configuration
        """
        self.config = risk_config
        self.logger = logging.getLogger(__name__)
        
        # Risk calculation parameters
        self.confidence_levels = [0.95, 0.99]
        self.var_methods = ['historical', 'parametric']
        self.lookback_days = 252  # 1 year for historical analysis
        
        # Historical data storage (for real-time risk calculation)
        self.price_history = deque(maxlen=self.lookback_days)
        self.pnl_history = deque(maxlen=self.lookback_days)
        
        # Risk thresholds (dynamic)
        self.base_var_threshold = 0.02  # 2%
        self.max_concentration = self.config.max_concentration
        self.max_position_size = self.config.max_position_size
    
    def calculate_var(
        self, 
        returns: np.ndarray, 
        confidence_level: float = 0.95,
        method: str = 'historical',
        holding_period: int = 1
    ) -> VaRResult:
        """
        Calculate Value at Risk using multiple methods.
        
        Args:
            returns: Array of historical returns
            confidence_level: Confidence level (0.95 or 0.99)
            method: Calculation method ('historical', 'parametric', 'monte_carlo')
            holding_period: Holding period in days
            
        Returns:
            VaR calculation results
        """
        if len(returns) == 0:
            return VaRResult(0, 0, 0, confidence_level, method, datetime.now())
        
        try:
            if method == 'historical':
                var_1d = self._historical_var(returns, confidence_level)
            elif method == 'parametric':
                var_1d = self._parametric_var(returns, confidence_level)
            elif method == 'monte_carlo':
                var_1d = self._monte_carlo_var(returns, confidence_level)
            else:
                raise ValueError(f"Unknown VaR method: {method}")
            
            # Scale VaR for different holding periods
            var_5d = var_1d * np.sqrt(5)
            var_10d = var_1d * np.sqrt(10)
            
            return VaRResult(
                var_1_day=var_1d,
                var_5_day=var_5d,
                var_10_day=var_10d,
                confidence_level=confidence_level,
                method=method,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            context = create_error_context(
                component="risk_manager",
                operation="calculate_var",
                method=method,
                confidence_level=confidence_level,
                returns_length=len(returns)
            )
            error_msg = f"VaR calculation failed: {e}"
            self.logger.error(error_msg)
            raise RiskError(error_msg, "var_calculation", context) from e
    
    def _historical_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate historical VaR."""
        if len(returns) == 0:
            return 0.0
        
        # Sort returns in ascending order
        sorted_returns = np.sort(returns)
        
        # Find the percentile
        index = int((1 - confidence_level) * len(sorted_returns))
        
        return abs(sorted_returns[index]) if index < len(sorted_returns) else 0.0
    
    def _parametric_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate parametric VaR assuming normal distribution."""
        if len(returns) == 0:
            return 0.0
        
        # Calculate mean and standard deviation
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # Get z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # Calculate VaR
        var = abs(mean_return + z_score * std_return)
        
        return var
    
    def _monte_carlo_var(
        self, 
        returns: np.ndarray, 
        confidence_level: float,
        num_simulations: int = 10000
    ) -> float:
        """Calculate Monte Carlo VaR."""
        if len(returns) == 0:
            return 0.0
        
        # Estimate parameters from historical data
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # Generate random scenarios
        simulated_returns = np.random.normal(mean_return, std_return, num_simulations)
        
        # Calculate VaR from simulations
        sorted_returns = np.sort(simulated_returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        
        return abs(sorted_returns[index]) if index < len(sorted_returns) else 0.0
    
    def calculate_maximum_drawdown(self, prices: np.ndarray) -> DrawdownResult:
        """
        Calculate maximum drawdown from price series.
        
        Args:
            prices: Array of price values
            
        Returns:
            Maximum drawdown analysis
        """
        if len(prices) == 0:
            return DrawdownResult(0, 0, datetime.now(), datetime.now(), None, 0)
        
        try:
            # Calculate running maximum (peak)
            peaks = np.maximum.accumulate(prices)
            
            # Calculate drawdown
            drawdowns = (prices - peaks) / peaks
            
            # Find maximum drawdown
            max_dd_idx = np.argmin(drawdowns)
            max_drawdown = abs(drawdowns[max_dd_idx])
            
            # Find peak before maximum drawdown
            peak_idx = np.argmax(peaks[:max_dd_idx + 1])
            
            # Find recovery point (first point after max DD where price >= peak)
            recovery_idx = None
            for i in range(max_dd_idx + 1, len(prices)):
                if prices[i] >= peaks[peak_idx]:
                    recovery_idx = i
                    break
            
            # Current drawdown
            current_drawdown = abs(drawdowns[-1]) if len(drawdowns) > 0 else 0
            
            return DrawdownResult(
                max_drawdown=max_drawdown * peaks[peak_idx],
                max_drawdown_pct=max_drawdown,
                peak_date=datetime.now() - timedelta(days=len(prices) - peak_idx),
                trough_date=datetime.now() - timedelta(days=len(prices) - max_dd_idx),
                recovery_date=(datetime.now() - timedelta(days=len(prices) - recovery_idx) 
                             if recovery_idx else None),
                current_drawdown=current_drawdown
            )
            
        except Exception as e:
            context = create_error_context(
                component="risk_manager",
                operation="calculate_maximum_drawdown",
                prices_length=len(prices)
            )
            error_msg = f"Drawdown calculation failed: {e}"
            self.logger.error(error_msg)
            raise RiskError(error_msg, "drawdown_calculation", context) from e
    
    def calculate_sharpe_ratio(
        self, 
        returns: np.ndarray, 
        risk_free_rate: float = 0.02
    ) -> SharpeRatioResult:
        """
        Calculate annualized Sharpe ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Sharpe ratio analysis
        """
        if len(returns) == 0:
            return SharpeRatioResult(0, 0, 0, risk_free_rate, 0)
        
        try:
            # Calculate annualized metrics
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            
            # Annualize (assuming daily returns)
            annualized_return = mean_return * 252
            annualized_volatility = std_return * np.sqrt(252)
            
            # Calculate Sharpe ratio
            sharpe_ratio = ((annualized_return - risk_free_rate) / 
                          annualized_volatility if annualized_volatility > 0 else 0)
            
            return SharpeRatioResult(
                sharpe_ratio=sharpe_ratio,
                annualized_return=annualized_return,
                annualized_volatility=annualized_volatility,
                risk_free_rate=risk_free_rate,
                period_days=len(returns)
            )
            
        except Exception as e:
            context = create_error_context(
                component="risk_manager",
                operation="calculate_sharpe_ratio",
                returns_length=len(returns)
            )
            error_msg = f"Sharpe ratio calculation failed: {e}"
            self.logger.error(error_msg)
            raise RiskError(error_msg, "sharpe_ratio_calculation", context) from e
    
    def detect_anomalies_zscore(
        self, 
        values: np.ndarray, 
        threshold: float = 3.0
    ) -> List[Tuple[int, float]]:
        """
        Detect anomalies using Z-score method.
        
        Args:
            values: Array of values to analyze
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            List of (index, z_score) tuples for anomalies
        """
        if len(values) < 3:
            return []
        
        try:
            # Calculate Z-scores
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            
            if std_val == 0:
                return []
            
            z_scores = np.abs((values - mean_val) / std_val)
            
            # Find anomalies
            anomaly_indices = np.where(z_scores > threshold)[0]
            
            return [(idx, z_scores[idx]) for idx in anomaly_indices]
            
        except Exception as e:
            context = create_error_context(
                component="risk_manager",
                operation="detect_anomalies_zscore",
                values_length=len(values)
            )
            error_msg = f"Anomaly detection failed: {e}"
            self.logger.error(error_msg)
            raise RiskError(error_msg, "anomaly_detection", context) from e
    
    def assess_portfolio_risk(
        self, 
        opportunities: List[ArbitrageOpportunity],
        historical_returns: Optional[np.ndarray] = None,
        current_portfolio_value: float = 100000.0
    ) -> PortfolioRiskMetrics:
        """
        Perform comprehensive portfolio risk assessment.
        
        Args:
            opportunities: List of arbitrage opportunities
            historical_returns: Historical return data
            current_portfolio_value: Current portfolio value
            
        Returns:
            Comprehensive portfolio risk metrics
        """
        try:
            # Calculate total exposure
            total_exposure = sum(abs(opp.expected_profit) for opp in opportunities)
            
            # Calculate concentration risk
            concentration_risk = self._calculate_concentration_risk(opportunities)
            
            # Calculate correlation risk (simplified)
            correlation_risk = self._calculate_correlation_risk(opportunities)
            
            # Calculate liquidity risk
            liquidity_risk = self._calculate_liquidity_risk(opportunities)
            
            # Calculate time decay risk
            time_decay_risk = self._calculate_time_decay_risk(opportunities)
            
            # VaR calculation
            if historical_returns is not None and len(historical_returns) > 0:
                var_metrics = self.calculate_var(historical_returns)
            else:
                var_metrics = VaRResult(0, 0, 0, 0.95, 'historical', datetime.now())
            
            # Drawdown calculation (simplified)
            if historical_returns is not None and len(historical_returns) > 0:
                cumulative_returns = np.cumprod(1 + historical_returns)
                drawdown_metrics = self.calculate_maximum_drawdown(cumulative_returns)
            else:
                drawdown_metrics = DrawdownResult(0, 0, datetime.now(), datetime.now(), None, 0)
            
            # Sharpe ratio calculation
            sharpe_metrics = None
            if historical_returns is not None and len(historical_returns) > 10:
                sharpe_metrics = self.calculate_sharpe_ratio(historical_returns)
            
            # Calculate overall risk score
            overall_risk_score = self._calculate_overall_risk_score(
                concentration_risk, correlation_risk, liquidity_risk, 
                time_decay_risk, var_metrics
            )
            
            # Determine risk level
            risk_level = self._determine_risk_level(overall_risk_score)
            
            return PortfolioRiskMetrics(
                total_exposure=total_exposure,
                concentration_risk=concentration_risk,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk,
                time_decay_risk=time_decay_risk,
                var_metrics=var_metrics,
                drawdown_metrics=drawdown_metrics,
                sharpe_metrics=sharpe_metrics,
                overall_risk_score=overall_risk_score,
                risk_level=risk_level
            )
            
        except Exception as e:
            context = create_error_context(
                component="risk_manager",
                operation="assess_portfolio_risk",
                opportunities_count=len(opportunities)
            )
            error_msg = f"Portfolio risk assessment failed: {e}"
            self.logger.error(error_msg)
            raise RiskError(error_msg, "portfolio_risk_assessment", context) from e
    
    def _calculate_concentration_risk(self, opportunities: List[ArbitrageOpportunity]) -> float:
        """Calculate concentration risk in the portfolio."""
        if not opportunities:
            return 0.0
        
        # Group by underlying asset
        underlying_exposure = {}
        total_exposure = 0
        
        for opp in opportunities:
            exposure = abs(opp.expected_profit)
            total_exposure += exposure
            
            underlying = opp.underlying
            underlying_exposure[underlying] = underlying_exposure.get(underlying, 0) + exposure
        
        if total_exposure == 0:
            return 0.0
        
        # Calculate Herfindahl-Hirschman Index (HHI) for concentration
        hhi = sum((exposure / total_exposure) ** 2 for exposure in underlying_exposure.values())
        
        return hhi
    
    def _calculate_correlation_risk(self, opportunities: List[ArbitrageOpportunity]) -> float:
        """Calculate correlation risk (simplified)."""
        # Simplified correlation risk based on strategy diversity
        strategy_types = set(opp.strategy_type for opp in opportunities)
        
        if len(strategy_types) <= 1:
            return 1.0  # High correlation risk
        elif len(strategy_types) <= 2:
            return 0.7
        elif len(strategy_types) <= 3:
            return 0.4
        else:
            return 0.2  # Low correlation risk
    
    def _calculate_liquidity_risk(self, opportunities: List[ArbitrageOpportunity]) -> float:
        """Calculate liquidity risk based on volumes."""
        if not opportunities:
            return 0.0
        
        total_liquidity_score = 0
        count = 0
        
        for opp in opportunities:
            if opp.volumes:
                min_volume = min(opp.volumes.values())
                # Normalize volume to risk score (higher volume = lower risk)
                liquidity_score = max(0, 1 - min_volume / 1000)  # Risk decreases with volume
                total_liquidity_score += liquidity_score
                count += 1
        
        return total_liquidity_score / count if count > 0 else 0.5
    
    def _calculate_time_decay_risk(self, opportunities: List[ArbitrageOpportunity]) -> float:
        """Calculate time decay risk."""
        if not opportunities:
            return 0.0
        
        total_time_risk = 0
        
        for opp in opportunities:
            # Higher risk as expiry approaches
            time_risk = max(0, 1 - opp.days_to_expiry / 30)
            total_time_risk += time_risk
        
        return total_time_risk / len(opportunities)
    
    def _calculate_overall_risk_score(
        self, 
        concentration_risk: float,
        correlation_risk: float,
        liquidity_risk: float,
        time_decay_risk: float,
        var_metrics: VaRResult
    ) -> float:
        """Calculate overall risk score (0-1, higher is riskier)."""
        # Weighted combination of different risk factors
        weights = {
            'concentration': 0.25,
            'correlation': 0.20,
            'liquidity': 0.20,
            'time_decay': 0.15,
            'var': 0.20
        }
        
        # Normalize VaR to 0-1 scale
        var_normalized = min(var_metrics.var_1_day / self.base_var_threshold, 1.0)
        
        overall_risk = (
            concentration_risk * weights['concentration'] +
            correlation_risk * weights['correlation'] +
            liquidity_risk * weights['liquidity'] +
            time_decay_risk * weights['time_decay'] +
            var_normalized * weights['var']
        )
        
        return min(overall_risk, 1.0)
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level based on overall risk score."""
        if risk_score <= 0.2:
            return RiskLevel.LOW
        elif risk_score <= 0.4:
            return RiskLevel.MEDIUM
        elif risk_score <= 0.7:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def calculate_position_size(
        self, 
        opportunity: ArbitrageOpportunity,
        portfolio_value: float,
        risk_budget: float = 0.02
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion and risk constraints.
        
        Args:
            opportunity: Arbitrage opportunity
            portfolio_value: Current portfolio value
            risk_budget: Maximum risk budget as fraction of portfolio
            
        Returns:
            Suggested position size
        """
        try:
            # Kelly Criterion calculation
            win_prob = opportunity.confidence_score
            win_amount = opportunity.profit_margin
            loss_amount = opportunity.risk_score
            
            if loss_amount <= 0 or win_prob <= 0:
                return 0.0
            
            # Kelly fraction
            kelly_fraction = (win_prob * win_amount - (1 - win_prob) * loss_amount) / loss_amount
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            # Risk budget constraint
            max_position_by_risk = portfolio_value * risk_budget / loss_amount
            
            # Configuration constraint
            max_position_by_config = min(self.max_position_size, 
                                       portfolio_value * self.max_concentration)
            
            # Take minimum of all constraints
            suggested_position = min(
                kelly_fraction * portfolio_value,
                max_position_by_risk,
                max_position_by_config
            )
            
            return max(0, suggested_position)
            
        except Exception as e:
            context = create_error_context(
                component="risk_manager",
                operation="calculate_position_size",
                opportunity_id=getattr(opportunity, 'id', 'unknown')
            )
            error_msg = f"Position size calculation failed: {e}"
            self.logger.error(error_msg)
            raise RiskError(error_msg, "position_sizing", context) from e
    
    def validate_risk_limits(
        self, 
        opportunities: List[ArbitrageOpportunity],
        current_portfolio: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate that opportunities comply with risk limits.
        
        Args:
            opportunities: List of opportunities to validate
            current_portfolio: Optional current portfolio state
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        try:
            # Check individual opportunity limits
            for opp in opportunities:
                # Maximum loss check
                if abs(opp.max_loss) > self.config.max_daily_loss:
                    violations.append(f"Opportunity {opp.id} exceeds max daily loss limit")
                
                # Days to expiry check
                if not (self.config.min_days_to_expiry <= 
                       opp.days_to_expiry <= self.config.max_days_to_expiry):
                    violations.append(f"Opportunity {opp.id} violates expiry constraints")
                
                # Liquidity check
                if opp.volumes:
                    min_volume = min(opp.volumes.values())
                    if min_volume < self.config.min_liquidity_volume:
                        violations.append(f"Opportunity {opp.id} has insufficient liquidity")
            
            # Portfolio level checks
            portfolio_metrics = self.assess_portfolio_risk(opportunities)
            
            if portfolio_metrics.concentration_risk > self.max_concentration:
                violations.append("Portfolio exceeds maximum concentration limit")
            
            if portfolio_metrics.overall_risk_score > 0.8:
                violations.append("Portfolio overall risk score too high")
            
            return len(violations) == 0, violations
            
        except Exception as e:
            context = create_error_context(
                component="risk_manager",
                operation="validate_risk_limits",
                opportunities_count=len(opportunities)
            )
            error_msg = f"Risk validation failed: {e}"
            self.logger.error(error_msg)
            raise RiskError(error_msg, "risk_validation", context) from e