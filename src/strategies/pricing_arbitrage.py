"""
Pricing arbitrage strategy implementation.
Identifies options trading at significant deviations from theoretical price.
"""

from datetime import datetime
from typing import List, Optional
import math

from ..config.models import StrategyType, ArbitrageOpportunity
from .base import (
    BaseStrategy, StrategyResult, StrategyParameters, OptionData, 
    TradingAction, ActionType, RiskMetrics, RiskLevel, StrategyRegistry
)


class PricingArbitrageParameters(StrategyParameters):
    """Parameters specific to pricing arbitrage strategy."""
    
    min_price_deviation: float = 0.05  # Minimum 5% price deviation
    max_price_deviation: float = 0.5   # Maximum 50% price deviation (sanity check)
    require_theoretical_price: bool = True  # Require theoretical price calculation
    min_implied_volatility: float = 0.01  # Minimum IV of 1%
    max_implied_volatility: float = 2.0   # Maximum IV of 200%


@StrategyRegistry.register(StrategyType.PRICING_ARBITRAGE)
class PricingArbitrageStrategy(BaseStrategy):
    """
    Strategy to identify pricing arbitrage opportunities.
    
    Finds options where market price significantly deviates from theoretical price,
    suggesting mispricing that can be exploited through buying underpriced options
    or selling overpriced options.
    """
    
    def __init__(self, parameters: Optional[PricingArbitrageParameters] = None):
        super().__init__(parameters or PricingArbitrageParameters())
        self.params = self.parameters  # Type hint helper
        
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.PRICING_ARBITRAGE
    
    def scan_opportunities(self, options_data: List[OptionData]) -> StrategyResult:
        """
        Scan for pricing arbitrage opportunities.
        
        Args:
            options_data: List of option market data
            
        Returns:
            StrategyResult: Results with found opportunities
        """
        start_time = datetime.now()
        opportunities = []
        
        try:
            # Filter options based on parameters
            filtered_options = self.filter_options(options_data)
            
            # Find mispriced options
            for option in filtered_options:
                opportunity = self._analyze_option_pricing(option)
                if opportunity and self.validate_opportunity(opportunity):
                    opportunities.append(opportunity)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return StrategyResult(
                strategy_name=self.name,
                opportunities=opportunities,
                execution_time=execution_time,
                data_timestamp=datetime.now(),
                success=True,
                metadata={
                    'total_options_analyzed': len(options_data),
                    'filtered_options': len(filtered_options),
                    'opportunities_found': len(opportunities)
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return StrategyResult(
                strategy_name=self.name,
                opportunities=[],
                execution_time=execution_time,
                data_timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            )
    
    def _analyze_option_pricing(self, option: OptionData) -> Optional[ArbitrageOpportunity]:
        """
        Analyze single option for pricing arbitrage.
        
        Args:
            option: Option to analyze
            
        Returns:
            ArbitrageOpportunity if mispricing found, None otherwise
        """
        # Skip if no theoretical price available and required
        if self.params.require_theoretical_price and option.theoretical_price is None:
            return None
            
        # Skip if implied volatility is out of bounds
        if option.implied_volatility is not None:
            if not (self.params.min_implied_volatility <= option.implied_volatility <= self.params.max_implied_volatility):
                return None
        
        # Calculate price deviation
        if option.theoretical_price is None or option.theoretical_price <= 0:
            return None
            
        price_deviation = (option.market_price - option.theoretical_price) / option.theoretical_price
        
        # Check if deviation meets threshold
        if abs(price_deviation) < self.params.min_price_deviation:
            return None
            
        # Sanity check for extreme deviations
        if abs(price_deviation) > self.params.max_price_deviation:
            return None
        
        # Determine trading action
        if price_deviation < 0:
            # Option is underpriced - BUY
            action_type = ActionType.BUY
            profit_potential = abs(option.theoretical_price - option.market_price)
        else:
            # Option is overpriced - SELL
            action_type = ActionType.SELL
            profit_potential = abs(option.market_price - option.theoretical_price)
        
        # Create trading action
        trading_actions = [
            TradingAction(
                instrument=option.code,
                action=action_type,
                quantity=1,  # Single contract for analysis
                price=option.market_price
            )
        ]
        
        # Calculate risk metrics
        risk_metrics = self.assess_risk([option], trading_actions)
        
        # Create opportunity
        opportunity = ArbitrageOpportunity(
            id=f"pricing_{option.code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            strategy_type=self.strategy_type,
            timestamp=datetime.now(),
            instruments=[option.code],
            underlying=option.underlying,
            expected_profit=profit_potential,
            profit_margin=abs(price_deviation),
            confidence_score=0.0,  # Will be calculated below
            max_loss=risk_metrics.max_loss,
            risk_score=self._calculate_risk_score(risk_metrics),
            days_to_expiry=option.days_to_expiry,
            market_prices={option.code: option.market_price},
            theoretical_prices={option.code: option.theoretical_price},
            volumes={option.code: option.volume},
            actions=[{
                'instrument': action.instrument,
                'action': action.action.value,
                'quantity': action.quantity,
                'price': action.price,
                'reasoning': f"Option is {'underpriced' if action_type == ActionType.BUY else 'overpriced'} by {abs(price_deviation)*100:.1f}%"
            } for action in trading_actions],
            data_source="strategy_analysis",
            parameters={
                'price_deviation': price_deviation,
                'theoretical_price': option.theoretical_price,
                'implied_volatility': option.implied_volatility,
                'time_to_expiry': option.time_to_expiry
            }
        )
        
        # Calculate and set confidence score
        opportunity.confidence_score = self.calculate_confidence_score(opportunity)
        
        return opportunity
    
    def calculate_profit(self, options: List[OptionData], actions: List[TradingAction]) -> float:
        """
        Calculate expected profit from pricing arbitrage.
        
        Args:
            options: Options involved
            actions: Trading actions
            
        Returns:
            float: Expected profit
        """
        total_profit = 0.0
        
        for action in actions:
            option = next((opt for opt in options if opt.code == action.instrument), None)
            if not option or option.theoretical_price is None:
                continue
                
            if action.action == ActionType.BUY:
                # Profit from buying underpriced option
                profit_per_unit = option.theoretical_price - (action.price or option.market_price)
            else:
                # Profit from selling overpriced option
                profit_per_unit = (action.price or option.market_price) - option.theoretical_price
                
            total_profit += profit_per_unit * action.quantity
        
        return total_profit
    
    def assess_risk(self, options: List[OptionData], actions: List[TradingAction]) -> RiskMetrics:
        """
        Assess risk for pricing arbitrage strategy.
        
        Args:
            options: Options involved
            actions: Trading actions
            
        Returns:
            RiskMetrics: Risk assessment
        """
        max_loss = 0.0
        max_gain = 0.0
        liquidity_risk = 0.0
        time_decay_risk = 0.0
        volatility_risk = 0.0
        
        for action in actions:
            option = next((opt for opt in options if opt.code == action.instrument), None)
            if not option:
                continue
            
            position_value = (action.price or option.market_price) * action.quantity
            
            if action.action == ActionType.BUY:
                # Long option - max loss is premium paid
                max_loss += position_value
                # Max gain depends on option type and movement
                if option.option_type.value == 'C':  # Call option
                    max_gain += float('inf')  # Theoretically unlimited for calls
                else:  # Put option
                    max_gain += (option.strike_price - (action.price or option.market_price)) * action.quantity
            else:
                # Short option - max gain is premium received
                max_gain += position_value
                # Max loss can be substantial
                if option.option_type.value == 'C':  # Call option
                    max_loss += float('inf')  # Theoretically unlimited for short calls
                else:  # Put option  
                    max_loss += option.strike_price * action.quantity
            
            # Liquidity risk based on bid-ask spread
            if option.spread_pct > 0:
                liquidity_risk = max(liquidity_risk, option.spread_pct)
            
            # Time decay risk (higher for short-term options)
            if option.time_to_expiry > 0:
                time_decay_risk = max(time_decay_risk, 1.0 / math.sqrt(option.time_to_expiry * 365))
            
            # Volatility risk
            if option.implied_volatility:
                volatility_risk = max(volatility_risk, option.implied_volatility)
        
        # Cap infinite values for practical risk assessment
        max_gain = min(max_gain, max_loss * 10) if max_gain != float('inf') else max_loss * 5
        max_loss = min(max_loss, 1000000)  # Cap at 1M for sanity
        
        # Calculate probability of profit (simplified heuristic)
        probability_profit = 0.6  # Default moderate probability
        if liquidity_risk > 0.1:  # High spread reduces probability
            probability_profit *= 0.8
        if time_decay_risk > 1.0:  # High time decay reduces probability
            probability_profit *= 0.9
            
        expected_return = (probability_profit * max_gain) - ((1 - probability_profit) * max_loss)
        
        # Determine risk level
        if max_loss < 1000:
            risk_level = RiskLevel.LOW
        elif max_loss < 5000:
            risk_level = RiskLevel.MEDIUM
        elif max_loss < 20000:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL
        
        return RiskMetrics(
            max_loss=max_loss,
            max_gain=max_gain,
            probability_profit=probability_profit,
            expected_return=expected_return,
            risk_level=risk_level,
            liquidity_risk=liquidity_risk,
            time_decay_risk=time_decay_risk,
            volatility_risk=volatility_risk
        )
    
    def _calculate_risk_score(self, risk_metrics: RiskMetrics) -> float:
        """
        Calculate normalized risk score from risk metrics.
        
        Args:
            risk_metrics: Risk metrics
            
        Returns:
            float: Risk score between 0 and 1
        """
        score = 0.0
        
        # Risk level component (0.4 weight)
        risk_level_scores = {
            RiskLevel.LOW: 0.1,
            RiskLevel.MEDIUM: 0.3,
            RiskLevel.HIGH: 0.6,
            RiskLevel.CRITICAL: 0.9
        }
        score += risk_level_scores.get(risk_metrics.risk_level, 0.5) * 0.4
        
        # Liquidity risk component (0.3 weight)  
        score += min(risk_metrics.liquidity_risk, 1.0) * 0.3
        
        # Time decay risk component (0.2 weight)
        score += min(risk_metrics.time_decay_risk, 1.0) * 0.2
        
        # Volatility risk component (0.1 weight)
        score += min(risk_metrics.volatility_risk, 1.0) * 0.1
        
        return min(score, 1.0)
    
    def filter_options(self, options_data: List[OptionData]) -> List[OptionData]:
        """
        Filter options for pricing arbitrage analysis.
        
        Args:
            options_data: Raw options data
            
        Returns:
            List[OptionData]: Filtered options
        """
        filtered = super().filter_options(options_data)
        
        # Additional filtering specific to pricing arbitrage
        result = []
        for option in filtered:
            # Must have theoretical price if required
            if self.params.require_theoretical_price and option.theoretical_price is None:
                continue
                
            # Must have reasonable implied volatility
            if option.implied_volatility is not None:
                if not (self.params.min_implied_volatility <= option.implied_volatility <= self.params.max_implied_volatility):
                    continue
            
            # Must have positive bid-ask spread (indicates active trading)
            if option.spread <= 0:
                continue
                
            result.append(option)
        
        return result