"""
Put-Call Parity Arbitrage Strategy Implementation.
看跌看涨平价策略实现

Exploits violations of put-call parity relationship to identify risk-free arbitrage opportunities.
The put-call parity theorem states that for European options:
C - P = S - K * e^(-r*T)

Where:
- C = Call option price
- P = Put option price  
- S = Current stock price
- K = Strike price
- r = Risk-free rate
- T = Time to expiration

Key Features:
- Identifies put-call parity violations
- Creates synthetic positions to exploit mispricings
- Supports both long and short parity trades
- Risk-free arbitrage when properly executed
- Handles early exercise considerations for American options
"""

import math
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple, Set
import itertools

from ..config.models import StrategyType, ArbitrageOpportunity
from .base import (
    BaseStrategy, StrategyResult, StrategyParameters, OptionData, 
    TradingAction, ActionType, RiskMetrics, RiskLevel, StrategyRegistry,
    OptionType
)


class PutCallParityParameters(StrategyParameters):
    """Parameters specific to put-call parity arbitrage strategy."""
    
    # Parity violation thresholds
    min_parity_deviation: float = 0.02  # Minimum 2% deviation to consider arbitrage
    max_parity_deviation: float = 0.5   # Maximum 50% deviation (sanity check)
    
    # Risk-free rate and dividend assumptions
    risk_free_rate: float = 0.03        # 3% annual risk-free rate
    dividend_yield: float = 0.0         # Assume no dividends unless specified
    
    # Strike price matching
    exact_strike_match: bool = True     # Require exact strike matches
    max_strike_difference: float = 0.01 # Maximum strike difference if not exact
    
    # Time to expiry matching
    exact_expiry_match: bool = True     # Require exact expiry matches
    max_expiry_difference: int = 1      # Maximum days difference if not exact
    
    # Transaction cost considerations
    include_transaction_costs: bool = True
    transaction_cost_rate: float = 0.005  # 0.5% transaction cost per side
    
    # Position sizing
    max_position_size: int = 10         # Maximum contracts per arbitrage
    prefer_liquid_options: bool = True  # Prefer options with higher volume
    
    # Early exercise protection (for American options)
    avoid_deep_itm_puts: bool = True    # Avoid deep ITM puts due to early exercise risk
    max_itm_amount: float = 0.1         # Maximum 10% ITM for puts


@StrategyRegistry.register(StrategyType.PUT_CALL_PARITY)
class PutCallParityStrategy(BaseStrategy):
    """
    Put-Call Parity Arbitrage Strategy.
    
    This strategy identifies and exploits violations of the put-call parity relationship.
    Put-call parity is a fundamental relationship in options pricing that creates 
    risk-free arbitrage opportunities when violated.
    
    The strategy:
    1. Finds matching call and put options (same strike, expiry, underlying)
    2. Calculates theoretical parity relationship
    3. Identifies significant deviations from parity
    4. Creates appropriate long/short synthetic positions
    5. Monitors for early exercise risks (American options)
    
    Types of arbitrage trades:
    - Long synthetic stock: Buy call, sell put, sell stock
    - Short synthetic stock: Sell call, buy put, buy stock  
    - Long synthetic call: Buy stock, buy put, sell call
    - Long synthetic put: Sell stock, buy call, sell put
    """
    
    def __init__(self, parameters: Optional[PutCallParityParameters] = None):
        super().__init__(parameters or PutCallParityParameters())
        self.params = self.parameters  # Type hint helper
    
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.PUT_CALL_PARITY
    
    def scan_opportunities(self, options_data: List[OptionData]) -> StrategyResult:
        """
        Scan for put-call parity arbitrage opportunities.
        
        Args:
            options_data: List of option market data
            
        Returns:
            StrategyResult: Results with found opportunities
        """
        start_time = datetime.now()
        opportunities = []
        
        try:
            # Filter options suitable for parity analysis
            filtered_options = self.filter_options(options_data)
            if len(filtered_options) < 2:
                return StrategyResult(
                    strategy_name=self.name,
                    opportunities=[],
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    data_timestamp=datetime.now(),
                    success=True,
                    metadata={'insufficient_data': True}
                )
            
            # Find put-call pairs for parity analysis
            parity_pairs = self._find_parity_pairs(filtered_options)
            
            if not parity_pairs:
                return StrategyResult(
                    strategy_name=self.name,
                    opportunities=[],
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    data_timestamp=datetime.now(),
                    success=True,
                    metadata={'no_matching_pairs': True}
                )
            
            # Analyze each pair for parity violations
            for call_option, put_option in parity_pairs:
                parity_opportunity = self._analyze_parity_violation(call_option, put_option)
                if parity_opportunity and self.validate_opportunity(parity_opportunity):
                    parity_opportunity.confidence_score = self.calculate_confidence_score(parity_opportunity)
                    opportunities.append(parity_opportunity)
            
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
                    'parity_pairs_found': len(parity_pairs),
                    'opportunities_found': len(opportunities),
                    'risk_free_rate': self.params.risk_free_rate,
                    'dividend_yield': self.params.dividend_yield
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
    
    def _find_parity_pairs(self, options_data: List[OptionData]) -> List[Tuple[OptionData, OptionData]]:
        """
        Find matching put-call pairs suitable for parity analysis.
        
        Args:
            options_data: Filtered options data
            
        Returns:
            List of (call_option, put_option) pairs
        """
        parity_pairs = []
        
        # Group options by underlying
        options_by_underlying = {}
        for option in options_data:
            if option.underlying not in options_by_underlying:
                options_by_underlying[option.underlying] = []
            options_by_underlying[option.underlying].append(option)
        
        # For each underlying, find matching call-put pairs
        for underlying, options in options_by_underlying.items():
            calls = [opt for opt in options if opt.option_type == OptionType.CALL]
            puts = [opt for opt in options if opt.option_type == OptionType.PUT]
            
            # Find matching pairs
            for call in calls:
                for put in puts:
                    if self._are_matching_options(call, put):
                        parity_pairs.append((call, put))
        
        return parity_pairs
    
    def _are_matching_options(self, call_option: OptionData, put_option: OptionData) -> bool:
        """
        Check if call and put options are suitable for parity analysis.
        
        Args:
            call_option: Call option data
            put_option: Put option data
            
        Returns:
            True if options form a valid parity pair
        """
        # Must be same underlying
        if call_option.underlying != put_option.underlying:
            return False
        
        # Check strike price matching
        if self.params.exact_strike_match:
            if call_option.strike_price != put_option.strike_price:
                return False
        else:
            strike_diff = abs(call_option.strike_price - put_option.strike_price)
            if strike_diff > self.params.max_strike_difference:
                return False
        
        # Check expiry date matching
        if self.params.exact_expiry_match:
            if call_option.expiry_date != put_option.expiry_date:
                return False
        else:
            expiry_diff = abs((call_option.expiry_date - put_option.expiry_date).days)
            if expiry_diff > self.params.max_expiry_difference:
                return False
        
        # Avoid deep ITM puts if configured (early exercise risk)
        if self.params.avoid_deep_itm_puts:
            # Estimate underlying price (simplified - use call option price as proxy)
            estimated_underlying = call_option.market_price + put_option.strike_price
            itm_amount = max(0, put_option.strike_price - estimated_underlying) / put_option.strike_price
            
            if itm_amount > self.params.max_itm_amount:
                return False
        
        return True
    
    def _analyze_parity_violation(
        self, 
        call_option: OptionData, 
        put_option: OptionData
    ) -> Optional[ArbitrageOpportunity]:
        """
        Analyze a put-call pair for parity violations.
        
        Args:
            call_option: Call option data
            put_option: Put option data
            
        Returns:
            ArbitrageOpportunity if violation found, None otherwise
        """
        try:
            # Calculate theoretical put-call parity relationship
            # C - P = S - K * e^(-r*T)
            
            # Get option parameters
            K = call_option.strike_price  # Strike price
            T = call_option.time_to_expiry  # Time to expiry in years
            r = self.params.risk_free_rate
            q = self.params.dividend_yield
            
            # Estimate underlying price from options (simplified approach)
            # In practice, you would use real underlying price
            S = self._estimate_underlying_price(call_option, put_option, K, T, r, q)
            
            # Calculate theoretical parity value
            discount_factor = math.exp(-r * T)
            dividend_factor = math.exp(-q * T) if q > 0 else 1.0
            
            theoretical_parity = S * dividend_factor - K * discount_factor
            actual_parity = call_option.mid_price - put_option.mid_price
            
            parity_deviation = actual_parity - theoretical_parity
            parity_deviation_pct = abs(parity_deviation) / max(abs(theoretical_parity), call_option.mid_price)
            
            # Check if deviation is significant
            if parity_deviation_pct < self.params.min_parity_deviation:
                return None
            
            # Sanity check for extreme deviations
            if parity_deviation_pct > self.params.max_parity_deviation:
                return None
            
            # Determine arbitrage strategy
            if parity_deviation > 0:
                # Actual parity > theoretical parity
                # Call is relatively expensive or put is relatively cheap
                # Strategy: Sell call, buy put, buy stock (short synthetic stock)
                actions = [
                    TradingAction(call_option.code, ActionType.SELL, 1, call_option.mid_price),
                    TradingAction(put_option.code, ActionType.BUY, 1, put_option.mid_price),
                    # Note: In practice, would also buy underlying stock
                ]
                strategy_type = "short_synthetic_stock"
                reasoning = f"Call overpriced relative to put by {parity_deviation:.4f}"
                
            else:
                # Actual parity < theoretical parity  
                # Put is relatively expensive or call is relatively cheap
                # Strategy: Buy call, sell put, sell stock (long synthetic stock)
                actions = [
                    TradingAction(call_option.code, ActionType.BUY, 1, call_option.mid_price),
                    TradingAction(put_option.code, ActionType.SELL, 1, put_option.mid_price),
                    # Note: In practice, would also sell underlying stock
                ]
                strategy_type = "long_synthetic_stock"
                reasoning = f"Put overpriced relative to call by {abs(parity_deviation):.4f}"
            
            # Calculate expected profit (before transaction costs)
            gross_profit = abs(parity_deviation)
            
            # Adjust for transaction costs if configured
            if self.params.include_transaction_costs:
                total_position_value = call_option.mid_price + put_option.mid_price
                transaction_costs = total_position_value * self.params.transaction_cost_rate * 2  # Both sides
                net_profit = gross_profit - transaction_costs
                
                # Skip if not profitable after costs
                if net_profit <= 0:
                    return None
            else:
                net_profit = gross_profit
                transaction_costs = 0
            
            # Calculate risk metrics
            risk_metrics = self.assess_risk([call_option, put_option], actions)
            
            # Create opportunity
            opportunity = ArbitrageOpportunity(
                id=f"pcp_{strategy_type}_{call_option.code}_{put_option.code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                strategy_type=self.strategy_type,
                timestamp=datetime.now(),
                instruments=[call_option.code, put_option.code],
                underlying=call_option.underlying,
                expected_profit=net_profit,
                profit_margin=parity_deviation_pct,
                confidence_score=0.0,  # Will be calculated later
                max_loss=risk_metrics.max_loss,
                risk_score=self._calculate_risk_score(risk_metrics),
                days_to_expiry=min(call_option.days_to_expiry, put_option.days_to_expiry),
                market_prices={
                    call_option.code: call_option.mid_price,
                    put_option.code: put_option.mid_price
                },
                volumes={
                    call_option.code: call_option.volume,
                    put_option.code: put_option.volume
                },
                actions=[{
                    'instrument': action.instrument,
                    'action': action.action.value,
                    'quantity': action.quantity,
                    'price': action.price,
                    'reasoning': reasoning
                } for action in actions],
                data_source="put_call_parity_analysis",
                parameters={
                    'theoretical_parity': theoretical_parity,
                    'actual_parity': actual_parity,
                    'parity_deviation': parity_deviation,
                    'parity_deviation_pct': parity_deviation_pct,
                    'estimated_underlying_price': S,
                    'strike_price': K,
                    'time_to_expiry': T,
                    'risk_free_rate': r,
                    'dividend_yield': q,
                    'strategy_subtype': strategy_type,
                    'transaction_costs': transaction_costs,
                    'gross_profit': gross_profit
                }
            )
            
            return opportunity
            
        except Exception as e:
            self.logger.error(f"Error analyzing parity violation: {e}")
            return None
    
    def _estimate_underlying_price(
        self, 
        call_option: OptionData, 
        put_option: OptionData,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        dividend_yield: float
    ) -> float:
        """
        Estimate underlying asset price from option prices.
        
        Uses put-call parity relationship to estimate underlying price:
        S = C - P + K * e^(-r*T)
        
        Args:
            call_option: Call option data
            put_option: Put option data  
            strike: Strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free rate
            dividend_yield: Dividend yield
            
        Returns:
            Estimated underlying price
        """
        try:
            discount_factor = math.exp(-risk_free_rate * time_to_expiry)
            dividend_factor = math.exp(dividend_yield * time_to_expiry) if dividend_yield > 0 else 1.0
            
            # S = (C - P + K * e^(-r*T)) / e^(-q*T)
            estimated_price = (call_option.mid_price - put_option.mid_price + strike * discount_factor) / dividend_factor
            
            # Sanity check - price should be positive and reasonable relative to strike
            if estimated_price <= 0 or estimated_price > strike * 10:
                # Fallback: use simple approximation
                estimated_price = call_option.mid_price + strike
            
            return estimated_price
            
        except Exception:
            # Fallback estimation
            return call_option.mid_price + strike
    
    def calculate_profit(self, options: List[OptionData], actions: List[TradingAction]) -> float:
        """
        Calculate expected profit from put-call parity arbitrage.
        
        Args:
            options: Options involved
            actions: Trading actions
            
        Returns:
            Expected profit (should be risk-free if properly executed)
        """
        # For put-call parity, profit is locked in at initiation
        # This is a theoretical risk-free arbitrage
        
        total_cash_flow = 0.0
        
        for action in actions:
            option = next((opt for opt in options if opt.code == action.instrument), None)
            if not option:
                continue
            
            position_value = (action.price or option.mid_price) * action.quantity
            
            if action.action == ActionType.BUY:
                total_cash_flow -= position_value  # Cash outflow
            else:
                total_cash_flow += position_value   # Cash inflow
        
        # In perfect parity arbitrage, profit equals the initial cash flow
        # (assuming the underlying position is also included)
        return total_cash_flow
    
    def assess_risk(self, options: List[OptionData], actions: List[TradingAction]) -> RiskMetrics:
        """
        Assess risk for put-call parity arbitrage.
        
        Note: True put-call parity arbitrage should be risk-free when properly executed
        with the underlying asset position included.
        
        Args:
            options: Options involved
            actions: Trading actions
            
        Returns:
            Risk assessment (should show minimal risk for true arbitrage)
        """
        # Theoretical put-call parity arbitrage is risk-free
        # However, practical considerations introduce some risks
        
        max_loss = 0.0
        max_gain = 0.0
        liquidity_risk = 0.0
        time_decay_risk = 0.0
        
        total_position_value = 0.0
        
        for action in actions:
            option = next((opt for opt in options if opt.code == action.instrument), None)
            if not option:
                continue
            
            position_value = (action.price or option.mid_price) * action.quantity
            total_position_value += abs(position_value)
            
            # In theory, max loss should be zero for true arbitrage
            # In practice, there are execution risks
            if action.action == ActionType.BUY:
                # Risk is limited to premium paid (if unable to execute other legs)
                max_loss += position_value * 0.1  # 10% of position as execution risk
            else:
                # Risk from uncovered short positions (if unable to execute other legs)
                max_loss += position_value * 0.1  # 10% execution risk
            
            # Max gain is the arbitrage profit (typically small but risk-free)
            max_gain += position_value * 0.02  # Assume ~2% arbitrage profit
            
            # Liquidity risk from bid-ask spreads
            if option.spread_pct > 0:
                liquidity_risk = max(liquidity_risk, option.spread_pct)
            
            # Time decay risk (minimal for arbitrage held to expiry)
            time_decay_risk = max(time_decay_risk, 0.1)  # Low time decay risk
        
        # Volatility risk is minimal for parity arbitrage
        volatility_risk = 0.05  # Very low vol risk
        
        # High probability of profit for true arbitrage
        probability_profit = 0.95
        
        # Adjust for liquidity issues
        if liquidity_risk > 0.05:
            probability_profit *= 0.9
        
        expected_return = probability_profit * max_gain - (1 - probability_profit) * max_loss
        
        # Risk level should be very low for true arbitrage
        risk_level = RiskLevel.LOW
        
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
        """Calculate normalized risk score - should be low for parity arbitrage."""
        # Put-call parity arbitrage should have very low risk scores
        base_score = 0.1  # Start with low base risk
        
        # Add liquidity risk component
        base_score += min(risk_metrics.liquidity_risk, 0.1) * 0.5  # Cap liquidity impact
        
        # Add execution risk
        base_score += 0.05  # Small execution risk
        
        return min(base_score, 0.3)  # Cap at 30% risk score
    
    def filter_options(self, options_data: List[OptionData]) -> List[OptionData]:
        """
        Filter options for put-call parity analysis.
        
        Args:
            options_data: Raw options data
            
        Returns:
            Options suitable for parity analysis
        """
        filtered = super().filter_options(options_data)
        
        result = []
        for option in filtered:
            # Must have reasonable bid-ask spread
            if option.spread_pct > 0.1:  # Skip if spread > 10%
                continue
            
            # Prefer liquid options if configured
            if self.params.prefer_liquid_options and option.volume < 10:
                continue
            
            # Must have positive prices
            if option.bid_price <= 0 or option.ask_price <= 0:
                continue
            
            result.append(option)
        
        return result
    
    def calculate_confidence_score(self, opportunity: ArbitrageOpportunity) -> float:
        """
        Calculate confidence score for put-call parity opportunities.
        
        Parity arbitrage should have high confidence scores due to theoretical risk-free nature.
        """
        base_score = 0.8  # Start with high confidence for arbitrage
        
        # Adjust based on parity deviation magnitude
        parity_deviation = opportunity.parameters.get('parity_deviation_pct', 0)
        if parity_deviation > 0.05:  # Strong signal
            base_score += 0.1
        
        # Adjust for liquidity
        min_volume = min(opportunity.volumes.values()) if opportunity.volumes else 0
        if min_volume > 50:
            base_score += 0.05
        elif min_volume < 10:
            base_score -= 0.1
        
        # Adjust for time to expiry (more time = higher confidence)
        if opportunity.days_to_expiry > 30:
            base_score += 0.05
        elif opportunity.days_to_expiry < 7:
            base_score -= 0.1
        
        # Adjust for transaction costs
        transaction_costs = opportunity.parameters.get('transaction_costs', 0)
        gross_profit = opportunity.parameters.get('gross_profit', opportunity.expected_profit)
        if gross_profit > 0:
            cost_ratio = transaction_costs / gross_profit
            if cost_ratio > 0.5:  # High cost ratio
                base_score -= 0.2
        
        return min(max(base_score, 0.0), 1.0)