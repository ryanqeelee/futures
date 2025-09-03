"""
Volatility Arbitrage Strategy Implementation.
波动率套利策略实现

Identifies arbitrage opportunities based on implied volatility discrepancies
between options with similar characteristics or between implied and historical volatility.

Key Features:
- Implied volatility surface analysis
- Historical vs implied volatility comparison  
- Calendar spread volatility arbitrage
- Volatility smile/skew exploitation
- Statistical arbitrage based on volatility mean reversion
"""

import math
import statistics
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import numpy as np

from ..config.models import StrategyType, ArbitrageOpportunity
from .base import (
    BaseStrategy, StrategyResult, StrategyParameters, OptionData, 
    TradingAction, ActionType, RiskMetrics, RiskLevel, StrategyRegistry,
    OptionType
)


class VolatilityArbitrageParameters(StrategyParameters):
    """Parameters specific to volatility arbitrage strategy."""
    
    # Volatility thresholds
    min_iv_spread: float = 0.05  # Minimum 5% IV spread between options
    max_iv_level: float = 1.0    # Maximum 100% IV level
    min_iv_level: float = 0.02   # Minimum 2% IV level
    
    # Historical volatility comparison
    use_historical_volatility: bool = True
    historical_volatility_window: int = 30  # Days for HV calculation
    min_hv_iv_spread: float = 0.03  # Minimum 3% spread between HV and IV
    
    # Calendar spread parameters
    enable_calendar_spreads: bool = True
    min_calendar_days: int = 7    # Minimum days between expiries
    max_calendar_days: int = 90   # Maximum days between expiries
    
    # Volatility surface analysis
    enable_surface_analysis: bool = True
    min_surface_points: int = 5   # Minimum options needed for surface analysis
    skew_threshold: float = 0.02  # Threshold for detecting volatility skew anomalies
    
    # Statistical parameters
    volatility_mean_reversion_period: int = 20  # Days for mean reversion analysis
    confidence_threshold: float = 0.7  # Minimum confidence for statistical arbitrage


@StrategyRegistry.register(StrategyType.VOLATILITY_ARBITRAGE)
class VolatilityArbitrageStrategy(BaseStrategy):
    """
    Advanced volatility arbitrage strategy.
    
    This strategy identifies and exploits discrepancies in implied volatility across:
    1. Different strikes at same expiry (volatility smile/skew)
    2. Different expiries at same strike (volatility term structure)  
    3. Implied vs historical volatility (statistical arbitrage)
    4. Calendar spreads with volatility advantages
    
    The strategy uses sophisticated statistical models to identify when volatility
    is mispriced and creates appropriate long/short volatility positions.
    """
    
    def __init__(self, parameters: Optional[VolatilityArbitrageParameters] = None):
        super().__init__(parameters or VolatilityArbitrageParameters())
        self.params = self.parameters  # Type hint helper
    
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.VOLATILITY_ARBITRAGE
    
    def scan_opportunities(self, options_data: List[OptionData]) -> StrategyResult:
        """
        Scan for volatility arbitrage opportunities.
        
        Args:
            options_data: List of option market data
            
        Returns:
            StrategyResult: Results with found opportunities
        """
        start_time = datetime.now()
        opportunities = []
        
        try:
            # Filter and organize options
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
            
            # Group options by underlying for analysis
            options_by_underlying = self._group_options_by_underlying(filtered_options)
            
            # Analyze each underlying separately
            for underlying, options in options_by_underlying.items():
                if len(options) < 2:
                    continue
                
                # 1. Volatility smile/skew arbitrage
                smile_opportunities = self._analyze_volatility_smile(options)
                opportunities.extend(smile_opportunities)
                
                # 2. Calendar spread volatility arbitrage
                if self.params.enable_calendar_spreads:
                    calendar_opportunities = self._analyze_calendar_spreads(options)
                    opportunities.extend(calendar_opportunities)
                
                # 3. Historical vs implied volatility arbitrage
                if self.params.use_historical_volatility:
                    hv_opportunities = self._analyze_historical_volatility(options)
                    opportunities.extend(hv_opportunities)
                
                # 4. Volatility surface analysis
                if self.params.enable_surface_analysis and len(options) >= self.params.min_surface_points:
                    surface_opportunities = self._analyze_volatility_surface(options)
                    opportunities.extend(surface_opportunities)
            
            # Filter and validate opportunities
            validated_opportunities = []
            for opportunity in opportunities:
                if self.validate_opportunity(opportunity):
                    opportunity.confidence_score = self.calculate_confidence_score(opportunity)
                    validated_opportunities.append(opportunity)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return StrategyResult(
                strategy_name=self.name,
                opportunities=validated_opportunities,
                execution_time=execution_time,
                data_timestamp=datetime.now(),
                success=True,
                metadata={
                    'total_options_analyzed': len(options_data),
                    'filtered_options': len(filtered_options),
                    'underlyings_analyzed': len(options_by_underlying),
                    'opportunities_found': len(validated_opportunities),
                    'analysis_methods': {
                        'volatility_smile': True,
                        'calendar_spreads': self.params.enable_calendar_spreads,
                        'historical_volatility': self.params.use_historical_volatility,
                        'surface_analysis': self.params.enable_surface_analysis
                    }
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
    
    def _group_options_by_underlying(self, options_data: List[OptionData]) -> Dict[str, List[OptionData]]:
        """Group options by underlying asset."""
        groups = {}
        for option in options_data:
            if option.underlying not in groups:
                groups[option.underlying] = []
            groups[option.underlying].append(option)
        return groups
    
    def _analyze_volatility_smile(self, options: List[OptionData]) -> List[ArbitrageOpportunity]:
        """
        Analyze volatility smile/skew for arbitrage opportunities.
        
        Looks for options with similar characteristics but significantly different
        implied volatilities, indicating potential mispricing.
        """
        opportunities = []
        
        # Group by expiry date and option type
        expiry_groups = {}
        for option in options:
            if option.implied_volatility is None:
                continue
                
            key = (option.expiry_date, option.option_type)
            if key not in expiry_groups:
                expiry_groups[key] = []
            expiry_groups[key].append(option)
        
        # Analyze each expiry/type group for volatility anomalies
        for (expiry_date, option_type), group_options in expiry_groups.items():
            if len(group_options) < 3:  # Need at least 3 points for smile analysis
                continue
            
            # Sort by strike price
            group_options.sort(key=lambda x: x.strike_price)
            
            # Calculate expected volatility using neighbors
            for i, option in enumerate(group_options):
                if i == 0 or i == len(group_options) - 1:
                    continue  # Skip endpoints
                
                # Get neighboring options
                left_option = group_options[i-1]
                right_option = group_options[i+1]
                
                # Calculate expected IV using linear interpolation
                strike_weight = ((option.strike_price - left_option.strike_price) / 
                               (right_option.strike_price - left_option.strike_price))
                
                expected_iv = (left_option.implied_volatility * (1 - strike_weight) + 
                              right_option.implied_volatility * strike_weight)
                
                # Check for significant deviation
                iv_spread = abs(option.implied_volatility - expected_iv)
                if iv_spread >= self.params.min_iv_spread:
                    
                    # Determine trade direction
                    if option.implied_volatility > expected_iv:
                        # Option is over-volatized - SELL
                        action_type = ActionType.SELL
                        profit_potential = (option.implied_volatility - expected_iv) * option.market_price
                    else:
                        # Option is under-volatized - BUY
                        action_type = ActionType.BUY
                        profit_potential = (expected_iv - option.implied_volatility) * option.market_price
                    
                    # Create opportunity
                    opportunity = self._create_volatility_opportunity(
                        option, expected_iv, action_type, profit_potential,
                        "volatility_smile", f"IV deviation: {iv_spread:.3f}"
                    )
                    
                    if opportunity:
                        opportunities.append(opportunity)
        
        return opportunities
    
    def _analyze_calendar_spreads(self, options: List[OptionData]) -> List[ArbitrageOpportunity]:
        """
        Analyze calendar spread opportunities based on volatility differentials.
        
        Looks for options with same strike but different expiries where the
        volatility term structure suggests arbitrage opportunities.
        """
        opportunities = []
        
        # Group by strike price and option type
        strike_groups = {}
        for option in options:
            if option.implied_volatility is None:
                continue
                
            key = (option.strike_price, option.option_type)
            if key not in strike_groups:
                strike_groups[key] = []
            strike_groups[key].append(option)
        
        # Analyze each strike group for calendar spread opportunities
        for (strike_price, option_type), group_options in strike_groups.items():
            if len(group_options) < 2:
                continue
            
            # Sort by expiry date
            group_options.sort(key=lambda x: x.expiry_date)
            
            # Look for calendar spread opportunities
            for i in range(len(group_options) - 1):
                near_option = group_options[i]
                far_option = group_options[i + 1]
                
                days_diff = (far_option.expiry_date - near_option.expiry_date).days
                
                # Check if within calendar spread parameters
                if not (self.params.min_calendar_days <= days_diff <= self.params.max_calendar_days):
                    continue
                
                # Calculate expected volatility relationship
                # Typically, longer-term options should have lower or similar IV
                iv_spread = far_option.implied_volatility - near_option.implied_volatility
                
                # Look for anomalous term structure
                if abs(iv_spread) >= self.params.min_iv_spread:
                    
                    # Determine calendar spread direction
                    if iv_spread > 0:  # Far option has higher IV than near - unusual
                        # Sell far, buy near (sell time spread)
                        actions = [
                            TradingAction(near_option.code, ActionType.BUY, 1, near_option.market_price),
                            TradingAction(far_option.code, ActionType.SELL, 1, far_option.market_price)
                        ]
                        profit_potential = abs(iv_spread) * (near_option.market_price + far_option.market_price) / 2
                        reasoning = f"Inverted term structure: far IV {far_option.implied_volatility:.3f} > near IV {near_option.implied_volatility:.3f}"
                        
                    else:  # Near option has much higher IV than far - exploit time decay
                        # Sell near, buy far (buy time spread) 
                        actions = [
                            TradingAction(near_option.code, ActionType.SELL, 1, near_option.market_price),
                            TradingAction(far_option.code, ActionType.BUY, 1, far_option.market_price)
                        ]
                        profit_potential = abs(iv_spread) * (near_option.market_price + far_option.market_price) / 2
                        reasoning = f"Steep term structure: near IV {near_option.implied_volatility:.3f} >> far IV {far_option.implied_volatility:.3f}"
                    
                    # Calculate risk metrics
                    risk_metrics = self.assess_risk([near_option, far_option], actions)
                    
                    # Create opportunity
                    opportunity = ArbitrageOpportunity(
                        id=f"vol_calendar_{strike_price}_{near_option.expiry_date.strftime('%Y%m%d')}_{far_option.expiry_date.strftime('%Y%m%d')}",
                        strategy_type=self.strategy_type,
                        timestamp=datetime.now(),
                        instruments=[near_option.code, far_option.code],
                        underlying=near_option.underlying,
                        expected_profit=profit_potential,
                        profit_margin=abs(iv_spread),
                        confidence_score=0.0,  # Will be calculated later
                        max_loss=risk_metrics.max_loss,
                        risk_score=self._calculate_risk_score(risk_metrics),
                        days_to_expiry=min(near_option.days_to_expiry, far_option.days_to_expiry),
                        market_prices={
                            near_option.code: near_option.market_price,
                            far_option.code: far_option.market_price
                        },
                        volumes={
                            near_option.code: near_option.volume,
                            far_option.code: far_option.volume
                        },
                        actions=[{
                            'instrument': action.instrument,
                            'action': action.action.value,
                            'quantity': action.quantity,
                            'price': action.price,
                            'reasoning': reasoning
                        } for action in actions],
                        data_source="volatility_calendar_spread",
                        parameters={
                            'near_iv': near_option.implied_volatility,
                            'far_iv': far_option.implied_volatility,
                            'iv_spread': iv_spread,
                            'days_difference': days_diff,
                            'strategy_subtype': 'calendar_spread'
                        }
                    )
                    
                    opportunities.append(opportunity)
        
        return opportunities
    
    def _analyze_historical_volatility(self, options: List[OptionData]) -> List[ArbitrageOpportunity]:
        """
        Analyze historical vs implied volatility for statistical arbitrage.
        
        Note: This is a simplified implementation. In practice, you would need
        historical price data to calculate actual historical volatility.
        """
        opportunities = []
        
        # For demonstration, we'll use a simple heuristic
        # In practice, this would require historical price data
        for option in options:
            if option.implied_volatility is None:
                continue
            
            # Simplified historical volatility estimate (placeholder)
            # In practice, calculate from historical underlying prices
            estimated_hv = self._estimate_historical_volatility(option)
            
            if estimated_hv is None:
                continue
            
            hv_iv_spread = option.implied_volatility - estimated_hv
            
            # Check for significant spread
            if abs(hv_iv_spread) >= self.params.min_hv_iv_spread:
                
                if hv_iv_spread > 0:
                    # IV > HV - option is over-volatized, sell volatility
                    action_type = ActionType.SELL
                    profit_potential = hv_iv_spread * option.market_price
                    reasoning = f"IV {option.implied_volatility:.3f} > HV {estimated_hv:.3f}, overpriced volatility"
                else:
                    # HV > IV - option is under-volatized, buy volatility
                    action_type = ActionType.BUY
                    profit_potential = abs(hv_iv_spread) * option.market_price
                    reasoning = f"HV {estimated_hv:.3f} > IV {option.implied_volatility:.3f}, underpriced volatility"
                
                opportunity = self._create_volatility_opportunity(
                    option, estimated_hv, action_type, profit_potential,
                    "historical_volatility", reasoning
                )
                
                if opportunity:
                    opportunities.append(opportunity)
        
        return opportunities
    
    def _analyze_volatility_surface(self, options: List[OptionData]) -> List[ArbitrageOpportunity]:
        """
        Analyze the entire volatility surface for anomalies.
        
        This advanced analysis looks for options that deviate significantly
        from the expected volatility surface model.
        """
        opportunities = []
        
        # Filter options with valid IV data
        valid_options = [opt for opt in options if opt.implied_volatility is not None]
        
        if len(valid_options) < self.params.min_surface_points:
            return opportunities
        
        # Create surface points: (strike, time_to_expiry, implied_vol)
        surface_points = []
        for option in valid_options:
            surface_points.append({
                'strike': option.strike_price,
                'time_to_expiry': option.time_to_expiry,
                'implied_vol': option.implied_volatility,
                'option': option
            })
        
        # Simple surface analysis - look for outliers
        # In practice, this would use more sophisticated surface fitting
        vol_values = [point['implied_vol'] for point in surface_points]
        
        if len(vol_values) < 3:
            return opportunities
        
        mean_vol = statistics.mean(vol_values)
        vol_std = statistics.stdev(vol_values) if len(vol_values) > 1 else 0
        
        # Find outliers (options significantly different from surface average)
        for point in surface_points:
            option = point['option']
            vol_deviation = abs(point['implied_vol'] - mean_vol)
            
            # Check if this is a significant outlier
            if vol_std > 0 and vol_deviation > (2 * vol_std):  # 2-sigma outlier
                
                if point['implied_vol'] > mean_vol:
                    # Above average - sell volatility
                    action_type = ActionType.SELL
                    reasoning = f"Surface outlier: IV {point['implied_vol']:.3f} >> surface mean {mean_vol:.3f}"
                else:
                    # Below average - buy volatility
                    action_type = ActionType.BUY
                    reasoning = f"Surface outlier: IV {point['implied_vol']:.3f} << surface mean {mean_vol:.3f}"
                
                profit_potential = vol_deviation * option.market_price
                
                opportunity = self._create_volatility_opportunity(
                    option, mean_vol, action_type, profit_potential,
                    "surface_analysis", reasoning
                )
                
                if opportunity:
                    opportunities.append(opportunity)
        
        return opportunities
    
    def _estimate_historical_volatility(self, option: OptionData) -> Optional[float]:
        """
        Estimate historical volatility for the underlying asset.
        
        This is a placeholder implementation. In practice, you would:
        1. Fetch historical price data for the underlying
        2. Calculate returns over the specified window
        3. Annualize the volatility
        
        Args:
            option: Option to estimate HV for
            
        Returns:
            Estimated historical volatility or None
        """
        # Placeholder: Simple heuristic based on current IV
        # In practice, replace with actual HV calculation from price history
        
        if option.implied_volatility is None:
            return None
        
        # Simple heuristic: assume HV is typically 80% of long-term average IV
        # This is just for demonstration - replace with real HV calculation
        estimated_hv = option.implied_volatility * 0.8
        
        # Add some randomness to simulate real HV variations
        import random
        noise_factor = 1.0 + random.uniform(-0.2, 0.2)  # +/- 20% noise
        estimated_hv *= noise_factor
        
        return max(0.01, estimated_hv)  # Minimum 1% HV
    
    def _create_volatility_opportunity(
        self, 
        option: OptionData, 
        expected_value: float,
        action_type: ActionType,
        profit_potential: float,
        strategy_subtype: str,
        reasoning: str
    ) -> Optional[ArbitrageOpportunity]:
        """Create a volatility arbitrage opportunity."""
        
        try:
            # Create trading action
            action = TradingAction(
                instrument=option.code,
                action=action_type,
                quantity=1,
                price=option.market_price
            )
            
            # Calculate risk metrics
            risk_metrics = self.assess_risk([option], [action])
            
            # Create opportunity
            opportunity = ArbitrageOpportunity(
                id=f"vol_{strategy_subtype}_{option.code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                strategy_type=self.strategy_type,
                timestamp=datetime.now(),
                instruments=[option.code],
                underlying=option.underlying,
                expected_profit=profit_potential,
                profit_margin=abs(option.implied_volatility - expected_value) if option.implied_volatility else 0,
                confidence_score=0.0,  # Will be calculated later
                max_loss=risk_metrics.max_loss,
                risk_score=self._calculate_risk_score(risk_metrics),
                days_to_expiry=option.days_to_expiry,
                market_prices={option.code: option.market_price},
                volumes={option.code: option.volume},
                actions=[{
                    'instrument': action.instrument,
                    'action': action.action.value,
                    'quantity': action.quantity,
                    'price': action.price,
                    'reasoning': reasoning
                }],
                data_source="volatility_analysis",
                parameters={
                    'current_iv': option.implied_volatility,
                    'expected_value': expected_value,
                    'strategy_subtype': strategy_subtype,
                    'time_to_expiry': option.time_to_expiry
                }
            )
            
            return opportunity
            
        except Exception as e:
            self.logger.error(f"Error creating volatility opportunity: {e}")
            return None
    
    def calculate_profit(self, options: List[OptionData], actions: List[TradingAction]) -> float:
        """
        Calculate expected profit from volatility arbitrage.
        
        Args:
            options: Options involved
            actions: Trading actions
            
        Returns:
            Expected profit amount
        """
        total_profit = 0.0
        
        for action in actions:
            option = next((opt for opt in options if opt.code == action.instrument), None)
            if not option or option.implied_volatility is None:
                continue
            
            # Simplified profit calculation based on volatility edge
            # In practice, this would involve complex volatility pricing models
            
            position_value = (action.price or option.market_price) * action.quantity
            
            # Estimate profit based on volatility advantage
            # This is simplified - real implementation would use Greeks and vol sensitivity
            vol_edge = 0.01  # Assume 1% volatility edge on average
            
            if action.action == ActionType.BUY:
                # Long volatility position - profit from vol increase
                profit_per_unit = position_value * vol_edge
            else:
                # Short volatility position - profit from vol decrease
                profit_per_unit = position_value * vol_edge
            
            total_profit += profit_per_unit
        
        return total_profit
    
    def assess_risk(self, options: List[OptionData], actions: List[TradingAction]) -> RiskMetrics:
        """
        Assess risk for volatility arbitrage strategies.
        
        Args:
            options: Options involved
            actions: Trading actions
            
        Returns:
            Risk assessment metrics
        """
        max_loss = 0.0
        max_gain = 0.0
        total_position_value = 0.0
        
        volatility_risk = 0.0
        time_decay_risk = 0.0
        liquidity_risk = 0.0
        
        for action in actions:
            option = next((opt for opt in options if opt.code == action.instrument), None)
            if not option:
                continue
            
            position_value = (action.price or option.market_price) * action.quantity
            total_position_value += abs(position_value)
            
            # Volatility strategies have specific risk characteristics
            if action.action == ActionType.BUY:
                # Long option - max loss is premium paid
                max_loss += position_value
                # Max gain depends on volatility expansion
                max_gain += position_value * 3  # Assume max 3x return on vol expansion
            else:
                # Short option - max gain is premium received, loss potentially unlimited
                max_gain += position_value
                # For short vol positions, risk is higher
                max_loss += position_value * 5  # Conservative estimate
            
            # Volatility-specific risks
            if option.implied_volatility:
                volatility_risk = max(volatility_risk, option.implied_volatility)
            
            # Time decay risk is significant for volatility strategies
            if option.time_to_expiry > 0:
                time_decay_risk = max(time_decay_risk, 1.0 / math.sqrt(option.time_to_expiry * 365))
            
            # Liquidity risk from bid-ask spreads
            if option.spread_pct > 0:
                liquidity_risk = max(liquidity_risk, option.spread_pct)
        
        # Probability of profit for volatility strategies
        # This is simplified - real implementation would use historical success rates
        probability_profit = 0.55  # Slightly positive edge
        
        # Adjust based on risk factors
        if volatility_risk > 0.5:  # High volatility environment
            probability_profit *= 0.9
        if time_decay_risk > 1.0:  # High time decay
            probability_profit *= 0.85
        if liquidity_risk > 0.05:  # High spreads
            probability_profit *= 0.9
        
        expected_return = (probability_profit * max_gain) - ((1 - probability_profit) * max_loss)
        
        # Risk level assessment
        risk_ratio = max_loss / total_position_value if total_position_value > 0 else 1.0
        
        if risk_ratio < 0.1:
            risk_level = RiskLevel.LOW
        elif risk_ratio < 0.3:
            risk_level = RiskLevel.MEDIUM
        elif risk_ratio < 0.6:
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
        """Calculate normalized risk score from risk metrics."""
        score = 0.0
        
        # Risk level component (40% weight)
        risk_level_scores = {
            RiskLevel.LOW: 0.15,
            RiskLevel.MEDIUM: 0.35,
            RiskLevel.HIGH: 0.65,
            RiskLevel.CRITICAL: 0.9
        }
        score += risk_level_scores.get(risk_metrics.risk_level, 0.5) * 0.4
        
        # Volatility risk component (30% weight)
        score += min(risk_metrics.volatility_risk, 1.0) * 0.3
        
        # Time decay risk component (20% weight)
        score += min(risk_metrics.time_decay_risk, 1.0) * 0.2
        
        # Liquidity risk component (10% weight)
        score += min(risk_metrics.liquidity_risk, 1.0) * 0.1
        
        return min(score, 1.0)
    
    def filter_options(self, options_data: List[OptionData]) -> List[OptionData]:
        """
        Filter options for volatility arbitrage analysis.
        
        Args:
            options_data: Raw options data
            
        Returns:
            Filtered options suitable for volatility analysis
        """
        filtered = super().filter_options(options_data)
        
        result = []
        for option in filtered:
            # Must have implied volatility data
            if option.implied_volatility is None:
                continue
                
            # IV must be within reasonable bounds
            if not (self.params.min_iv_level <= option.implied_volatility <= self.params.max_iv_level):
                continue
            
            # Must have reasonable bid-ask spread for volatility trading
            if option.spread_pct > 0.15:  # Skip if spread > 15%
                continue
            
            result.append(option)
        
        return result