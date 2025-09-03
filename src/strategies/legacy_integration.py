"""
Legacy integration layer for bridging new interfaces with existing arbitrage logic.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# Add legacy logic to path
legacy_path = Path(__file__).parent.parent.parent / "legacy_logic"
if legacy_path.exists():
    sys.path.append(str(legacy_path))

from ..config.models import StrategyType, ArbitrageOpportunity
from .base import (
    BaseStrategy, StrategyResult, StrategyParameters, OptionData,
    TradingAction, ActionType, RiskMetrics, RiskLevel, StrategyRegistry
)


class LegacyIntegrationStrategy(BaseStrategy):
    """
    Strategy that wraps legacy arbitrage logic with new interface.
    
    This allows existing proven algorithms to work with the new
    plugin architecture while maintaining backward compatibility.
    """
    
    def __init__(self, parameters: Optional[StrategyParameters] = None):
        super().__init__(parameters or StrategyParameters())
        self._legacy_functions = self._import_legacy_functions()
        
    @property 
    def strategy_type(self) -> StrategyType:
        return StrategyType.PRICING_ARBITRAGE  # Default, can be overridden
    
    def _import_legacy_functions(self) -> Dict[str, Any]:
        """Import legacy functions safely."""
        functions = {}
        
        try:
            import option_arbitrage_scanner as legacy_scanner
            
            functions.update({
                'black_scholes_call': getattr(legacy_scanner, 'black_scholes_call', None),
                'black_scholes_put': getattr(legacy_scanner, 'black_scholes_put', None),
                'implied_volatility': getattr(legacy_scanner, 'implied_volatility', None),
                'find_pricing_arbitrage': getattr(legacy_scanner, 'find_pricing_arbitrage', None),
                'find_put_call_parity_arbitrage': getattr(legacy_scanner, 'find_put_call_parity_arbitrage', None),
                'find_volatility_arbitrage': getattr(legacy_scanner, 'find_volatility_arbitrage', None),
                'find_calendar_spread_arbitrage': getattr(legacy_scanner, 'find_calendar_spread_arbitrage', None)
            })
        except ImportError:
            print("Warning: Legacy option_arbitrage_scanner module not found")
        
        return functions
    
    def convert_option_data_to_legacy_format(self, options_data: List[OptionData]) -> pd.DataFrame:
        """
        Convert OptionData list to legacy DataFrame format.
        
        Args:
            options_data: List of OptionData objects
            
        Returns:
            pd.DataFrame: Legacy format DataFrame
        """
        if not options_data:
            return pd.DataFrame()
        
        records = []
        for option in options_data:
            record = {
                'ts_code': option.code,
                'name': option.name,
                'underlying': option.underlying,
                'call_put': option.option_type.value,
                'exercise_price': option.strike_price,
                'delist_date': option.expiry_date.strftime('%Y%m%d'),
                'days_to_expiry': option.days_to_expiry,
                'close': option.market_price,
                'vol': option.volume,
                'oi': option.open_interest,
                'underlying_price': self._estimate_underlying_price(option, options_data),
                'theoretical_price': option.theoretical_price,
                'implied_volatility': option.implied_volatility,
                'delta': option.delta,
                'gamma': option.gamma,
                'theta': option.theta,
                'vega': option.vega
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def _estimate_underlying_price(self, target_option: OptionData, all_options: List[OptionData]) -> float:
        """Estimate underlying price using put-call parity or other methods."""
        # First check if we have underlying price data
        if hasattr(target_option, 'underlying_price') and target_option.underlying_price:
            return target_option.underlying_price
        
        # Try put-call parity with same underlying and strike
        same_underlying = [opt for opt in all_options if opt.underlying == target_option.underlying]
        same_strike = [opt for opt in same_underlying if abs(opt.strike_price - target_option.strike_price) < 0.01]
        
        if len(same_strike) >= 2:
            calls = [opt for opt in same_strike if opt.option_type.value == 'C']
            puts = [opt for opt in same_strike if opt.option_type.value == 'P']
            
            if calls and puts:
                call_price = calls[0].market_price
                put_price = puts[0].market_price
                strike = target_option.strike_price
                
                # S ≈ C - P + K (simplified, assuming r=0)
                estimated_price = call_price - put_price + strike
                if estimated_price > 0:
                    return estimated_price
        
        # Fallback: use strike price as rough estimate
        return target_option.strike_price
    
    def convert_legacy_opportunities_to_new_format(self, legacy_opportunities: List[Dict], strategy_type: StrategyType) -> List[ArbitrageOpportunity]:
        """
        Convert legacy opportunity format to new ArbitrageOpportunity objects.
        
        Args:
            legacy_opportunities: List of legacy opportunity dicts
            strategy_type: Type of strategy that found these opportunities
            
        Returns:
            List[ArbitrageOpportunity]: Converted opportunities
        """
        converted = []
        
        for legacy_op in legacy_opportunities:
            try:
                # Extract common fields
                instruments = []
                market_prices = {}
                theoretical_prices = {}
                volumes = {}
                actions = []
                
                # Handle different legacy opportunity formats
                if 'code' in legacy_op:
                    # Single instrument opportunity (pricing arbitrage)
                    instruments = [legacy_op['code']]
                    market_prices[legacy_op['code']] = legacy_op.get('market_price', 0)
                    if 'theoretical_price' in legacy_op:
                        theoretical_prices[legacy_op['code']] = legacy_op['theoretical_price']
                    if 'volume' in legacy_op:
                        volumes[legacy_op['code']] = legacy_op['volume']
                    
                    actions.append({
                        'instrument': legacy_op['code'],
                        'action': legacy_op.get('action', 'UNKNOWN'),
                        'quantity': 1,
                        'reasoning': f"Legacy {strategy_type.value} opportunity"
                    })
                    
                elif 'call_code' in legacy_op and 'put_code' in legacy_op:
                    # Two instrument opportunity (parity arbitrage)
                    instruments = [legacy_op['call_code'], legacy_op['put_code']]
                    market_prices[legacy_op['call_code']] = legacy_op.get('call_price', 0)
                    market_prices[legacy_op['put_code']] = legacy_op.get('put_price', 0)
                    
                    if 'call_vol' in legacy_op:
                        volumes[legacy_op['call_code']] = legacy_op['call_vol']
                    if 'put_vol' in legacy_op:
                        volumes[legacy_op['put_code']] = legacy_op['put_vol']
                    
                    # Parse action string to determine individual actions
                    action_str = legacy_op.get('action', '')
                    if '买入看涨' in action_str or 'buy call' in action_str.lower():
                        actions.append({
                            'instrument': legacy_op['call_code'],
                            'action': 'BUY',
                            'quantity': 1
                        })
                    if '卖出看涨' in action_str or 'sell call' in action_str.lower():
                        actions.append({
                            'instrument': legacy_op['call_code'],  
                            'action': 'SELL',
                            'quantity': 1
                        })
                    if '买入看跌' in action_str or 'buy put' in action_str.lower():
                        actions.append({
                            'instrument': legacy_op['put_code'],
                            'action': 'BUY', 
                            'quantity': 1
                        })
                    if '卖出看跌' in action_str or 'sell put' in action_str.lower():
                        actions.append({
                            'instrument': legacy_op['put_code'],
                            'action': 'SELL',
                            'quantity': 1
                        })
                
                elif 'near_code' in legacy_op and 'far_code' in legacy_op:
                    # Calendar spread opportunity
                    instruments = [legacy_op['near_code'], legacy_op['far_code']]
                    market_prices[legacy_op['near_code']] = legacy_op.get('near_price', 0)
                    market_prices[legacy_op['far_code']] = legacy_op.get('far_price', 0)
                    
                    # Parse calendar spread actions
                    action_str = legacy_op.get('action', '')
                    if '买入远月' in action_str:
                        actions.append({'instrument': legacy_op['far_code'], 'action': 'BUY', 'quantity': 1})
                    if '卖出近月' in action_str:
                        actions.append({'instrument': legacy_op['near_code'], 'action': 'SELL', 'quantity': 1})
                    if '卖出远月' in action_str:
                        actions.append({'instrument': legacy_op['far_code'], 'action': 'SELL', 'quantity': 1})
                    if '买入近月' in action_str:
                        actions.append({'instrument': legacy_op['near_code'], 'action': 'BUY', 'quantity': 1})
                
                # Calculate profit and risk metrics
                expected_profit = legacy_op.get('potential_profit', 0)
                if isinstance(expected_profit, str):
                    try:
                        expected_profit = float(expected_profit)
                    except:
                        expected_profit = 0
                
                # Extract profit margin from various fields
                profit_margin = 0
                if 'deviation' in legacy_op:
                    deviation_str = str(legacy_op['deviation'])
                    if '%' in deviation_str:
                        try:
                            profit_margin = abs(float(deviation_str.replace('%', ''))) / 100
                        except:
                            profit_margin = 0.05  # Default 5%
                elif 'relative_error' in legacy_op:
                    error_str = str(legacy_op['relative_error'])
                    if '%' in error_str:
                        try:
                            profit_margin = float(error_str.replace('%', '')) / 100
                        except:
                            profit_margin = 0.05
                
                # Estimate max loss (simplified)
                max_loss = sum(market_prices.values()) * 0.5  # Rough estimate
                
                # Create opportunity
                opportunity = ArbitrageOpportunity(
                    id=f"legacy_{strategy_type.value}_{len(converted)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    strategy_type=strategy_type,
                    timestamp=datetime.now(),
                    instruments=instruments,
                    underlying=legacy_op.get('underlying', instruments[0].split('.')[0] if instruments else 'UNKNOWN'),
                    expected_profit=expected_profit,
                    profit_margin=profit_margin,
                    confidence_score=0.7,  # Default moderate confidence for legacy
                    max_loss=max_loss,
                    risk_score=0.3,  # Default moderate risk
                    days_to_expiry=legacy_op.get('days_to_expiry', 30),
                    market_prices=market_prices,
                    theoretical_prices=theoretical_prices,
                    volumes=volumes,
                    actions=actions,
                    data_source="legacy_scanner",
                    parameters=legacy_op.copy()
                )
                
                converted.append(opportunity)
                
            except Exception as e:
                print(f"Warning: Failed to convert legacy opportunity: {e}")
                continue
        
        return converted
    
    def scan_opportunities(self, options_data: List[OptionData]) -> StrategyResult:
        """
        Scan for opportunities using legacy algorithms.
        
        Args:
            options_data: List of option market data
            
        Returns:
            StrategyResult: Results with found opportunities
        """
        start_time = datetime.now()
        all_opportunities = []
        
        try:
            # Convert to legacy format
            legacy_df = self.convert_option_data_to_legacy_format(options_data)
            
            if legacy_df.empty:
                return StrategyResult(
                    strategy_name=self.name,
                    opportunities=[],
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    data_timestamp=datetime.now(),
                    success=True,
                    metadata={'message': 'No data to analyze'}
                )
            
            # Run legacy algorithms if available
            if self._legacy_functions.get('find_pricing_arbitrage'):
                try:
                    pricing_ops = self._legacy_functions['find_pricing_arbitrage'](
                        legacy_df, 
                        min_deviation=self.parameters.min_profit_threshold
                    )
                    converted_pricing = self.convert_legacy_opportunities_to_new_format(
                        pricing_ops, StrategyType.PRICING_ARBITRAGE
                    )
                    all_opportunities.extend(converted_pricing)
                except Exception as e:
                    print(f"Warning: Pricing arbitrage scan failed: {e}")
            
            if self._legacy_functions.get('find_put_call_parity_arbitrage'):
                try:
                    parity_ops = self._legacy_functions['find_put_call_parity_arbitrage'](
                        legacy_df,
                        tolerance=self.parameters.max_risk_tolerance
                    )
                    converted_parity = self.convert_legacy_opportunities_to_new_format(
                        parity_ops, StrategyType.PUT_CALL_PARITY
                    )
                    all_opportunities.extend(converted_parity)
                except Exception as e:
                    print(f"Warning: Put-call parity scan failed: {e}")
            
            if self._legacy_functions.get('find_volatility_arbitrage'):
                try:
                    vol_ops = self._legacy_functions['find_volatility_arbitrage'](legacy_df)
                    converted_vol = self.convert_legacy_opportunities_to_new_format(
                        vol_ops, StrategyType.VOLATILITY_ARBITRAGE
                    )
                    all_opportunities.extend(converted_vol)
                except Exception as e:
                    print(f"Warning: Volatility arbitrage scan failed: {e}")
            
            if self._legacy_functions.get('find_calendar_spread_arbitrage'):
                try:
                    calendar_ops = self._legacy_functions['find_calendar_spread_arbitrage'](legacy_df)
                    converted_calendar = self.convert_legacy_opportunities_to_new_format(
                        calendar_ops, StrategyType.CALENDAR_SPREAD
                    )
                    all_opportunities.extend(converted_calendar)
                except Exception as e:
                    print(f"Warning: Calendar spread scan failed: {e}")
            
            # Filter opportunities using new validation
            validated_opportunities = [
                opp for opp in all_opportunities
                if self.validate_opportunity(opp)
            ]
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return StrategyResult(
                strategy_name=self.name,
                opportunities=validated_opportunities,
                execution_time=execution_time,
                data_timestamp=datetime.now(),
                success=True,
                metadata={
                    'total_options_analyzed': len(options_data),
                    'legacy_opportunities_found': len(all_opportunities),
                    'validated_opportunities': len(validated_opportunities),
                    'legacy_functions_available': len([f for f in self._legacy_functions.values() if f is not None])
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
                error_message=f"Legacy integration failed: {e}"
            )
    
    def calculate_profit(self, options: List[OptionData], actions: List[TradingAction]) -> float:
        """Calculate profit using legacy methods when available."""
        # Fallback to simple calculation if legacy methods not available
        total_profit = 0.0
        
        for action in actions:
            option = next((opt for opt in options if opt.code == action.instrument), None)
            if not option:
                continue
            
            position_value = (action.price or option.market_price) * action.quantity
            
            if action.action == ActionType.BUY:
                # Rough profit estimate for buying - assume 10% gain potential
                total_profit += position_value * 0.1
            else:
                # Rough profit estimate for selling - assume premium capture
                total_profit += position_value * 0.05
        
        return total_profit
    
    def assess_risk(self, options: List[OptionData], actions: List[TradingAction]) -> RiskMetrics:
        """Assess risk using simplified heuristics."""
        max_loss = 0.0
        max_gain = 0.0
        
        for action in actions:
            option = next((opt for opt in options if opt.code == action.instrument), None)
            if not option:
                continue
            
            position_value = (action.price or option.market_price) * action.quantity
            
            if action.action == ActionType.BUY:
                max_loss += position_value  # Long option - max loss is premium
                max_gain += position_value * 5  # Conservative gain estimate
            else:
                max_gain += position_value  # Short option - max gain is premium  
                max_loss += position_value * 10  # Conservative loss estimate for short options
        
        return RiskMetrics(
            max_loss=max_loss,
            max_gain=max_gain,
            probability_profit=0.6,  # Conservative estimate
            expected_return=max_gain * 0.6 - max_loss * 0.4,
            risk_level=RiskLevel.MEDIUM,
            liquidity_risk=0.2,
            time_decay_risk=0.3,
            volatility_risk=0.25
        )