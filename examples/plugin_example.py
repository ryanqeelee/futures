#!/usr/bin/env python3
"""
Example demonstrating plugin architecture for custom strategies and adapters.
Shows how to extend the system with custom implementations.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.models import StrategyType, DataSourceType, ArbitrageOpportunity
from strategies.base import (
    BaseStrategy, StrategyResult, StrategyParameters, OptionData,
    TradingAction, ActionType, RiskMetrics, RiskLevel, StrategyRegistry
)
from adapters.base import (
    BaseDataAdapter, DataRequest, DataResponse, DataQuality,
    ConnectionStatus, AdapterRegistry
)


# ============================================================================
# Custom Strategy Example
# ============================================================================

class CustomArbitrageParameters(StrategyParameters):
    """Parameters for our custom strategy."""
    momentum_threshold: float = 0.1  # 10% momentum threshold
    volume_spike_ratio: float = 2.0  # 2x normal volume
    correlation_threshold: float = 0.8  # 80% correlation


@StrategyRegistry.register(StrategyType.VOLATILITY_ARBITRAGE)
class MomentumVolatilityStrategy(BaseStrategy):
    """
    Custom strategy that combines momentum and volatility analysis.
    
    This is an example of how to create a custom strategy by inheriting
    from BaseStrategy and implementing the required methods.
    """
    
    def __init__(self, parameters: Optional[CustomArbitrageParameters] = None):
        super().__init__(parameters or CustomArbitrageParameters())
        self.custom_params = self.parameters
    
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.VOLATILITY_ARBITRAGE
    
    def scan_opportunities(self, options_data: List[OptionData]) -> StrategyResult:
        """Scan for momentum-volatility arbitrage opportunities."""
        start_time = datetime.now()
        opportunities = []
        
        try:
            filtered_options = self.filter_options(options_data)
            
            # Group options by underlying
            by_underlying = {}
            for option in filtered_options:
                if option.underlying not in by_underlying:
                    by_underlying[option.underlying] = []
                by_underlying[option.underlying].append(option)
            
            # Analyze each underlying for momentum-volatility patterns
            for underlying, options in by_underlying.items():
                underlying_opportunities = self._analyze_underlying_momentum(underlying, options)
                opportunities.extend(underlying_opportunities)
            
            # Validate opportunities
            validated_opportunities = [
                opp for opp in opportunities
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
                    'underlyings_analyzed': len(by_underlying),
                    'total_options': len(filtered_options),
                    'raw_opportunities': len(opportunities),
                    'validated_opportunities': len(validated_opportunities)
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
    
    def _analyze_underlying_momentum(self, underlying: str, options: List[OptionData]) -> List[ArbitrageOpportunity]:
        """Analyze momentum patterns for an underlying."""
        opportunities = []
        
        # Simple momentum analysis (in real implementation, you'd use historical data)
        high_volume_options = [opt for opt in options if opt.volume > 1000]
        
        if len(high_volume_options) < 2:
            return opportunities
        
        # Look for volume spikes combined with IV anomalies
        avg_volume = sum(opt.volume for opt in options) / len(options)
        
        for option in high_volume_options:
            volume_ratio = option.volume / avg_volume if avg_volume > 0 else 0
            
            # Check for volume spike
            if volume_ratio >= self.custom_params.volume_spike_ratio:
                # Check for IV anomaly (simplified)
                if option.implied_volatility and option.implied_volatility > 0.3:  # High IV
                    
                    opportunity = ArbitrageOpportunity(
                        id=f"momentum_vol_{option.code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        strategy_type=self.strategy_type,
                        timestamp=datetime.now(),
                        instruments=[option.code],
                        underlying=underlying,
                        expected_profit=option.market_price * 0.1,  # Estimate 10% profit potential
                        profit_margin=volume_ratio * 0.05,  # Profit based on volume spike
                        confidence_score=min(volume_ratio / 10, 0.9),  # Confidence from volume
                        max_loss=option.market_price,  # Max loss is premium for long options
                        risk_score=0.4,  # Moderate risk
                        days_to_expiry=option.days_to_expiry,
                        market_prices={option.code: option.market_price},
                        volumes={option.code: option.volume},
                        actions=[{
                            'instrument': option.code,
                            'action': 'BUY',
                            'quantity': 1,
                            'reasoning': f"Volume spike {volume_ratio:.1f}x with high IV {option.implied_volatility:.1%}"
                        }],
                        data_source="custom_momentum_strategy",
                        parameters={
                            'volume_ratio': volume_ratio,
                            'avg_volume': avg_volume,
                            'implied_volatility': option.implied_volatility
                        }
                    )
                    
                    opportunities.append(opportunity)
        
        return opportunities
    
    def calculate_profit(self, options: List[OptionData], actions: List[TradingAction]) -> float:
        """Calculate expected profit from momentum-volatility play."""
        total_profit = 0.0
        
        for action in actions:
            option = next((opt for opt in options if opt.code == action.instrument), None)
            if not option:
                continue
            
            # Estimate profit based on expected volatility expansion
            if option.implied_volatility:
                vol_expansion_profit = option.market_price * option.implied_volatility * 0.5
                total_profit += vol_expansion_profit * action.quantity
        
        return total_profit
    
    def assess_risk(self, options: List[OptionData], actions: List[TradingAction]) -> RiskMetrics:
        """Assess risk for momentum-volatility strategy."""
        max_loss = sum((action.price or 0) * action.quantity for action in actions if action.action == ActionType.BUY)
        max_gain = max_loss * 3  # Conservative 3:1 gain potential
        
        return RiskMetrics(
            max_loss=max_loss,
            max_gain=max_gain,
            probability_profit=0.65,  # Moderate probability
            expected_return=max_gain * 0.65 - max_loss * 0.35,
            risk_level=RiskLevel.MEDIUM,
            liquidity_risk=0.2,
            time_decay_risk=0.4,  # Higher time decay risk
            volatility_risk=0.6   # High volatility risk (but that's our edge)
        )


# ============================================================================
# Custom Data Adapter Example  
# ============================================================================

@AdapterRegistry.register(DataSourceType.MOCK)
class MockDataAdapter(BaseDataAdapter):
    """
    Mock data adapter for testing and development.
    
    Generates synthetic option data for testing strategies without
    requiring real market data connections.
    """
    
    def __init__(self, config: Dict[str, Any], name: Optional[str] = None):
        super().__init__(config, name or "MockDataAdapter")
        self.option_count = config.get('option_count', 50)
        self.seed = config.get('seed', 42)
        self.generate_random = config.get('generate_random_data', True)
    
    @property
    def data_source_type(self) -> DataSourceType:
        return DataSourceType.MOCK
    
    async def connect(self) -> None:
        """Mock connection - always succeeds."""
        import time
        time.sleep(0.1)  # Simulate connection time
        self._update_connection_status(ConnectionStatus.CONNECTED)
    
    async def disconnect(self) -> None:
        """Mock disconnection."""
        self._update_connection_status(ConnectionStatus.DISCONNECTED)
    
    async def get_option_data(self, request: DataRequest) -> DataResponse:
        """Generate mock option data."""
        import random
        import numpy as np
        
        if self.generate_random:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        options_data = []
        
        # Generate mock data
        underlyings = ['ETF50', 'HS300', 'CU', 'AU', 'ZN']
        option_types = ['C', 'P']
        
        for i in range(self.option_count):
            underlying = random.choice(underlyings)
            option_type = random.choice(option_types)
            
            # Generate realistic option parameters
            strike = random.uniform(2500, 4500)
            days_to_expiry = random.randint(
                request.min_days_to_expiry or 1,
                request.max_days_to_expiry or 90
            )
            
            market_price = random.uniform(10, 200)
            volume = random.randint(request.min_volume or 0, 10000)
            
            # Skip if volume too low
            if request.min_volume and volume < request.min_volume:
                continue
            
            option = OptionData(
                code=f"{underlying}{strike:.0f}{option_type}{days_to_expiry:02d}.SH",
                name=f"{underlying} {strike} {option_type}",
                underlying=underlying,
                option_type=option_type,
                strike_price=strike,
                expiry_date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
                market_price=market_price,
                bid_price=market_price * 0.98,
                ask_price=market_price * 1.02,
                volume=volume,
                open_interest=random.randint(0, 50000),
                implied_volatility=random.uniform(0.1, 0.8) if request.include_iv else None,
                theoretical_price=market_price * random.uniform(0.95, 1.05),
                delta=random.uniform(-1, 1) if request.include_greeks else None,
                gamma=random.uniform(0, 0.1) if request.include_greeks else None,
                theta=random.uniform(-5, 0) if request.include_greeks else None,
                vega=random.uniform(0, 20) if request.include_greeks else None
            )
            
            options_data.append(option)
        
        return DataResponse(
            request=request,
            data=options_data,
            source=self.name,
            quality=DataQuality.HIGH,
            metadata={
                'generated_count': len(options_data),
                'seed_used': self.seed,
                'random_generation': self.generate_random
            }
        )
    
    async def get_underlying_price(self, symbol: str, as_of_date=None) -> Optional[float]:
        """Generate mock underlying price."""
        import random
        
        # Mock prices for common underlyings
        mock_prices = {
            'ETF50': random.uniform(2.8, 3.2),
            'HS300': random.uniform(3800, 4200),
            'CU': random.uniform(45000, 55000),
            'AU': random.uniform(380, 420),
            'ZN': random.uniform(20000, 25000)
        }
        
        return mock_prices.get(symbol, random.uniform(1000, 5000))


# ============================================================================
# Plugin Demo Function
# ============================================================================

async def demo_custom_plugins():
    """Demonstrate custom strategy and adapter plugins."""
    print("🔌 Custom Plugin Architecture Demo")
    print("=" * 60)
    
    # 1. Show registered plugins
    print("📋 Registered Strategies:")
    strategies = StrategyRegistry.get_registered_strategies()
    for strategy_type, strategy_class in strategies.items():
        print(f"  - {strategy_type.value}: {strategy_class.__name__}")
    
    print("\n📋 Registered Adapters:")
    adapters = AdapterRegistry.get_registered_adapters()
    for adapter_type, adapter_class in adapters.items():
        print(f"  - {adapter_type.value}: {adapter_class.__name__}")
    
    # 2. Test custom mock adapter
    print(f"\n🧪 Testing Mock Data Adapter...")
    mock_config = {
        'option_count': 20,
        'seed': 123,
        'generate_random_data': True
    }
    
    mock_adapter = MockDataAdapter(mock_config)
    await mock_adapter.connect()
    
    request = DataRequest(
        min_days_to_expiry=5,
        max_days_to_expiry=30,
        min_volume=100,
        include_iv=True,
        include_greeks=True
    )
    
    mock_response = await mock_adapter.get_option_data(request)
    print(f"✅ Generated {mock_response.record_count} mock options")
    
    # 3. Test custom strategy
    print(f"\n🎯 Testing Custom Momentum-Volatility Strategy...")
    custom_params = CustomArbitrageParameters(
        momentum_threshold=0.1,
        volume_spike_ratio=1.5,
        min_profit_threshold=0.05
    )
    
    momentum_strategy = MomentumVolatilityStrategy(custom_params)
    strategy_result = momentum_strategy.scan_opportunities(mock_response.data)
    
    print(f"📊 Strategy Results:")
    print(f"  - Success: {strategy_result.success}")
    print(f"  - Opportunities: {len(strategy_result.opportunities)}")
    print(f"  - Execution time: {strategy_result.execution_time:.3f}s")
    
    if strategy_result.opportunities:
        opp = strategy_result.opportunities[0]
        print(f"  - Sample opportunity:")
        print(f"    * Instrument: {opp.instruments[0]}")
        print(f"    * Expected profit: ${opp.expected_profit:.2f}")
        print(f"    * Confidence: {opp.confidence_score:.1%}")
    
    await mock_adapter.disconnect()
    
    print(f"\n✅ Plugin demo completed successfully!")


if __name__ == "__main__":
    import asyncio
    
    print("🚀 Plugin Architecture Example")
    print(f"⏰ Start time: {datetime.now()}")
    
    try:
        asyncio.run(demo_custom_plugins())
    except Exception as e:
        print(f"❌ Plugin demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n⏰ End time: {datetime.now()}")
    print("👋 Plugin demo completed")