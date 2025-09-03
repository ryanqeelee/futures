#!/usr/bin/env python3
"""
Demonstration of the new interface system for options arbitrage scanning.
Shows how to use configurations, adapters, and strategies together.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, date

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import get_config_manager, ConfigManager
from adapters import TushareAdapter, DataRequest, AdapterRegistry
from strategies import (
    StrategyRegistry, PricingArbitrageStrategy, 
    PricingArbitrageParameters, LegacyIntegrationStrategy
)


async def demo_configuration_system():
    """Demonstrate configuration management."""
    print("=" * 60)
    print("Configuration System Demo")
    print("=" * 60)
    
    try:
        # Initialize configuration manager
        config_manager = ConfigManager()
        
        # Load configuration
        config = config_manager.load_config()
        
        print(f"✅ Loaded configuration for: {config.app_name} v{config.version}")
        print(f"📊 Environment: {config.environment}")
        print(f"🗂️  Data directory: {config.data_dir}")
        print(f"📁 Log directory: {config.log_dir}")
        
        # Show data sources
        enabled_sources = config_manager.get_enabled_data_sources()
        print(f"\n📡 Enabled data sources ({len(enabled_sources)}):")
        for name, ds_config in enabled_sources.items():
            print(f"  - {name}: {ds_config.type.value} (priority: {ds_config.priority})")
        
        # Show strategies
        enabled_strategies = config_manager.get_enabled_strategies()
        print(f"\n🎯 Enabled strategies ({len(enabled_strategies)}):")
        for name, strategy_config in enabled_strategies.items():
            print(f"  - {name}: {strategy_config.type.value} (priority: {strategy_config.priority})")
        
        return config_manager
        
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return None


async def demo_data_adapter():
    """Demonstrate data adapter usage."""
    print("\n" + "=" * 60)
    print("Data Adapter Demo")
    print("=" * 60)
    
    try:
        # Create Tushare adapter
        config = {
            'api_token': None,  # Will use environment variable
            'timeout': 30,
            'retry_count': 3
        }
        
        adapter = TushareAdapter(config)
        print(f"📡 Created adapter: {adapter}")
        
        # Connect to data source
        print("🔗 Connecting to Tushare...")
        await adapter.connect()
        
        if adapter.is_connected:
            print("✅ Successfully connected!")
            print(f"📊 Connection info: {adapter.connection_info.status}")
        else:
            print("❌ Connection failed")
            return None
        
        # Create a data request
        request = DataRequest(
            min_days_to_expiry=5,
            max_days_to_expiry=60,
            min_volume=50,
            include_iv=True,
            include_greeks=False
        )
        
        print(f"\n📋 Created data request:")
        print(f"  - Days to expiry: {request.min_days_to_expiry}-{request.max_days_to_expiry}")
        print(f"  - Minimum volume: {request.min_volume}")
        print(f"  - Include IV: {request.include_iv}")
        
        # Fetch data
        print("\n📥 Fetching option data...")
        response = await adapter.get_option_data(request)
        
        print(f"✅ Retrieved {response.record_count} option records")
        print(f"🔍 Data quality: {response.quality.value}")
        print(f"📅 Data timestamp: {response.timestamp}")
        print(f"🏷️  Data source: {response.source}")
        
        if response.data:
            sample_option = response.data[0]
            print(f"\n📄 Sample option:")
            print(f"  - Code: {sample_option.code}")
            print(f"  - Underlying: {sample_option.underlying}")
            print(f"  - Strike: {sample_option.strike_price}")
            print(f"  - Type: {sample_option.option_type.value}")
            print(f"  - Price: {sample_option.market_price}")
            print(f"  - Volume: {sample_option.volume}")
            print(f"  - Days to expiry: {sample_option.days_to_expiry}")
            if sample_option.implied_volatility:
                print(f"  - Implied Vol: {sample_option.implied_volatility:.2%}")
        
        await adapter.disconnect()
        return response.data
        
    except Exception as e:
        print(f"❌ Data adapter demo failed: {e}")
        return None


async def demo_strategy_system(options_data):
    """Demonstrate strategy system."""
    print("\n" + "=" * 60)
    print("Strategy System Demo") 
    print("=" * 60)
    
    if not options_data:
        print("❌ No options data available for strategy demo")
        return
    
    try:
        # Show registered strategies
        registered = StrategyRegistry.get_registered_strategies()
        print(f"🎯 Registered strategies ({len(registered)}):")
        for strategy_type, strategy_class in registered.items():
            print(f"  - {strategy_type.value}: {strategy_class.__name__}")
        
        # Create pricing arbitrage strategy
        print(f"\n🔧 Creating pricing arbitrage strategy...")
        pricing_params = PricingArbitrageParameters(
            min_price_deviation=0.05,  # 5% minimum deviation
            max_price_deviation=0.3,   # 30% maximum deviation
            min_profit_threshold=0.02, # 2% minimum profit
            max_risk_tolerance=0.15    # 15% maximum risk
        )
        
        pricing_strategy = PricingArbitrageStrategy(pricing_params)
        print(f"✅ Created: {pricing_strategy}")
        
        # Scan for opportunities
        print(f"\n🔍 Scanning for pricing arbitrage opportunities...")
        result = pricing_strategy.scan_opportunities(options_data)
        
        print(f"📊 Scan results:")
        print(f"  - Success: {result.success}")
        print(f"  - Execution time: {result.execution_time:.2f}s")
        print(f"  - Opportunities found: {len(result.opportunities)}")
        
        if result.opportunities:
            print(f"\n💰 Top opportunities:")
            for i, opp in enumerate(result.opportunities[:3], 1):
                print(f"  {i}. {opp.instruments[0]}")
                print(f"     Expected profit: ${opp.expected_profit:.2f}")
                print(f"     Profit margin: {opp.profit_margin:.2%}")
                print(f"     Risk score: {opp.risk_score:.2f}")
                print(f"     Confidence: {opp.confidence_score:.2%}")
                print(f"     Days to expiry: {opp.days_to_expiry}")
        
        # Try legacy integration strategy
        print(f"\n🔧 Testing legacy integration strategy...")
        legacy_strategy = LegacyIntegrationStrategy()
        legacy_result = legacy_strategy.scan_opportunities(options_data)
        
        print(f"📊 Legacy integration results:")
        print(f"  - Success: {legacy_result.success}")
        print(f"  - Execution time: {legacy_result.execution_time:.2f}s")
        print(f"  - Opportunities found: {len(legacy_result.opportunities)}")
        
        if legacy_result.error_message:
            print(f"  - Error: {legacy_result.error_message}")
        
        return result.opportunities + legacy_result.opportunities
        
    except Exception as e:
        print(f"❌ Strategy demo failed: {e}")
        return []


async def demo_complete_workflow():
    """Demonstrate complete arbitrage scanning workflow."""
    print("\n" + "=" * 60)
    print("Complete Workflow Demo")
    print("=" * 60)
    
    try:
        # 1. Configuration
        config_manager = await demo_configuration_system()
        if not config_manager:
            return
        
        # 2. Data retrieval  
        options_data = await demo_data_adapter()
        if not options_data:
            return
        
        # 3. Strategy execution
        opportunities = await demo_strategy_system(options_data)
        
        # 4. Summary
        print(f"\n📋 Workflow Summary:")
        print(f"  - Options analyzed: {len(options_data)}")
        print(f"  - Total opportunities: {len(opportunities)}")
        
        if opportunities:
            total_expected_profit = sum(opp.expected_profit for opp in opportunities)
            avg_confidence = sum(opp.confidence_score for opp in opportunities) / len(opportunities)
            
            print(f"  - Total expected profit: ${total_expected_profit:.2f}")
            print(f"  - Average confidence: {avg_confidence:.2%}")
            
            # Group by strategy type
            by_strategy = {}
            for opp in opportunities:
                strategy_type = opp.strategy_type.value
                if strategy_type not in by_strategy:
                    by_strategy[strategy_type] = []
                by_strategy[strategy_type].append(opp)
            
            print(f"  - Opportunities by strategy:")
            for strategy, ops in by_strategy.items():
                print(f"    * {strategy}: {len(ops)} opportunities")
        
        print(f"\n✅ Workflow completed successfully!")
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")


async def main():
    """Main demo function."""
    print("🚀 Options Arbitrage Interface System Demo")
    print(f"⏰ Start time: {datetime.now()}")
    
    try:
        await demo_complete_workflow()
    except KeyboardInterrupt:
        print(f"\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n⏰ End time: {datetime.now()}")
    print("👋 Demo completed")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())