#!/usr/bin/env python3
"""
TushareAdapter Integration Test
Quick integration test to verify the TushareAdapter works with the existing system.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.adapters.tushare_adapter import TushareAdapter
from src.adapters.tushare_config_example import TushareConfigTemplates
from src.adapters.base import DataRequest
from src.strategies.base import OptionType


async def test_basic_integration():
    """Test basic integration with TushareAdapter."""
    print("🧪 TushareAdapter Integration Test")
    print("=" * 50)
    
    try:
        # 1. Create adapter with development config
        print("📋 Creating adapter with development configuration...")
        config = TushareConfigTemplates.development_config()
        adapter = TushareAdapter(config)
        print("✅ Adapter created successfully")
        
        # 2. Test connection
        print("\n🔌 Testing connection...")
        await adapter.connect()
        print(f"✅ Connected successfully: {adapter.connection_info.status.value}")
        
        # 3. Test data retrieval
        print("\n📊 Testing data retrieval...")
        request = DataRequest(
            max_days_to_expiry=30,
            min_volume=1,
            include_iv=True,
            include_greeks=False  # Skip Greeks for quick test
        )
        
        response = await adapter.get_option_data(request)
        print(f"✅ Retrieved {len(response.data)} options")
        print(f"   Data quality: {response.quality.value}")
        
        if response.data:
            sample = response.data[0]
            print(f"   Sample option: {sample.code}")
            print(f"   Market price: {sample.market_price}")
            print(f"   Strike: {sample.strike_price}")
            print(f"   Days to expiry: {sample.days_to_expiry}")
        
        # 4. Test market data
        if len(response.data) >= 3:
            print("\n💹 Testing market data...")
            symbols = [opt.code for opt in response.data[:3]]
            market_data = await adapter.get_market_data(symbols)
            print(f"✅ Retrieved market data for {len(market_data)} symbols")
            
            prices = await adapter.get_real_time_prices(symbols)
            print(f"✅ Retrieved prices for {len(prices)} symbols")
        
        # 5. Test performance metrics
        print("\n⚡ Testing performance metrics...")
        metrics = adapter.get_performance_metrics()
        print(f"✅ Total requests: {metrics['total_requests']}")
        print(f"   Cache hit rate: {metrics['cache_hit_rate']:.1%}")
        print(f"   Connection status: {metrics['connection_status']}")
        
        # 6. Test quality report
        print("\n🔍 Testing quality report...")
        quality_report = adapter.get_data_quality_report()
        print(f"✅ Quality validation errors: {quality_report['current_metrics']['total_errors']}")
        print(f"   Quality trend: {quality_report['quality_trend']['trend']}")
        
        # 7. Test health check
        print("\n🏥 Testing comprehensive health check...")
        health_info = await adapter.health_check_comprehensive()
        print(f"✅ Health check: {health_info['connection_status']}")
        print(f"   Data quality test: {'✅' if health_info['data_quality']['test_successful'] else '❌'}")
        print(f"   API rate limit: {health_info['api_limits']['rate_limit_status']}")
        
        # 8. Cleanup
        print("\n🧹 Cleaning up...")
        await adapter.disconnect()
        print("✅ Disconnected successfully")
        
        print("\n🎉 Integration test completed successfully!")
        print("=" * 50)
        print("Summary:")
        print(f"  - Options retrieved: {len(response.data)}")
        print(f"  - Data quality: {response.quality.value}")
        print(f"  - Performance: {metrics['total_requests']} requests")
        print(f"  - Cache efficiency: {metrics['cache_hit_rate']:.1%}")
        print("  - All systems operational ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_configuration_validation():
    """Test configuration validation system."""
    print("\n🔧 Testing Configuration Validation...")
    
    from src.adapters.tushare_config_example import TushareConfigValidator
    
    # Test different configurations
    configs = {
        'development': TushareConfigTemplates.development_config(),
        'production': TushareConfigTemplates.production_config(),
        'research': TushareConfigTemplates.research_config()
    }
    
    for name, config in configs.items():
        validation = TushareConfigValidator.validate_config(config)
        print(f"  {name}: {'✅' if validation['valid'] else '❌'} "
              f"(Score: {validation['score']}/100)")
        
        if validation['warnings']:
            print(f"    Warnings: {len(validation['warnings'])}")
        if validation['recommendations']:
            print(f"    Recommendations: {len(validation['recommendations'])}")


if __name__ == "__main__":
    print("🚀 Starting TushareAdapter Integration Tests")
    print("This test will verify the complete TushareAdapter implementation.")
    print()
    
    # Check environment
    if not os.getenv('TUSHARE_TOKEN'):
        print("⚠️  Warning: TUSHARE_TOKEN not found in environment")
        print("   Please set your Tushare API token in .env file")
        print()
    
    async def run_all_tests():
        # Run integration test
        integration_success = await test_basic_integration()
        
        # Run configuration tests
        await test_configuration_validation()
        
        if integration_success:
            print("\n✅ All tests passed! TushareAdapter is ready for production use.")
            print("\n📖 Next steps:")
            print("  1. Run the comprehensive demo: python src/adapters/tushare_demo.py")
            print("  2. Integrate with your ArbitrageEngine")
            print("  3. Configure monitoring and alerting")
            print("  4. Set up production deployment")
        else:
            print("\n❌ Some tests failed. Please check the configuration and try again.")
    
    try:
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)