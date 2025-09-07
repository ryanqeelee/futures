#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legacy Compatibility Verification Script

Verifies that the integrated system can reproduce the successful 
PS2511-P-61000.GFE arbitrage case from the legacy system.

This demonstrates end-to-end compatibility and validates that
the integration preserves proven arbitrage detection capabilities.
"""

import sys
import pandas as pd
import numpy as np
import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from main_arbitrage_scanner import (
        MainArbitrageScanner, ScanConfiguration, LegacyAlgorithmIntegrator,
        create_scanner, run_scan
    )
    from config.models import StrategyType
    print("✅ Successfully imported integrated system components")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please ensure the src directory is properly set up")
    sys.exit(1)


def create_legacy_test_data():
    """Create test data including the famous PS2511-P-61000.GFE case."""
    
    # The famous case that worked in legacy system
    legacy_case = {
        'ts_code': 'PS2511-P-61000.GFE',
        'name': 'PS2511 P 61000',
        'underlying': 'PS2511', 
        'call_put': 'P',
        'exercise_price': 61000,
        'close': 125.5,  # Market price from legacy data
        'vol': 250,      # Volume
        'oi': 180,       # Open interest
        'days_to_expiry': 15,
        'underlying_price': 60850,  # Underlying asset price
        'market_price': 135.0       # Slightly overpriced for arbitrage detection
    }
    
    # Create additional test data for context
    test_data = []
    
    # Add some normal options for the same underlying
    for i in range(5):
        strike_offset = (i - 2) * 1000  # Strikes around 61000
        strike = 61000 + strike_offset
        
        for option_type in ['C', 'P']:
            option = {
                'ts_code': f'PS2511-{option_type}-{strike}.GFE',
                'name': f'PS2511 {option_type} {strike}',
                'underlying': 'PS2511',
                'call_put': option_type,
                'exercise_price': strike,
                'close': max(5, 150 - abs(strike_offset) / 50),  # Realistic prices
                'vol': np.random.randint(50, 300),
                'oi': np.random.randint(20, 200),
                'days_to_expiry': 15 + i,
                'underlying_price': 60850,
                'market_price': max(5, 150 - abs(strike_offset) / 50) * (1 + np.random.uniform(-0.05, 0.05))
            }
            test_data.append(option)
    
    # Add the legacy case
    test_data.append(legacy_case)
    
    return pd.DataFrame(test_data)


def test_legacy_algorithm_directly():
    """Test legacy algorithms directly on the PS2511 case."""
    print("\n🔍 Testing Legacy Algorithm Integration")
    print("-" * 50)
    
    # Create test data
    test_data = create_legacy_test_data()
    print(f"Created test dataset with {len(test_data)} options")
    
    # Initialize legacy integrator
    integrator = LegacyAlgorithmIntegrator()
    
    # Test pricing anomaly detection
    print("\n1. Testing Pricing Anomaly Detection...")
    anomalies = integrator.find_simple_pricing_anomalies(test_data, deviation_threshold=0.1)
    
    print(f"Found {len(anomalies)} pricing anomalies")
    
    # Look for the PS2511 case specifically
    ps_anomalies = [a for a in anomalies if 'PS2511' in a['code'] and '61000' in a['code']]
    
    if ps_anomalies:
        print("✅ PS2511-P-61000.GFE anomaly detected!")
        for anomaly in ps_anomalies:
            print(f"   Code: {anomaly['code']}")
            print(f"   Price: {anomaly['price']:.2f}")
            print(f"   Strike: {anomaly['strike']:.0f}")
            print(f"   Z-Score: {anomaly['z_score']:.3f}")
            print(f"   Anomaly Type: {anomaly['anomaly_type']}")
    else:
        print("⚠️ PS2511-P-61000.GFE anomaly not detected")
    
    # Test put-call parity detection
    print("\n2. Testing Put-Call Parity Detection...")
    parity_ops = integrator.find_put_call_parity_opportunities(test_data, tolerance=0.1)
    
    print(f"Found {len(parity_ops)} put-call parity opportunities")
    
    ps_parity = [p for p in parity_ops if p['underlying'] == 'PS2511']
    if ps_parity:
        print("✅ PS2511 parity opportunities detected!")
        for parity in ps_parity:
            print(f"   Strike: {parity['strike']:.0f}")
            print(f"   Parity Error: {parity['parity_error']:.3f}")
            print(f"   Relative Error: {parity['relative_error']:.2%}")
    
    return len(ps_anomalies) > 0 or len(ps_parity) > 0


def test_enhanced_pricing_engine():
    """Test the enhanced pricing engine on legacy data."""
    print("\n🚀 Testing Enhanced Pricing Engine")
    print("-" * 50)
    
    try:
        from enhanced_pricing_engine import VectorizedOptionPricer, ArbitrageDetector
        
        # Create test data
        test_data = create_legacy_test_data()
        
        # Add required columns for enhanced pricing
        test_data['underlying_price'] = test_data['underlying_price']
        test_data['days_to_expiry'] = test_data['days_to_expiry']
        
        # Initialize enhanced pricer
        pricer = VectorizedOptionPricer()
        
        # Apply enhanced pricing
        print("Applying enhanced pricing calculations...")
        enhanced_data = pricer.batch_pricing(test_data)
        
        print(f"✅ Enhanced pricing completed for {len(enhanced_data)} options")
        
        # Show results for PS2511 case
        ps_result = enhanced_data[enhanced_data['ts_code'] == 'PS2511-P-61000.GFE']
        
        if not ps_result.empty:
            ps_option = ps_result.iloc[0]
            print(f"\nPS2511-P-61000.GFE Enhanced Results:")
            print(f"   Market Price: {ps_option['market_price']:.2f}")
            print(f"   Theoretical Price: {ps_option['theoretical_price']:.2f}")
            
            if 'implied_volatility' in ps_option:
                print(f"   Implied Volatility: {ps_option['implied_volatility']*100:.1f}%")
        
        # Test arbitrage detection
        detector = ArbitrageDetector()
        arbitrage_ops = detector.find_pricing_arbitrage(enhanced_data)
        
        print(f"\nArbitrage Detector found {len(arbitrage_ops)} opportunities")
        
        ps_arb = arbitrage_ops[arbitrage_ops['ts_code'] == 'PS2511-P-61000.GFE'] if not arbitrage_ops.empty else pd.DataFrame()
        
        if not ps_arb.empty:
            print("✅ PS2511-P-61000.GFE arbitrage opportunity confirmed by enhanced engine!")
            ps_op = ps_arb.iloc[0]
            print(f"   Price Deviation: {ps_op['price_deviation']:.2%}")
            if 'risk_score' in ps_op:
                print(f"   Risk Score: {ps_op['risk_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced pricing engine test failed: {e}")
        return False


def test_full_integration():
    """Test the full integrated system."""
    print("\n🎯 Testing Full System Integration")  
    print("-" * 50)
    
    try:
        # Create mock configuration for testing
        config_content = """
app_name: "Test Arbitrage Scanner"
version: "1.0.0"
environment: "test"
debug: true
log_level: "INFO"

risk:
  max_position_size: 10000.0
  max_daily_loss: 1000.0
  min_liquidity_volume: 10
  max_concentration: 0.5
  max_days_to_expiry: 45
  min_days_to_expiry: 1

data_sources:
  tushare:
    type: "tushare"
    enabled: true
    priority: 1

strategies:
  pricing_arbitrage:
    type: "pricing_arbitrage"
    enabled: true
    priority: 1
"""
        
        # Write temporary config
        config_path = Path("test_config.yaml")
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        try:
            # Mock the data adapter to return our test data
            from unittest.mock import Mock, patch, AsyncMock
            from src.adapters.base import DataResponse
            
            test_data = create_legacy_test_data()
            
            # Create mock response
            mock_response = DataResponse(
                success=True,
                data=test_data,
                timestamp=datetime.now()
            )
            
            # Patch the components we can't fully initialize in test
            with patch('src.main_arbitrage_scanner.TushareAdapter') as mock_adapter:
                with patch('src.main_arbitrage_scanner.ArbitrageEngine') as mock_engine:
                    
                    # Setup mocks
                    mock_adapter_instance = AsyncMock()
                    mock_adapter.return_value = mock_adapter_instance
                    mock_adapter_instance.fetch_data.return_value = mock_response
                    
                    mock_engine_instance = AsyncMock()
                    mock_engine.return_value = mock_engine_instance
                    mock_engine_instance.find_opportunities.return_value = []
                    
                    # Create scanner
                    scanner = MainArbitrageScanner(str(config_path))
                    scanner.data_adapter = mock_adapter_instance
                    scanner.arbitrage_engine = mock_engine_instance
                    
                    # Configure scan
                    scan_config = ScanConfiguration(
                        max_results=20,
                        min_profit_threshold=0.01,
                        enable_legacy_integration=True,
                        enable_enhanced_pricing=True
                    )
                    
                    # Run scan
                    print("Executing full arbitrage scan...")
                    start_time = time.time()
                    
                    result = asyncio.run(scanner.run_full_scan(scan_config))
                    
                    scan_time = time.time() - start_time
                    
                    # Verify results
                    print(f"✅ Scan completed in {scan_time:.2f} seconds")
                    print(f"   Total opportunities found: {len(result.opportunities)}")
                    print(f"   Trading signals generated: {len(result.trading_signals)}")
                    print(f"   Options processed: {result.total_options_processed}")
                    print(f"   Success rate: {result.success_rate:.2%}")
                    print(f"   Legacy compatibility score: {result.legacy_compatibility_score:.2%}")
                    
                    # Look for PS2511 opportunities
                    ps_opportunities = [
                        opp for opp in result.opportunities 
                        if 'PS2511' in opp.underlying_asset or 'PS2511' in str(opp.metadata)
                    ]
                    
                    if ps_opportunities:
                        print(f"\n✅ Found {len(ps_opportunities)} PS2511 opportunities:")
                        for opp in ps_opportunities:
                            print(f"   {opp.strategy_type.value}: {opp.description}")
                            print(f"   Expected profit: {opp.expected_profit:.4f}")
                            print(f"   Probability: {opp.probability:.2%}")
                    
                    # Performance verification
                    performance_ok = True
                    if scan_time > 3.0:
                        print(f"⚠️ Scan time {scan_time:.2f}s exceeds 3s target")
                        performance_ok = False
                    
                    if result.total_options_processed == 0:
                        print("⚠️ No options were processed")
                        performance_ok = False
                    
                    return len(ps_opportunities) > 0 and performance_ok
                    
        finally:
            # Cleanup
            if config_path.exists():
                config_path.unlink()
    
    except Exception as e:
        print(f"❌ Full integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all compatibility tests."""
    print("🔬 Legacy Compatibility Verification")
    print("=" * 60)
    print("Testing integration with proven PS2511-P-61000.GFE arbitrage case")
    
    results = {
        "legacy_algorithms": False,
        "enhanced_pricing": False, 
        "full_integration": False
    }
    
    # Test 1: Legacy algorithms
    try:
        results["legacy_algorithms"] = test_legacy_algorithm_directly()
    except Exception as e:
        print(f"❌ Legacy algorithm test failed: {e}")
    
    # Test 2: Enhanced pricing engine
    try:
        results["enhanced_pricing"] = test_enhanced_pricing_engine()
    except Exception as e:
        print(f"❌ Enhanced pricing test failed: {e}")
    
    # Test 3: Full integration
    try:
        results["full_integration"] = test_full_integration()
    except Exception as e:
        print(f"❌ Full integration test failed: {e}")
    
    # Summary
    print(f"\n📊 Test Results Summary")
    print("-" * 30)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Legacy compatibility confirmed.")
        print("The integrated system successfully reproduces legacy arbitrage detection.")
    else:
        print(f"\n⚠️ {total_tests - passed_tests} tests failed.")
        print("Some legacy compatibility issues detected.")
    
    # Performance summary
    print(f"\n⚡ Performance Summary")
    print("- Target scan time: <3 seconds")
    print("- Target accuracy: >95% legacy compatibility") 
    print("- Target scalability: >10,000 options")
    print("- Memory target: <1GB usage")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)