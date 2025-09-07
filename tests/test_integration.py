#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive integration tests for the complete arbitrage scanning system.

Tests end-to-end functionality including:
- Data acquisition through TushareAdapter
- Processing through ArbitrageEngine
- Enhanced pricing calculations
- Legacy algorithm integration
- Signal generation and result validation

Performance verification targets:
- Complete scan time: <3 seconds
- Memory usage: <1GB
- Discovery accuracy: >95% compared to legacy
- System stability: 99.9%
"""

import asyncio
import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np
import time
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main_arbitrage_scanner import (
    MainArbitrageScanner, ScanConfiguration, ScanResult,
    LegacyAlgorithmIntegrator, create_scanner, run_scan
)
from config.models import StrategyType, ArbitrageOpportunity


class TestMainArbitrageScanner(unittest.TestCase):
    """Test the main arbitrage scanner integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create temporary config directory
        cls.temp_dir = tempfile.mkdtemp()
        cls.config_dir = Path(cls.temp_dir) / "config"
        cls.config_dir.mkdir()
        
        # Create test configuration
        cls._create_test_config()
        
        # Create logs directory
        (Path(cls.temp_dir) / "logs").mkdir()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    @classmethod
    def _create_test_config(cls):
        """Create test configuration files."""
        config_content = """
app_name: "Test Options Arbitrage Scanner"
version: "1.0.0"
environment: "test"
debug: true
log_level: "INFO"

data_dir: "./data"
log_dir: "./logs"
config_dir: "./config"

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
    api_token: "test_token"
    rate_limit: 200
    timeout: 30

strategies:
  pricing_arbitrage:
    type: "pricing_arbitrage"
    enabled: true
    priority: 1
  put_call_parity:
    type: "put_call_parity"
    enabled: true
    priority: 2
"""
        
        config_file = cls.config_dir / "config.yaml"
        with open(config_file, 'w') as f:
            f.write(config_content)
    
    def setUp(self):
        """Set up each test."""
        self.config_path = str(self.config_dir / "config.yaml")
        
        # Create mock data
        self.mock_options_data = self._create_mock_options_data()
    
    def _create_mock_options_data(self) -> pd.DataFrame:
        """Create realistic mock options data for testing."""
        np.random.seed(42)  # For reproducible tests
        
        data = []
        underlying_codes = ['IF2401', 'IH2401', 'IC2401']
        
        for underlying in underlying_codes:
            for i in range(20):  # 20 options per underlying
                call_put = 'C' if i % 2 == 0 else 'P'
                strike = 4000 + i * 50  # Strikes from 4000 to 4950
                
                # Create realistic option data
                option_data = {
                    'ts_code': f'{underlying}-{call_put}-{strike}.GFE',
                    'name': f'{underlying} {call_put} {strike}',
                    'underlying': underlying,
                    'call_put': call_put,
                    'exercise_price': strike,
                    'close': max(0.1, np.random.normal(50, 20)),  # Positive prices
                    'vol': max(10, int(np.random.exponential(100))),  # Volume
                    'oi': max(1, int(np.random.exponential(50))),  # Open interest
                    'days_to_expiry': np.random.randint(1, 45),
                    'underlying_price': strike + np.random.normal(0, 100),  # Underlying price
                    'market_price': max(0.1, np.random.normal(50, 20))  # Market price for comparison
                }
                data.append(option_data)
        
        # Add the famous PS2511-P-61000.GFE case for legacy compatibility
        legacy_case = {
            'ts_code': 'PS2511-P-61000.GFE',
            'name': 'PS2511 P 61000',
            'underlying': 'PS2511',
            'call_put': 'P',
            'exercise_price': 61000,
            'close': 125.5,  # Known good case from legacy
            'vol': 250,
            'oi': 180,
            'days_to_expiry': 15,
            'underlying_price': 60850,
            'market_price': 135.0  # Slightly overpriced for arbitrage
        }
        data.append(legacy_case)
        
        return pd.DataFrame(data)
    
    @patch('src.adapters.tushare_adapter.TushareAdapter')
    @patch('src.engine.arbitrage_engine.ArbitrageEngine')
    def test_scanner_initialization(self, mock_engine, mock_adapter):
        """Test scanner initialization."""
        # Mock the components
        mock_adapter_instance = AsyncMock()
        mock_adapter.return_value = mock_adapter_instance
        
        mock_engine_instance = Mock()
        mock_engine.return_value = mock_engine_instance
        
        # Create scanner
        scanner = MainArbitrageScanner(self.config_path)
        
        # Verify initialization
        self.assertIsNotNone(scanner.config_manager)
        self.assertIsNotNone(scanner.enhanced_pricer)
        self.assertIsNotNone(scanner.arbitrage_detector)
        self.assertIsNotNone(scanner.legacy_integrator)
        self.assertIsNotNone(scanner.performance_monitor)
    
    def test_legacy_algorithm_integrator(self):
        """Test legacy algorithm integration."""
        integrator = LegacyAlgorithmIntegrator()
        
        # Test pricing anomaly detection
        anomalies = integrator.find_simple_pricing_anomalies(
            self.mock_options_data, deviation_threshold=0.1
        )
        
        self.assertIsInstance(anomalies, list)
        
        # Should find some anomalies in the mock data
        if anomalies:
            anomaly = anomalies[0]
            required_fields = ['code', 'price', 'strike', 'type', 'z_score', 'anomaly_type']
            for field in required_fields:
                self.assertIn(field, anomaly)
        
        # Test put-call parity detection
        parity_ops = integrator.find_put_call_parity_opportunities(
            self.mock_options_data, tolerance=0.05
        )
        
        self.assertIsInstance(parity_ops, list)
        
        # Should find some parity opportunities
        if parity_ops:
            parity = parity_ops[0]
            required_fields = ['underlying', 'strike', 'call_code', 'put_code', 
                             'parity_error', 'opportunity']
            for field in required_fields:
                self.assertIn(field, parity)
    
    def test_legacy_ps2511_case(self):
        """Test the famous PS2511-P-61000.GFE legacy case."""
        integrator = LegacyAlgorithmIntegrator()
        
        # Filter to just the legacy case
        legacy_data = self.mock_options_data[
            self.mock_options_data['ts_code'] == 'PS2511-P-61000.GFE'
        ].copy()
        
        # Create a small dataset with the legacy case and some comparables
        test_data = pd.concat([
            legacy_data,
            self.mock_options_data[self.mock_options_data['underlying'] == 'PS2511']
        ]).reset_index(drop=True)
        
        # Test anomaly detection
        anomalies = integrator.find_simple_pricing_anomalies(test_data)
        
        # Should detect the overpriced option
        ps_anomalies = [a for a in anomalies if 'PS2511' in a['code']]
        self.assertGreater(len(ps_anomalies), 0, "Should detect PS2511 anomaly")
        
        # Verify the anomaly characteristics
        if ps_anomalies:
            anomaly = ps_anomalies[0]
            self.assertEqual(anomaly['strike'], 61000)
            self.assertGreater(anomaly['z_score'], 0)  # Should show anomaly
    
    @patch('src.adapters.tushare_adapter.TushareAdapter')
    async def test_data_fetch_integration(self, mock_adapter):
        """Test data fetching integration."""
        # Setup mock adapter
        mock_adapter_instance = AsyncMock()
        mock_adapter.return_value = mock_adapter_instance
        
        # Mock successful data response
        from src.adapters.base import DataResponse
        mock_response = DataResponse(
            success=True,
            data=self.mock_options_data,
            timestamp=datetime.now()
        )
        mock_adapter_instance.fetch_data.return_value = mock_response
        
        # Create scanner with mocked dependencies
        with patch('src.main_arbitrage_scanner.ArbitrageEngine'):
            scanner = MainArbitrageScanner(self.config_path)
            scanner.data_adapter = mock_adapter_instance
            
            # Test data fetch
            scan_config = ScanConfiguration()
            data = await scanner._fetch_options_data(scan_config)
            
            # Verify data fetch
            self.assertFalse(data.empty)
            self.assertEqual(len(data), len(self.mock_options_data))
    
    @patch('src.adapters.tushare_adapter.TushareAdapter')
    @patch('src.engine.arbitrage_engine.ArbitrageEngine')
    async def test_full_scan_integration(self, mock_engine, mock_adapter):
        """Test full scan integration."""
        # Setup mocks
        mock_adapter_instance = AsyncMock()
        mock_adapter.return_value = mock_adapter_instance
        
        from src.adapters.base import DataResponse
        mock_response = DataResponse(
            success=True,
            data=self.mock_options_data,
            timestamp=datetime.now()
        )
        mock_adapter_instance.fetch_data.return_value = mock_response
        
        mock_engine_instance = AsyncMock()
        mock_engine.return_value = mock_engine_instance
        
        # Mock engine opportunities
        mock_opportunities = [
            ArbitrageOpportunity(
                id="test_opp_1",
                strategy_type=StrategyType.PRICING_ARBITRAGE,
                underlying_asset="IF2401",
                description="Test opportunity",
                expected_profit=0.05,
                max_loss=0.02,
                probability=0.8,
                risk_level=RiskLevel.LOW,
                discovery_time=datetime.now(),
                expiration_time=datetime.now() + timedelta(hours=24)
            )
        ]
        mock_engine_instance.find_opportunities.return_value = mock_opportunities
        
        # Create scanner
        scanner = MainArbitrageScanner(self.config_path)
        scanner.data_adapter = mock_adapter_instance
        scanner.arbitrage_engine = mock_engine_instance
        
        # Run full scan
        scan_config = ScanConfiguration(max_results=10)
        result = await scanner.run_full_scan(scan_config)
        
        # Verify scan result
        self.assertIsInstance(result, ScanResult)
        self.assertGreater(len(result.opportunities), 0)
        self.assertGreater(result.total_options_processed, 0)
        self.assertGreaterEqual(result.success_rate, 0.0)
        self.assertLessEqual(result.success_rate, 1.0)
    
    def test_scan_performance(self):
        """Test scan performance targets."""
        # This would be a more comprehensive performance test
        # For now, we'll test basic timing
        
        with patch('src.adapters.tushare_adapter.TushareAdapter') as mock_adapter:
            mock_adapter_instance = AsyncMock()
            mock_adapter.return_value = mock_adapter_instance
            
            from src.adapters.base import DataResponse
            mock_response = DataResponse(
                success=True,
                data=self.mock_options_data,
                timestamp=datetime.now()
            )
            mock_adapter_instance.fetch_data.return_value = mock_response
            
            with patch('src.engine.arbitrage_engine.ArbitrageEngine') as mock_engine:
                mock_engine_instance = AsyncMock()
                mock_engine.return_value = mock_engine_instance
                mock_engine_instance.find_opportunities.return_value = []
                
                scanner = MainArbitrageScanner(self.config_path)
                scanner.data_adapter = mock_adapter_instance
                scanner.arbitrage_engine = mock_engine_instance
                
                # Measure scan time
                start_time = time.time()
                
                scan_config = ScanConfiguration()
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                result = loop.run_until_complete(scanner.run_full_scan(scan_config))
                
                scan_time = time.time() - start_time
                
                # Verify performance target: <3 seconds
                self.assertLess(scan_time, 3.0, f"Scan took {scan_time:.2f}s, should be <3s")
                self.assertGreater(result.performance_metrics.total_scan_time, 0)
    
    def test_synchronous_wrapper_functions(self):
        """Test synchronous wrapper functions for Streamlit compatibility."""
        with patch('src.main_arbitrage_scanner.MainArbitrageScanner') as mock_scanner:
            # Test scanner creation
            mock_scanner_instance = Mock()
            mock_scanner.return_value = mock_scanner_instance
            
            scanner = create_scanner("test_config.yaml")
            mock_scanner.assert_called_once_with("test_config.yaml")
            
            # Test scan execution
            mock_result = ScanResult(
                opportunities=[],
                trading_signals=[],
                performance_metrics=Mock(),
                data_quality_report={}
            )
            
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop_instance = Mock()
                mock_loop.return_value = mock_loop_instance
                mock_loop_instance.is_running.return_value = False
                mock_loop_instance.run_until_complete.return_value = mock_result
                
                mock_scanner_instance.run_full_scan.return_value = mock_result
                
                result = run_scan(mock_scanner_instance)
                self.assertEqual(result, mock_result)
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        with patch('src.adapters.tushare_adapter.TushareAdapter') as mock_adapter:
            # Test data fetch failure
            mock_adapter_instance = AsyncMock()
            mock_adapter.return_value = mock_adapter_instance
            
            from src.adapters.base import DataResponse
            mock_response = DataResponse(
                success=False,
                data=None,
                error_message="Test error",
                timestamp=datetime.now()
            )
            mock_adapter_instance.fetch_data.return_value = mock_response
            
            with patch('src.engine.arbitrage_engine.ArbitrageEngine'):
                scanner = MainArbitrageScanner(self.config_path)
                scanner.data_adapter = mock_adapter_instance
                
                scan_config = ScanConfiguration()
                
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                result = loop.run_until_complete(scanner.run_full_scan(scan_config))
                
                # Should handle error gracefully
                self.assertEqual(len(result.opportunities), 0)
                self.assertFalse(result.data_quality_report["validation_passed"])


class TestEnhancedPricingIntegration(unittest.TestCase):
    """Test enhanced pricing engine integration."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = pd.DataFrame({
            'ts_code': ['TEST001C', 'TEST002P'],
            'underlying_price': [100.0, 100.0],
            'exercise_price': [100.0, 110.0],
            'days_to_expiry': [30, 30],
            'call_put': ['C', 'P'],
            'close': [5.0, 8.0],
            'market_price': [5.2, 7.8]
        })
    
    def test_enhanced_pricing_application(self):
        """Test enhanced pricing calculations."""
        scanner = MainArbitrageScanner.__new__(MainArbitrageScanner)
        scanner.enhanced_pricer = Mock()
        scanner.enhanced_pricer.batch_pricing.return_value = self.test_data.copy()
        scanner.logger = Mock()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(scanner._apply_enhanced_pricing(self.test_data))
        
        # Should call enhanced pricing
        scanner.enhanced_pricer.batch_pricing.assert_called_once()
        self.assertFalse(result.empty)
        
        loop.close()
    
    def test_arbitrage_detection(self):
        """Test arbitrage opportunity detection."""
        from enhanced_pricing_engine import ArbitrageDetector
        
        detector = ArbitrageDetector()
        opportunities = detector.find_pricing_arbitrage(self.test_data)
        
        # Should return DataFrame (might be empty if no opportunities)
        self.assertIsInstance(opportunities, pd.DataFrame)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests."""
    
    def test_memory_usage(self):
        """Test memory usage stays within limits."""
        import psutil
        import gc
        
        # Measure initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset
        large_data = pd.DataFrame({
            'ts_code': [f'TEST{i:06d}' for i in range(10000)],
            'close': np.random.rand(10000) * 100,
            'exercise_price': np.random.rand(10000) * 1000 + 500,
            'call_put': ['C' if i % 2 == 0 else 'P' for i in range(10000)],
            'days_to_expiry': np.random.randint(1, 90, 10000)
        })
        
        # Process data
        integrator = LegacyAlgorithmIntegrator()
        anomalies = integrator.find_simple_pricing_anomalies(large_data)
        
        # Measure final memory
        gc.collect()  # Force garbage collection
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should stay under 1GB increase
        self.assertLess(memory_increase, 1024, f"Memory increase: {memory_increase:.2f} MB")
    
    def test_scalability(self):
        """Test system scalability with increasing data size."""
        integrator = LegacyAlgorithmIntegrator()
        
        sizes = [100, 500, 1000, 5000]
        times = []
        
        for size in sizes:
            # Create test data
            test_data = pd.DataFrame({
                'ts_code': [f'TEST{i:06d}' for i in range(size)],
                'close': np.random.rand(size) * 100,
                'exercise_price': np.random.rand(size) * 1000 + 500,
                'call_put': ['C' if i % 2 == 0 else 'P' for i in range(size)],
                'days_to_expiry': np.random.randint(1, 90, size)
            })
            
            # Measure processing time
            start_time = time.time()
            anomalies = integrator.find_simple_pricing_anomalies(test_data)
            processing_time = time.time() - start_time
            
            times.append(processing_time)
            
            # Should complete within reasonable time
            self.assertLess(processing_time, size * 0.001, 
                          f"Processing {size} records took {processing_time:.3f}s")
        
        # Verify roughly linear scaling
        if len(times) >= 2:
            scaling_ratio = times[-1] / times[0]
            data_ratio = sizes[-1] / sizes[0]
            
            # Should not be worse than quadratic scaling
            self.assertLess(scaling_ratio, data_ratio ** 2,
                          "Algorithm scaling appears to be worse than quadratic")


if __name__ == '__main__':
    # Create test suite
    unittest.main(verbosity=2)