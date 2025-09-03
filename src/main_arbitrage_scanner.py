#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Arbitrage Scanner - Comprehensive Integration System

This module provides the main coordinator that integrates all components:
- TushareAdapter for enterprise-grade data acquisition
- ArbitrageEngine for high-performance opportunity detection
- Enhanced pricing engine for optimized calculations
- Legacy algorithm integration for proven strategies
- Complete end-to-end scanning and signal generation

Performance targets:
- Complete scan time: <3 seconds
- Memory usage: <1GB
- Discovery accuracy: >95% compared to legacy
- System stability: 99.9% uptime
- Scalability: >10,000 options processing
"""

import asyncio
import logging
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
import numpy as np

# Import core components
from .config.manager import ConfigManager
from .config.models import (
    ArbitrageOpportunity, StrategyType, SystemConfig,
    RiskConfig, StrategyConfig
)
from .engine.arbitrage_engine import (
    ArbitrageEngine, ScanParameters, TradingSignal,
    EnginePerformanceMetrics
)
from .adapters.tushare_adapter import TushareAdapter
from .strategies.base import (
    BaseStrategy, StrategyResult, OptionData,
    RiskMetrics, TradingAction, ActionType, RiskLevel
)
from .utils.performance_monitor import PerformanceMonitor


# Import enhanced pricing engine
import sys
sys.path.append(str(Path(__file__).parent.parent))
from enhanced_pricing_engine import (
    VectorizedOptionPricer, ArbitrageDetector,
    EnhancedBlackScholesEngine, RobustImpliedVolatility
)


@dataclass
class ScanResult:
    """Complete scan result with all findings and metrics."""
    opportunities: List[ArbitrageOpportunity]
    trading_signals: List[TradingSignal]
    performance_metrics: EnginePerformanceMetrics
    data_quality_report: Dict[str, Any]
    scan_timestamp: datetime = field(default_factory=datetime.now)
    total_options_processed: int = 0
    success_rate: float = 0.0
    legacy_compatibility_score: float = 0.0


@dataclass
class ScanConfiguration:
    """Configuration for scan execution."""
    max_days_to_expiry: int = 45
    min_days_to_expiry: int = 1
    min_volume: int = 100
    min_profit_threshold: float = 0.02
    max_risk_tolerance: float = 0.15
    enable_legacy_integration: bool = True
    enable_enhanced_pricing: bool = True
    parallel_processing: bool = True
    max_results: int = 50
    cache_results: bool = True


class LegacyAlgorithmIntegrator:
    """Integration layer for proven legacy algorithms."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.LegacyAlgorithmIntegrator")
    
    def find_simple_pricing_anomalies(self, options_df: pd.DataFrame, 
                                    deviation_threshold: float = 0.15) -> List[Dict]:
        """
        Legacy pricing anomaly detection (from simple_arbitrage_demo.py)
        Adapted for integration with new system.
        """
        if options_df.empty:
            return []
        
        anomalies = []
        
        # Group by underlying asset
        for underlying in options_df.get('underlying', options_df['ts_code'].str.extract(r'([A-Z]+\d{4})')[0]).unique():
            if pd.isna(underlying):
                continue
                
            underlying_options = options_df[
                options_df['ts_code'].str.contains(str(underlying), na=False)
            ].copy()
            
            if len(underlying_options) < 5:
                continue
            
            # Analyze by option type
            for option_type in ['C', 'P']:
                type_options = underlying_options[
                    underlying_options['call_put'] == option_type
                ].copy()
                
                if len(type_options) < 3:
                    continue
                
                # Calculate price ratios
                type_options['price_ratio'] = type_options['close'] / type_options['exercise_price']
                
                # Statistical analysis
                mean_ratio = type_options['price_ratio'].mean()
                std_ratio = type_options['price_ratio'].std()
                
                if std_ratio == 0:
                    continue
                
                # Find anomalies
                for _, option in type_options.iterrows():
                    z_score = abs((option['price_ratio'] - mean_ratio) / std_ratio)
                    
                    if z_score > 2:  # 2 standard deviations
                        anomalies.append({
                            'code': option['ts_code'],
                            'name': option.get('name', ''),
                            'price': option['close'],
                            'strike': option['exercise_price'],
                            'type': '认购' if option_type == 'C' else '认沽',
                            'price_ratio': option['price_ratio'],
                            'z_score': z_score,
                            'volume': option.get('vol', 0),
                            'days_to_expiry': option.get('days_to_expiry', 30),
                            'anomaly_type': '价格异常高' if option['price_ratio'] > mean_ratio else '价格异常低'
                        })
        
        # Sort by anomaly severity
        anomalies.sort(key=lambda x: x['z_score'], reverse=True)
        return anomalies
    
    def find_put_call_parity_opportunities(self, options_df: pd.DataFrame,
                                         tolerance: float = 0.05) -> List[Dict]:
        """
        Legacy put-call parity detection (from simple_arbitrage_demo.py)
        Adapted for integration with new system.
        """
        if options_df.empty:
            return []
        
        parity_ops = []
        
        # Group by underlying
        options_df['underlying'] = options_df.get('underlying', 
                                                options_df['ts_code'].str.extract(r'([A-Z]+\d{4})')[0])
        
        for underlying in options_df['underlying'].unique():
            if pd.isna(underlying):
                continue
            
            underlying_options = options_df[options_df['underlying'] == underlying].copy()
            
            # Group by strike price
            for strike in underlying_options['exercise_price'].unique():
                if pd.isna(strike):
                    continue
                
                strike_options = underlying_options[underlying_options['exercise_price'] == strike]
                
                calls = strike_options[strike_options['call_put'] == 'C']
                puts = strike_options[strike_options['call_put'] == 'P']
                
                if calls.empty or puts.empty:
                    continue
                
                # Select highest volume options
                best_call = calls.loc[calls.get('vol', calls.get('volume', pd.Series([0]))).idxmax()]
                best_put = puts.loc[puts.get('vol', puts.get('volume', pd.Series([0]))).idxmax()]
                
                # Simplified parity check
                estimated_underlying = strike  # Simplified assumption
                theoretical_diff = estimated_underlying - strike
                actual_diff = best_call['close'] - best_put['close']
                
                parity_error = abs(actual_diff - theoretical_diff)
                relative_error = parity_error / max(abs(theoretical_diff), 1)
                
                if relative_error > tolerance:
                    parity_ops.append({
                        'underlying': underlying,
                        'strike': strike,
                        'call_code': best_call['ts_code'],
                        'put_code': best_put['ts_code'],
                        'call_price': best_call['close'],
                        'put_price': best_put['close'],
                        'actual_diff': actual_diff,
                        'theoretical_diff': theoretical_diff,
                        'parity_error': parity_error,
                        'relative_error': relative_error,
                        'opportunity': '买入看涨卖出看跌' if actual_diff < theoretical_diff else '卖出看涨买入看跌'
                    })
        
        # Sort by error magnitude
        parity_ops.sort(key=lambda x: x['parity_error'], reverse=True)
        return parity_ops


class MainArbitrageScanner:
    """
    Main arbitrage scanner coordinator.
    
    Integrates all system components into a unified scanning system:
    - Data acquisition through TushareAdapter
    - High-performance opportunity detection via ArbitrageEngine
    - Enhanced pricing calculations
    - Legacy algorithm compatibility
    - Comprehensive risk assessment
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the main arbitrage scanner.
        
        Args:
            config_path: Path to system configuration file
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_system_config()
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(f"{__name__}.MainArbitrageScanner")
        
        # Initialize core components
        self.data_adapter: Optional[TushareAdapter] = None
        self.arbitrage_engine: Optional[ArbitrageEngine] = None
        self.enhanced_pricer = VectorizedOptionPricer()
        self.arbitrage_detector = ArbitrageDetector()
        self.legacy_integrator = LegacyAlgorithmIntegrator()
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.scan_count = 0
        self.total_scan_time = 0.0
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("MainArbitrageScanner initialized successfully")
    
    def _setup_logging(self):
        """Configure logging for the scanner."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/arbitrage_scanner.log', mode='a')
            ]
        )
        
        # Suppress noisy third-party logs
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        warnings.filterwarnings('ignore')
    
    async def _initialize_components(self):
        """Initialize all system components asynchronously."""
        try:
            # Initialize data adapter
            self.data_adapter = TushareAdapter(
                source_type="tushare",
                config=self.config_manager.get_data_source_config("tushare")
            )
            await self.data_adapter.initialize()
            
            # Initialize arbitrage engine
            self.arbitrage_engine = ArbitrageEngine(
                config=self.config,
                performance_monitor=self.performance_monitor
            )
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise
    
    def _initialize_components(self):
        """Synchronous wrapper for component initialization."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            # If loop is already running, create a new thread
            def init_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                new_loop.run_until_complete(self._initialize_components_async())
                new_loop.close()
            
            thread = threading.Thread(target=init_in_thread)
            thread.start()
            thread.join()
        else:
            loop.run_until_complete(self._initialize_components_async())
    
    async def _initialize_components_async(self):
        """Actual async component initialization."""
        try:
            # Initialize data adapter
            self.data_adapter = TushareAdapter(
                source_type="tushare",
                config=self.config_manager.get_data_source_config("tushare")
            )
            await self.data_adapter.initialize()
            
            # Initialize arbitrage engine
            self.arbitrage_engine = ArbitrageEngine(
                config=self.config,
                performance_monitor=self.performance_monitor
            )
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise
    
    async def run_full_scan(self, scan_config: Optional[ScanConfiguration] = None) -> ScanResult:
        """
        Execute complete arbitrage scan with all strategies.
        
        Args:
            scan_config: Custom scan configuration
            
        Returns:
            Complete scan result with opportunities and metrics
        """
        start_time = time.time()
        self.scan_count += 1
        
        if scan_config is None:
            scan_config = ScanConfiguration()
        
        self.logger.info(f"Starting full arbitrage scan #{self.scan_count}")
        
        try:
            # Step 1: Data acquisition
            options_data = await self._fetch_options_data(scan_config)
            if options_data.empty:
                self.logger.warning("No options data available for scanning")
                return self._create_empty_result()
            
            # Step 2: Data quality validation
            quality_report = await self._validate_data_quality(options_data)
            
            # Step 3: Enhanced pricing calculations
            enhanced_data = await self._apply_enhanced_pricing(options_data)
            
            # Step 4: Legacy algorithm integration
            legacy_opportunities = await self._apply_legacy_algorithms(enhanced_data, scan_config)
            
            # Step 5: Modern strategy execution
            modern_opportunities = await self._execute_modern_strategies(enhanced_data, scan_config)
            
            # Step 6: Opportunity consolidation and ranking
            all_opportunities = self._consolidate_opportunities(legacy_opportunities, modern_opportunities)
            
            # Step 7: Trading signal generation
            trading_signals = self._generate_trading_signals(all_opportunities, scan_config)
            
            # Step 8: Performance metrics collection
            scan_time = time.time() - start_time
            self.total_scan_time += scan_time
            
            performance_metrics = EnginePerformanceMetrics(
                total_scan_time=scan_time,
                total_opportunities_found=len(all_opportunities),
                avg_scan_time=self.total_scan_time / self.scan_count,
                strategies_executed=len(self.config.strategies),
                data_fetch_time=0.0,  # Will be populated by engine
                pricing_calculation_time=0.0  # Will be populated by pricing engine
            )
            
            result = ScanResult(
                opportunities=all_opportunities[:scan_config.max_results],
                trading_signals=trading_signals,
                performance_metrics=performance_metrics,
                data_quality_report=quality_report,
                total_options_processed=len(options_data),
                success_rate=len(all_opportunities) / len(options_data) if len(options_data) > 0 else 0.0,
                legacy_compatibility_score=self._calculate_legacy_compatibility(legacy_opportunities, modern_opportunities)
            )
            
            self.logger.info(f"Scan completed in {scan_time:.2f}s, found {len(all_opportunities)} opportunities")
            return result
            
        except Exception as e:
            self.logger.error(f"Scan failed: {e}")
            raise
    
    async def _fetch_options_data(self, scan_config: ScanConfiguration) -> pd.DataFrame:
        """Fetch options data using TushareAdapter."""
        try:
            from .adapters.base import DataRequest
            
            # Create data request
            request = DataRequest(
                request_type="options_data",
                filters={
                    "max_days_to_expiry": scan_config.max_days_to_expiry,
                    "min_days_to_expiry": scan_config.min_days_to_expiry,
                    "min_volume": scan_config.min_volume
                }
            )
            
            # Fetch data
            response = await self.data_adapter.fetch_data(request)
            
            if response.success and response.data is not None:
                return response.data
            else:
                self.logger.warning(f"Data fetch failed: {response.error_message}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Data fetch error: {e}")
            return pd.DataFrame()
    
    async def _validate_data_quality(self, options_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and generate report."""
        try:
            # Basic validation
            quality_report = {
                "total_records": len(options_data),
                "null_values": options_data.isnull().sum().to_dict(),
                "data_types": options_data.dtypes.astype(str).to_dict(),
                "validation_passed": True,
                "issues": []
            }
            
            # Check for required columns
            required_columns = ['ts_code', 'close', 'exercise_price', 'call_put']
            missing_columns = [col for col in required_columns if col not in options_data.columns]
            
            if missing_columns:
                quality_report["validation_passed"] = False
                quality_report["issues"].append(f"Missing columns: {missing_columns}")
            
            # Check for invalid prices
            if 'close' in options_data.columns:
                invalid_prices = (options_data['close'] <= 0).sum()
                if invalid_prices > 0:
                    quality_report["issues"].append(f"Invalid prices: {invalid_prices}")
            
            return quality_report
            
        except Exception as e:
            return {"validation_passed": False, "error": str(e)}
    
    async def _apply_enhanced_pricing(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """Apply enhanced pricing calculations."""
        try:
            if options_data.empty:
                return options_data
            
            # Prepare data for enhanced pricing
            enhanced_data = options_data.copy()
            
            # Estimate underlying prices if not available
            if 'underlying_price' not in enhanced_data.columns:
                enhanced_data['underlying_price'] = enhanced_data['exercise_price']  # Simplified
            
            # Calculate days to expiry if not available
            if 'days_to_expiry' not in enhanced_data.columns:
                enhanced_data['days_to_expiry'] = 30  # Default assumption
            
            # Apply enhanced pricing
            enhanced_data = self.enhanced_pricer.batch_pricing(enhanced_data)
            
            return enhanced_data
            
        except Exception as e:
            self.logger.error(f"Enhanced pricing failed: {e}")
            return options_data
    
    async def _apply_legacy_algorithms(self, options_data: pd.DataFrame, 
                                     scan_config: ScanConfiguration) -> List[ArbitrageOpportunity]:
        """Apply proven legacy algorithms."""
        if not scan_config.enable_legacy_integration:
            return []
        
        try:
            opportunities = []
            
            # Legacy pricing anomalies
            pricing_anomalies = self.legacy_integrator.find_simple_pricing_anomalies(
                options_data, deviation_threshold=scan_config.min_profit_threshold
            )
            
            # Legacy put-call parity
            parity_ops = self.legacy_integrator.find_put_call_parity_opportunities(
                options_data, tolerance=scan_config.max_risk_tolerance
            )
            
            # Convert legacy results to ArbitrageOpportunity objects
            for anomaly in pricing_anomalies:
                opportunity = ArbitrageOpportunity(
                    id=f"legacy_pricing_{anomaly['code']}_{datetime.now().microsecond}",
                    strategy_type=StrategyType.PRICING_ARBITRAGE,
                    underlying_asset=anomaly.get('underlying', ''),
                    description=f"Legacy pricing anomaly: {anomaly['anomaly_type']}",
                    expected_profit=abs(anomaly['z_score']) * 0.01,  # Rough estimate
                    max_loss=anomaly['price'] * 0.1,  # 10% max loss assumption
                    probability=min(anomaly['z_score'] / 5.0, 0.95),  # Convert z-score to probability
                    risk_level=RiskLevel.MEDIUM,
                    discovery_time=datetime.now(),
                    expiration_time=datetime.now() + timedelta(hours=24),
                    metadata={
                        "legacy_source": "simple_arbitrage_demo",
                        "z_score": anomaly['z_score'],
                        "price_ratio": anomaly['price_ratio'],
                        "volume": anomaly['volume']
                    }
                )
                opportunities.append(opportunity)
            
            for parity in parity_ops:
                opportunity = ArbitrageOpportunity(
                    id=f"legacy_parity_{parity['underlying']}_{datetime.now().microsecond}",
                    strategy_type=StrategyType.PUT_CALL_PARITY,
                    underlying_asset=parity['underlying'],
                    description=f"Legacy put-call parity: {parity['opportunity']}",
                    expected_profit=parity['parity_error'],
                    max_loss=max(parity['call_price'], parity['put_price']),
                    probability=min(parity['relative_error'] * 2, 0.9),
                    risk_level=RiskLevel.MEDIUM,
                    discovery_time=datetime.now(),
                    expiration_time=datetime.now() + timedelta(hours=24),
                    metadata={
                        "legacy_source": "simple_arbitrage_demo",
                        "strike": parity['strike'],
                        "call_price": parity['call_price'],
                        "put_price": parity['put_price'],
                        "parity_error": parity['parity_error']
                    }
                )
                opportunities.append(opportunity)
            
            self.logger.info(f"Legacy algorithms found {len(opportunities)} opportunities")
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Legacy algorithm application failed: {e}")
            return []
    
    async def _execute_modern_strategies(self, options_data: pd.DataFrame,
                                       scan_config: ScanConfiguration) -> List[ArbitrageOpportunity]:
        """Execute modern arbitrage strategies through ArbitrageEngine."""
        try:
            if self.arbitrage_engine is None:
                return []
            
            # Prepare scan parameters
            scan_params = ScanParameters(
                strategy_types=[StrategyType.PRICING_ARBITRAGE, StrategyType.PUT_CALL_PARITY],
                min_profit_threshold=scan_config.min_profit_threshold,
                max_risk_tolerance=scan_config.max_risk_tolerance,
                min_liquidity_volume=scan_config.min_volume,
                max_days_to_expiry=scan_config.max_days_to_expiry
            )
            
            # Execute engine scan
            opportunities = await self.arbitrage_engine.find_opportunities(
                options_data, scan_params
            )
            
            self.logger.info(f"Modern strategies found {len(opportunities)} opportunities")
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Modern strategy execution failed: {e}")
            return []
    
    def _consolidate_opportunities(self, legacy_ops: List[ArbitrageOpportunity],
                                 modern_ops: List[ArbitrageOpportunity]) -> List[ArbitrageOpportunity]:
        """Consolidate and rank opportunities from all sources."""
        all_opportunities = legacy_ops + modern_ops
        
        # Remove duplicates based on underlying asset and strategy type
        seen = set()
        unique_opportunities = []
        
        for opp in all_opportunities:
            key = (opp.underlying_asset, opp.strategy_type, opp.expected_profit)
            if key not in seen:
                seen.add(key)
                unique_opportunities.append(opp)
        
        # Rank by expected profit adjusted for risk
        unique_opportunities.sort(
            key=lambda x: x.expected_profit * x.probability / (1 + x.max_loss),
            reverse=True
        )
        
        return unique_opportunities
    
    def _generate_trading_signals(self, opportunities: List[ArbitrageOpportunity],
                                scan_config: ScanConfiguration) -> List[TradingSignal]:
        """Generate actionable trading signals from opportunities."""
        signals = []
        
        for opp in opportunities:
            if opp.expected_profit < scan_config.min_profit_threshold:
                continue
            
            # Create basic trading actions based on strategy type
            actions = []
            if opp.strategy_type == StrategyType.PRICING_ARBITRAGE:
                actions.append(TradingAction(
                    action_type=ActionType.BUY,
                    instrument=opp.underlying_asset,
                    quantity=100,  # Default quantity
                    price=0.0,  # Will be determined at execution
                    timestamp=datetime.now()
                ))
            
            signal = TradingSignal(
                opportunity_id=opp.id,
                signal_type="BUY" if opp.expected_profit > 0 else "HOLD",
                actions=actions,
                confidence=opp.probability,
                expected_profit=opp.expected_profit,
                max_loss=opp.max_loss
            )
            signals.append(signal)
        
        return signals
    
    def _calculate_legacy_compatibility(self, legacy_ops: List[ArbitrageOpportunity],
                                      modern_ops: List[ArbitrageOpportunity]) -> float:
        """Calculate compatibility score with legacy algorithms."""
        if not legacy_ops and not modern_ops:
            return 1.0
        
        total_ops = len(legacy_ops) + len(modern_ops)
        if total_ops == 0:
            return 1.0
        
        # Simple compatibility metric
        legacy_ratio = len(legacy_ops) / total_ops
        return min(legacy_ratio * 2, 1.0)  # Boost legacy contribution
    
    def _create_empty_result(self) -> ScanResult:
        """Create empty scan result for error cases."""
        return ScanResult(
            opportunities=[],
            trading_signals=[],
            performance_metrics=EnginePerformanceMetrics(),
            data_quality_report={"validation_passed": False, "error": "No data available"}
        )
    
    def get_top_opportunities(self, limit: int = 10) -> List[ArbitrageOpportunity]:
        """
        Get top opportunities from the most recent scan.
        
        Args:
            limit: Maximum number of opportunities to return
            
        Returns:
            List of top opportunities sorted by profit potential
        """
        # This would typically maintain state from the last scan
        # For now, return empty list as this is a stateless implementation
        return []
    
    def get_performance_metrics(self) -> EnginePerformanceMetrics:
        """Get current performance metrics."""
        return EnginePerformanceMetrics(
            total_scan_time=self.total_scan_time,
            avg_scan_time=self.total_scan_time / max(self.scan_count, 1),
            strategies_executed=self.scan_count
        )
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'data_adapter') and self.data_adapter:
            try:
                # Close data adapter connections
                pass
            except:
                pass


# Synchronous wrapper functions for Streamlit compatibility
def create_scanner(config_path: str = "config/config.yaml") -> MainArbitrageScanner:
    """Create and initialize scanner (synchronous)."""
    return MainArbitrageScanner(config_path)


def run_scan(scanner: MainArbitrageScanner, 
            scan_config: Optional[ScanConfiguration] = None) -> ScanResult:
    """Run scan synchronously for Streamlit compatibility."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        # If loop is already running, we need to handle this differently
        # This is common in Jupyter/Streamlit environments
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(scanner.run_full_scan(scan_config))
    else:
        return loop.run_until_complete(scanner.run_full_scan(scan_config))


if __name__ == "__main__":
    """Command-line interface for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run arbitrage scan")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--max-results", type=int, default=20, help="Maximum results")
    parser.add_argument("--min-profit", type=float, default=0.02, help="Minimum profit threshold")
    
    args = parser.parse_args()
    
    # Create scanner
    scanner = create_scanner(args.config)
    
    # Configure scan
    scan_config = ScanConfiguration(
        max_results=args.max_results,
        min_profit_threshold=args.min_profit
    )
    
    # Run scan
    print("Starting arbitrage scan...")
    result = run_scan(scanner, scan_config)
    
    # Display results
    print(f"\nScan Results:")
    print(f"- Total opportunities: {len(result.opportunities)}")
    print(f"- Trading signals: {len(result.trading_signals)}")
    print(f"- Scan time: {result.performance_metrics.total_scan_time:.2f}s")
    print(f"- Options processed: {result.total_options_processed}")
    print(f"- Success rate: {result.success_rate:.2%}")
    
    if result.opportunities:
        print(f"\nTop 5 Opportunities:")
        for i, opp in enumerate(result.opportunities[:5], 1):
            print(f"{i}. {opp.strategy_type.value}: {opp.description}")
            print(f"   Expected profit: {opp.expected_profit:.4f}")
            print(f"   Probability: {opp.probability:.2%}")
            print(f"   Risk level: {opp.risk_level.value}")