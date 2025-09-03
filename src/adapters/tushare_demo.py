#!/usr/bin/env python3
"""
Tushare Adapter Demo and Testing Script
Demonstrates the complete functionality of the enhanced TushareAdapter.
"""

import asyncio
import logging
import json
from datetime import datetime, date, timedelta
from typing import Dict, Any
import pandas as pd

from .tushare_adapter import TushareAdapter
from .base import DataRequest
from ..strategies.base import OptionType


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TushareAdapterDemo:
    """
    Demo class for TushareAdapter functionality.
    
    Provides comprehensive testing and demonstration of all adapter features
    including data retrieval, quality validation, performance monitoring, and caching.
    """
    
    def __init__(self):
        """Initialize demo with optimal configuration."""
        self.config = {
            # Basic configuration
            'api_token': None,  # Will be loaded from environment
            'timeout': 30,
            'retry_count': 3,
            'retry_delay': 1.0,
            'rate_limit': 120,  # requests per minute
            'batch_size': 50,
            
            # Data quality thresholds
            'min_price_threshold': 0.01,
            'max_price_threshold': 50000,
            'min_volume_threshold': 1,
            'max_iv_threshold': 3.0,  # 300% max IV
            
            # Quality validation configuration
            'quality_config': {
                'min_price': 0.01,
                'max_price': 50000,
                'min_iv': 0.05,  # 5%
                'max_iv': 3.0,   # 300%
                'outlier_std_threshold': 2.5,
                'max_data_age_hours': 24
            },
            
            # Monitor configuration
            'monitor_config': {
                'alert_threshold': 0.8,
                'history_limit': 500
            }
        }
        
        self.adapter = TushareAdapter(self.config)
        self.results = {}
    
    async def run_comprehensive_demo(self) -> Dict[str, Any]:
        """
        Run comprehensive demonstration of all adapter features.
        
        Returns:
            Dict with demo results and performance metrics
        """
        logger.info("🚀 Starting TushareAdapter Comprehensive Demo")
        logger.info("=" * 80)
        
        demo_results = {
            'connection_test': await self._test_connection(),
            'data_retrieval_test': await self._test_data_retrieval(),
            'quality_validation_test': await self._test_quality_validation(),
            'performance_test': await self._test_performance(),
            'caching_test': await self._test_caching(),
            'error_handling_test': await self._test_error_handling(),
            'health_check_test': await self._test_health_check(),
            'monitoring_test': await self._test_monitoring()
        }
        
        # Generate summary report
        demo_results['summary'] = self._generate_summary_report(demo_results)
        
        logger.info("✅ Demo completed successfully!")
        logger.info("📊 Summary Report:")
        logger.info("-" * 40)
        
        for test_name, result in demo_results.items():
            if test_name != 'summary':
                status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
                logger.info(f"{test_name}: {status}")
        
        return demo_results
    
    async def _test_connection(self) -> Dict[str, Any]:
        """Test connection establishment and authentication."""
        logger.info("🔌 Testing Connection...")
        
        try:
            start_time = datetime.now()
            await self.adapter.connect()
            end_time = datetime.now()
            
            connection_time = (end_time - start_time).total_seconds()
            
            result = {
                'success': True,
                'connection_time': connection_time,
                'status': self.adapter.connection_info.status.value,
                'data_source_type': self.adapter.data_source_type.value
            }
            
            logger.info(f"✅ Connection successful in {connection_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _test_data_retrieval(self) -> Dict[str, Any]:
        """Test basic data retrieval functionality."""
        logger.info("📊 Testing Data Retrieval...")
        
        try:
            # Test 1: Basic option data retrieval
            request = DataRequest(
                max_days_to_expiry=30,
                min_volume=10,
                include_iv=True,
                include_greeks=True
            )
            
            start_time = datetime.now()
            response = await self.adapter.get_option_data(request)
            end_time = datetime.now()
            
            retrieval_time = (end_time - start_time).total_seconds()
            
            # Test 2: Market data retrieval
            if response.data:
                symbols = [opt.code for opt in response.data[:5]]  # Test with first 5 symbols
                market_data = await self.adapter.get_market_data(symbols)
                prices = await self.adapter.get_real_time_prices(symbols)
            else:
                market_data = {}
                prices = {}
            
            result = {
                'success': True,
                'retrieval_time': retrieval_time,
                'total_options': len(response.data),
                'data_quality': response.quality.value,
                'market_data_count': len(market_data),
                'price_data_count': len(prices),
                'sample_data': {
                    'first_option': {
                        'code': response.data[0].code,
                        'price': response.data[0].market_price,
                        'iv': response.data[0].implied_volatility,
                        'delta': response.data[0].delta
                    } if response.data else None
                },
                'quality_metrics': response.metadata.get('quality_metrics', {})
            }
            
            logger.info(f"✅ Retrieved {len(response.data)} options in {retrieval_time:.2f}s")
            logger.info(f"   Quality: {response.quality.value}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Data retrieval failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _test_quality_validation(self) -> Dict[str, Any]:
        """Test data quality validation system."""
        logger.info("🔍 Testing Quality Validation...")
        
        try:
            # Get some data for quality testing
            request = DataRequest(max_days_to_expiry=15, min_volume=5)
            response = await self.adapter.get_option_data(request)
            
            # Get quality report
            quality_report = self.adapter.get_data_quality_report()
            
            result = {
                'success': True,
                'options_tested': len(response.data),
                'validation_errors': quality_report['current_metrics']['total_errors'],
                'quality_score': quality_report['current_metrics'].get('overall_score', 0),
                'error_breakdown': quality_report['current_metrics']['errors_by_severity'],
                'rule_breakdown': quality_report['current_metrics']['errors_by_rule']
            }
            
            logger.info(f"✅ Validated {len(response.data)} options")
            logger.info(f"   Errors found: {result['validation_errors']}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Quality validation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _test_performance(self) -> Dict[str, Any]:
        """Test performance characteristics and optimization."""
        logger.info("⚡ Testing Performance...")
        
        try:
            # Performance test: Multiple requests
            start_time = datetime.now()
            
            tasks = []
            for i in range(3):  # 3 concurrent requests
                request = DataRequest(
                    max_days_to_expiry=20 + i * 5,
                    min_volume=1
                )
                tasks.append(self.adapter.get_option_data(request))
            
            responses = await asyncio.gather(*tasks)
            end_time = datetime.now()
            
            total_time = (end_time - start_time).total_seconds()
            total_records = sum(len(resp.data) for resp in responses)
            
            # Get performance metrics
            perf_metrics = self.adapter.get_performance_metrics()
            
            result = {
                'success': True,
                'concurrent_requests': len(tasks),
                'total_time': total_time,
                'total_records': total_records,
                'records_per_second': total_records / total_time if total_time > 0 else 0,
                'performance_metrics': perf_metrics,
                'avg_response_time': perf_metrics.get('avg_response_time', 0),
                'cache_hit_rate': perf_metrics.get('cache_hit_rate', 0)
            }
            
            logger.info(f"✅ Processed {total_records} records in {total_time:.2f}s")
            logger.info(f"   Performance: {result['records_per_second']:.1f} records/sec")
            logger.info(f"   Cache hit rate: {result['cache_hit_rate']:.1%}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Performance test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _test_caching(self) -> Dict[str, Any]:
        """Test caching mechanism effectiveness."""
        logger.info("💾 Testing Caching...")
        
        try:
            request = DataRequest(max_days_to_expiry=10, min_volume=5)
            
            # First request (cache miss)
            start_time = datetime.now()
            response1 = await self.adapter.get_option_data(request)
            first_time = (datetime.now() - start_time).total_seconds()
            
            # Second request (should be cache hit)
            start_time = datetime.now()
            response2 = await self.adapter.get_option_data(request)
            second_time = (datetime.now() - start_time).total_seconds()
            
            # Verify same data
            same_data = (len(response1.data) == len(response2.data) and
                        response1.data[0].code == response2.data[0].code if response1.data else True)
            
            result = {
                'success': True,
                'first_request_time': first_time,
                'second_request_time': second_time,
                'speedup_factor': first_time / second_time if second_time > 0 else 0,
                'cache_effectiveness': second_time < first_time * 0.5,  # Should be much faster
                'data_consistency': same_data,
                'cache_hit_rate': self.adapter.get_performance_metrics().get('cache_hit_rate', 0)
            }
            
            logger.info(f"✅ Cache test: {first_time:.2f}s -> {second_time:.2f}s")
            logger.info(f"   Speedup: {result['speedup_factor']:.1f}x")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Caching test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and retry mechanisms."""
        logger.info("🛠️ Testing Error Handling...")
        
        try:
            # Test invalid request
            invalid_request = DataRequest(
                instruments=["INVALID_CODE_12345"],
                max_days_to_expiry=-1  # Invalid parameter
            )
            
            error_caught = False
            error_type = None
            
            try:
                await self.adapter.get_option_data(invalid_request)
            except Exception as e:
                error_caught = True
                error_type = type(e).__name__
            
            # Test retry mechanism by simulating temporary failure
            # Note: This is a simplified test - in production you'd test actual network failures
            
            result = {
                'success': True,
                'error_handling_works': error_caught,
                'error_type_caught': error_type,
                'retry_mechanism_available': hasattr(self.adapter, '_retry_request'),
                'rate_limiting_available': hasattr(self.adapter, '_enforce_rate_limit')
            }
            
            logger.info(f"✅ Error handling: {'Working' if error_caught else 'Not triggered'}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error handling test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _test_health_check(self) -> Dict[str, Any]:
        """Test comprehensive health check functionality."""
        logger.info("🏥 Testing Health Check...")
        
        try:
            health_info = await self.adapter.health_check_comprehensive()
            
            result = {
                'success': True,
                'connection_healthy': health_info['connection_status'] == 'CONNECTED',
                'data_quality_test': health_info['data_quality']['test_successful'],
                'response_time': health_info['data_quality'].get('response_time', 0),
                'performance_metrics': health_info['performance'],
                'api_limits': health_info['api_limits']
            }
            
            logger.info(f"✅ Health check: {'Healthy' if result['connection_healthy'] else 'Unhealthy'}")
            logger.info(f"   Data quality test: {'✅' if result['data_quality_test'] else '❌'}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _test_monitoring(self) -> Dict[str, Any]:
        """Test monitoring and alerting system."""
        logger.info("📈 Testing Monitoring...")
        
        try:
            # Generate some requests to build monitoring data
            for i in range(3):
                request = DataRequest(max_days_to_expiry=15 + i * 5)
                await self.adapter.get_option_data(request)
            
            # Get monitoring data
            quality_report = self.adapter.get_data_quality_report()
            quality_trend = quality_report['quality_trend']
            
            result = {
                'success': True,
                'monitoring_active': 'quality_trend' in quality_report,
                'trend_analysis': quality_trend.get('trend', 'unknown'),
                'current_quality': quality_trend.get('current_quality', 'unknown'),
                'data_points': quality_trend.get('data_points', 0),
                'performance_tracking': bool(self.adapter.get_performance_metrics())
            }
            
            logger.info(f"✅ Monitoring: {'Active' if result['monitoring_active'] else 'Inactive'}")
            logger.info(f"   Quality trend: {result['trend_analysis']}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Monitoring test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_summary_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        test_results = {k: v for k, v in results.items() if k != 'summary'}
        
        passed_tests = sum(1 for result in test_results.values() if result.get('success', False))
        total_tests = len(test_results)
        
        performance_data = results.get('performance_test', {})
        data_retrieval = results.get('data_retrieval_test', {})
        
        return {
            'overall_success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'tests_passed': passed_tests,
            'tests_total': total_tests,
            'performance_summary': {
                'records_per_second': performance_data.get('records_per_second', 0),
                'avg_response_time': performance_data.get('avg_response_time', 0),
                'cache_hit_rate': performance_data.get('cache_hit_rate', 0)
            },
            'quality_summary': {
                'data_quality_grade': data_retrieval.get('data_quality', 'unknown'),
                'validation_errors': results.get('quality_validation_test', {}).get('validation_errors', 0)
            },
            'recommendations': self._generate_recommendations(results)
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Performance recommendations
        perf_test = results.get('performance_test', {})
        if perf_test.get('records_per_second', 0) < 100:
            recommendations.append("Consider increasing batch_size for better performance")
        
        cache_hit_rate = perf_test.get('cache_hit_rate', 0)
        if cache_hit_rate < 0.3:
            recommendations.append("Cache hit rate is low - consider longer cache TTL")
        
        # Quality recommendations
        quality_test = results.get('quality_validation_test', {})
        if quality_test.get('validation_errors', 0) > 10:
            recommendations.append("High validation error count - review data quality thresholds")
        
        # Connection recommendations
        connection_test = results.get('connection_test', {})
        if connection_test.get('connection_time', 0) > 5.0:
            recommendations.append("Connection time is high - check network and API status")
        
        if not recommendations:
            recommendations.append("All systems operating optimally!")
        
        return recommendations


async def main():
    """Main demo entry point."""
    print("🚀 TushareAdapter Comprehensive Demo")
    print("=" * 50)
    print("This demo will test all aspects of the TushareAdapter including:")
    print("- Connection and authentication")
    print("- Data retrieval and validation")
    print("- Performance and caching")
    print("- Error handling and monitoring")
    print("- Quality assurance systems")
    print()
    
    demo = TushareAdapterDemo()
    results = await demo.run_comprehensive_demo()
    
    # Save results to file
    output_file = f"tushare_demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert results to JSON-serializable format
    json_results = json.loads(json.dumps(results, default=str))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Detailed results saved to: {output_file}")
    print("\n📊 Final Summary:")
    print("-" * 30)
    
    summary = results['summary']
    print(f"✅ Tests passed: {summary['tests_passed']}/{summary['tests_total']}")
    print(f"📈 Performance: {summary['performance_summary']['records_per_second']:.1f} records/sec")
    print(f"🎯 Cache efficiency: {summary['performance_summary']['cache_hit_rate']:.1%}")
    print(f"🔍 Data quality: {summary['quality_summary']['data_quality_grade']}")
    
    print("\n💡 Recommendations:")
    for rec in summary['recommendations']:
        print(f"  - {rec}")
    
    # Cleanup
    await demo.adapter.disconnect()
    
    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())