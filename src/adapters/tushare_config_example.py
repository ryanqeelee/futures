"""
TushareAdapter Configuration Examples and Best Practices
Provides production-ready configuration templates for different use cases.
"""

from datetime import timedelta
from typing import Dict, Any


class TushareConfigTemplates:
    """
    Configuration templates for different TushareAdapter use cases.
    
    Provides optimized configurations for development, testing, and production
    environments with appropriate performance and quality settings.
    """
    
    @staticmethod
    def development_config() -> Dict[str, Any]:
        """
        Configuration optimized for development and testing.
        
        Features:
        - Lower rate limits for API conservation
        - More verbose logging and validation
        - Shorter cache TTLs for fresh data
        - Conservative quality thresholds
        """
        return {
            # Basic adapter settings
            'api_token': None,  # Will be loaded from TUSHARE_TOKEN env var
            'timeout': 30,
            'retry_count': 2,
            'retry_delay': 1.0,
            'rate_limit': 60,   # Conservative for development
            'batch_size': 25,   # Smaller batches
            'max_days_back': 3,
            
            # Data validation thresholds (conservative)
            'min_price_threshold': 0.01,
            'max_price_threshold': 10000,
            'min_volume_threshold': 1,
            'max_iv_threshold': 3.0,  # 300% max IV
            
            # Quality validation configuration
            'quality_config': {
                'min_price': 0.01,
                'max_price': 10000,
                'max_price_change_pct': 0.3,  # 30% max price deviation
                'min_volume': 1,
                'max_volume': 100000,
                'min_iv': 0.02,    # 2% minimum IV
                'max_iv': 3.0,     # 300% maximum IV
                'min_delta': -1.0,
                'max_delta': 1.0,
                'max_gamma': 5.0,
                'max_vega': 50.0,
                'max_theta': 5.0,
                'max_data_age_hours': 48,
                'outlier_std_threshold': 2.0,  # Conservative outlier detection
            },
            
            # Monitor configuration
            'monitor_config': {
                'alert_threshold': 0.7,
                'history_limit': 100
            }
        }
    
    @staticmethod
    def production_config() -> Dict[str, Any]:
        """
        Configuration optimized for production trading systems.
        
        Features:
        - Higher rate limits for maximum throughput
        - Optimized caching for performance
        - Balanced quality thresholds
        - Production-grade error handling
        """
        return {
            # Basic adapter settings (production optimized)
            'api_token': None,  # Will be loaded from TUSHARE_TOKEN env var
            'timeout': 45,
            'retry_count': 3,
            'retry_delay': 0.5,
            'rate_limit': 120,  # Maximum sustainable rate
            'batch_size': 100,  # Larger batches for efficiency
            'max_days_back': 5,
            
            # Data validation thresholds (production balanced)
            'min_price_threshold': 0.01,
            'max_price_threshold': 50000,
            'min_volume_threshold': 0,  # Allow zero volume for monitoring
            'max_iv_threshold': 5.0,    # 500% max IV for extreme scenarios
            
            # Quality validation configuration
            'quality_config': {
                'min_price': 0.001,
                'max_price': 50000,
                'max_price_change_pct': 0.5,  # 50% max price deviation
                'min_volume': 0,
                'max_volume': 1000000,
                'min_iv': 0.01,    # 1% minimum IV
                'max_iv': 5.0,     # 500% maximum IV
                'min_delta': -1.2,  # Allow slight overflow for edge cases
                'max_delta': 1.2,
                'max_gamma': 20.0,
                'max_vega': 200.0,
                'max_theta': 20.0,
                'max_data_age_hours': 24,
                'outlier_std_threshold': 3.0,  # Standard outlier detection
            },
            
            # Monitor configuration
            'monitor_config': {
                'alert_threshold': 0.8,
                'history_limit': 1000
            }
        }
    
    @staticmethod
    def high_frequency_config() -> Dict[str, Any]:
        """
        Configuration optimized for high-frequency trading systems.
        
        Features:
        - Maximum rate limits and concurrent processing
        - Aggressive caching for ultra-low latency
        - Relaxed quality thresholds for speed
        - Minimal logging for performance
        """
        return {
            # Basic adapter settings (high frequency optimized)
            'api_token': None,
            'timeout': 15,      # Shorter timeout for speed
            'retry_count': 1,   # Minimal retries
            'retry_delay': 0.1,
            'rate_limit': 200,  # Push limits for HFT
            'batch_size': 200,  # Maximum batch size
            'max_days_back': 2, # Recent data only
            
            # Data validation thresholds (relaxed for speed)
            'min_price_threshold': 0.001,
            'max_price_threshold': 100000,
            'min_volume_threshold': 0,
            'max_iv_threshold': 10.0,  # Very high for edge cases
            
            # Quality validation configuration (minimal for speed)
            'quality_config': {
                'min_price': 0.001,
                'max_price': 100000,
                'max_price_change_pct': 1.0,  # 100% max price deviation
                'min_volume': 0,
                'max_volume': 10000000,
                'min_iv': 0.005,   # 0.5% minimum IV
                'max_iv': 10.0,    # 1000% maximum IV
                'min_delta': -2.0,
                'max_delta': 2.0,
                'max_gamma': 100.0,
                'max_vega': 1000.0,
                'max_theta': 100.0,
                'max_data_age_hours': 12,
                'outlier_std_threshold': 4.0,  # Very relaxed outlier detection
            },
            
            # Monitor configuration (minimal)
            'monitor_config': {
                'alert_threshold': 0.5,  # Lower threshold for HFT
                'history_limit': 500
            }
        }
    
    @staticmethod
    def research_config() -> Dict[str, Any]:
        """
        Configuration optimized for research and backtesting.
        
        Features:
        - Conservative rate limits to preserve API quota
        - Comprehensive data quality validation
        - Extended historical data lookback
        - Detailed error reporting and validation
        """
        return {
            # Basic adapter settings (research optimized)
            'api_token': None,
            'timeout': 60,      # Longer timeout for large datasets
            'retry_count': 5,   # More retries for reliability
            'retry_delay': 2.0,
            'rate_limit': 30,   # Conservative to preserve quota
            'batch_size': 50,
            'max_days_back': 10, # Extended historical lookback
            
            # Data validation thresholds (strict for research)
            'min_price_threshold': 0.01,
            'max_price_threshold': 20000,
            'min_volume_threshold': 1,
            'max_iv_threshold': 2.0,  # 200% max IV for research
            
            # Quality validation configuration (comprehensive)
            'quality_config': {
                'min_price': 0.01,
                'max_price': 20000,
                'max_price_change_pct': 0.2,  # 20% max price deviation
                'min_volume': 1,
                'max_volume': 500000,
                'min_iv': 0.05,    # 5% minimum IV
                'max_iv': 2.0,     # 200% maximum IV
                'min_delta': -0.99,
                'max_delta': 0.99,
                'max_gamma': 10.0,
                'max_vega': 100.0,
                'max_theta': 10.0,
                'max_data_age_hours': 72,  # Accept older data for research
                'outlier_std_threshold': 1.5,  # Strict outlier detection
            },
            
            # Monitor configuration (comprehensive)
            'monitor_config': {
                'alert_threshold': 0.9,  # High threshold for research quality
                'history_limit': 2000
            }
        }
    
    @staticmethod
    def get_config_by_environment(environment: str = 'development') -> Dict[str, Any]:
        """
        Get configuration by environment name.
        
        Args:
            environment: Environment name ('development', 'production', 'hft', 'research')
            
        Returns:
            Dict with configuration for the specified environment
            
        Raises:
            ValueError: If environment is not supported
        """
        configs = {
            'development': TushareConfigTemplates.development_config(),
            'production': TushareConfigTemplates.production_config(),
            'hft': TushareConfigTemplates.high_frequency_config(),
            'research': TushareConfigTemplates.research_config()
        }
        
        if environment not in configs:
            raise ValueError(f"Unsupported environment: {environment}. "
                           f"Available: {list(configs.keys())}")
        
        return configs[environment]
    
    @staticmethod
    def create_custom_config(base_environment: str = 'production', **overrides) -> Dict[str, Any]:
        """
        Create custom configuration based on a base environment.
        
        Args:
            base_environment: Base configuration to start with
            **overrides: Configuration overrides
            
        Returns:
            Dict with custom configuration
            
        Example:
            config = TushareConfigTemplates.create_custom_config(
                'production',
                rate_limit=150,
                quality_config={'min_price': 0.005}
            )
        """
        config = TushareConfigTemplates.get_config_by_environment(base_environment).copy()
        
        for key, value in overrides.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                # Deep merge for nested dicts
                config[key].update(value)
            else:
                config[key] = value
        
        return config


class TushareConfigValidator:
    """
    Validator for TushareAdapter configurations.
    
    Provides validation and recommendations for configuration parameters
    to ensure optimal performance and reliability.
    """
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration parameters and provide recommendations.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Dict with validation results and recommendations
        """
        issues = []
        warnings = []
        recommendations = []
        
        # Validate basic parameters
        if config.get('rate_limit', 0) > 200:
            warnings.append("Rate limit > 200 may exceed API limits")
        
        if config.get('batch_size', 0) > 500:
            warnings.append("Large batch size may cause memory issues")
        
        if config.get('timeout', 0) < 10:
            warnings.append("Timeout < 10s may cause premature failures")
        
        # Validate quality config
        quality_config = config.get('quality_config', {})
        if quality_config.get('min_price', 0) <= 0:
            issues.append("Minimum price must be positive")
        
        if quality_config.get('max_iv', 0) > 10:
            warnings.append("Maximum IV > 1000% may be too permissive")
        
        # Generate recommendations
        if config.get('rate_limit', 0) < 60:
            recommendations.append("Consider increasing rate_limit for better performance")
        
        if not quality_config:
            recommendations.append("Add quality_config for data validation")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'recommendations': recommendations,
            'score': max(0, 100 - len(issues) * 30 - len(warnings) * 10)
        }
    
    @staticmethod
    def recommend_config(use_case: str, data_volume: str = 'medium', 
                        quality_priority: str = 'balanced') -> Dict[str, Any]:
        """
        Recommend configuration based on use case and requirements.
        
        Args:
            use_case: 'trading', 'research', 'monitoring', 'backtesting'
            data_volume: 'low', 'medium', 'high'
            quality_priority: 'speed', 'balanced', 'quality'
            
        Returns:
            Dict with recommended configuration
        """
        # Base configuration selection
        if use_case == 'trading':
            if data_volume == 'high':
                base_config = TushareConfigTemplates.high_frequency_config()
            else:
                base_config = TushareConfigTemplates.production_config()
        elif use_case == 'research':
            base_config = TushareConfigTemplates.research_config()
        else:
            base_config = TushareConfigTemplates.development_config()
        
        # Adjust for quality priority
        if quality_priority == 'speed':
            # Optimize for speed
            base_config['retry_count'] = min(base_config.get('retry_count', 3), 2)
            base_config['quality_config']['outlier_std_threshold'] = 4.0
        elif quality_priority == 'quality':
            # Optimize for quality
            base_config['retry_count'] = max(base_config.get('retry_count', 3), 3)
            base_config['quality_config']['outlier_std_threshold'] = 1.5
        
        # Adjust for data volume
        if data_volume == 'high':
            base_config['batch_size'] = min(base_config.get('batch_size', 100) * 2, 500)
            base_config['rate_limit'] = min(base_config.get('rate_limit', 120) * 1.5, 200)
        elif data_volume == 'low':
            base_config['batch_size'] = max(base_config.get('batch_size', 100) // 2, 10)
            base_config['rate_limit'] = max(base_config.get('rate_limit', 120) // 2, 30)
        
        return base_config


# Usage examples and best practices
USAGE_EXAMPLES = {
    'basic_usage': '''
# Basic usage with default configuration
from src.adapters.tushare_adapter import TushareAdapter
from src.adapters.tushare_config_example import TushareConfigTemplates

config = TushareConfigTemplates.development_config()
adapter = TushareAdapter(config)

# Connect and get data
await adapter.connect()
request = DataRequest(max_days_to_expiry=30)
response = await adapter.get_option_data(request)
''',

    'production_setup': '''
# Production setup with monitoring
config = TushareConfigTemplates.production_config()
adapter = TushareAdapter(config)

# Regular health checks
health_info = await adapter.health_check_comprehensive()
if health_info['data_quality']['test_successful']:
    print("System healthy")

# Monitor performance
metrics = adapter.get_performance_metrics()
print(f"Cache hit rate: {metrics['cache_hit_rate']:.1%}")
''',

    'custom_configuration': '''
# Custom configuration for specific needs
config = TushareConfigTemplates.create_custom_config(
    'production',
    rate_limit=150,  # Custom rate limit
    quality_config={
        'min_iv': 0.02,  # Custom minimum IV
        'outlier_std_threshold': 2.5
    }
)

# Validate before use
validation = TushareConfigValidator.validate_config(config)
if validation['valid']:
    adapter = TushareAdapter(config)
else:
    print("Configuration issues:", validation['issues'])
''',

    'research_workflow': '''
# Research workflow with comprehensive validation
config = TushareConfigTemplates.research_config()
adapter = TushareAdapter(config)

# Get historical data with full validation
request = DataRequest(
    max_days_to_expiry=60,
    include_iv=True,
    include_greeks=True,
    as_of_date=date(2024, 1, 15)  # Historical date
)

response = await adapter.get_option_data(request)

# Analyze quality
quality_report = adapter.get_data_quality_report()
print(f"Data quality: {quality_report['quality_trend']['current_quality']}")
print(f"Validation errors: {quality_report['current_metrics']['total_errors']}")
'''
}


if __name__ == "__main__":
    # Demonstrate configuration templates
    print("TushareAdapter Configuration Templates")
    print("=" * 50)
    
    for env_name in ['development', 'production', 'hft', 'research']:
        config = TushareConfigTemplates.get_config_by_environment(env_name)
        validation = TushareConfigValidator.validate_config(config)
        
        print(f"\n{env_name.upper()} Configuration:")
        print(f"  Rate limit: {config['rate_limit']} req/min")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Quality score: {validation['score']}/100")
        print(f"  Recommendations: {len(validation['recommendations'])}")
    
    print("\n" + "=" * 50)
    print("Ready to use with TushareAdapter!")