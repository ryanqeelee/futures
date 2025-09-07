"""
Configuration models using Pydantic for type safety and validation.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DataSourceType(str, Enum):
    """Supported data source types."""
    TUSHARE = "tushare"
    WIND = "wind"
    EASTMONEY = "eastmoney"
    # MOCK data source removed for production security


class StrategyType(str, Enum):
    """Arbitrage strategy types."""
    PRICING_ARBITRAGE = "pricing_arbitrage"
    PUT_CALL_PARITY = "put_call_parity"
    VOLATILITY_ARBITRAGE = "volatility_arbitrage"
    CALENDAR_SPREAD = "calendar_spread"
    PREDICTION_BASED = "prediction_based"  # 新增：基于价格预测的期权交易策略


class DatabaseConfig(BaseModel):
    """Database configuration."""
    model_config = ConfigDict(extra='forbid')
    
    url: str = Field(..., description="Database connection URL")
    pool_size: int = Field(10, ge=1, le=50, description="Connection pool size")
    max_overflow: int = Field(20, ge=0, le=100, description="Max pool overflow")
    pool_timeout: int = Field(30, ge=1, description="Pool timeout in seconds")
    echo: bool = Field(False, description="Enable SQL query logging")


class CacheConfig(BaseModel):
    """Cache configuration."""
    model_config = ConfigDict(extra='forbid')
    
    enabled: bool = Field(True, description="Enable caching")
    backend: str = Field("memory", pattern="^(memory|redis|memcached)$")
    ttl_seconds: int = Field(300, ge=1, description="Default TTL in seconds")
    max_size: int = Field(1000, ge=1, description="Maximum cache entries")
    redis_url: Optional[str] = Field(None, description="Redis connection URL")


class DataSourceConfig(BaseModel):
    """Data source configuration."""
    model_config = ConfigDict(extra='forbid')
    
    type: DataSourceType = Field(..., description="Data source type")
    enabled: bool = Field(True, description="Enable this data source")
    priority: int = Field(1, ge=1, le=10, description="Priority for selection")
    timeout: int = Field(30, ge=1, description="Request timeout in seconds")
    retry_count: int = Field(3, ge=0, description="Retry attempts")
    retry_delay: float = Field(1.0, ge=0, description="Delay between retries")
    rate_limit: Optional[int] = Field(None, ge=1, description="Requests per second limit")
    
    # Data source specific configs
    api_token: Optional[str] = Field(None, description="API authentication token")
    base_url: Optional[str] = Field(None, description="Base API URL")
    extra_params: Dict[str, Any] = Field(default_factory=dict, description="Extra parameters")


class RiskConfig(BaseModel):
    """Risk management configuration."""
    model_config = ConfigDict(extra='forbid')
    
    max_position_size: float = Field(100000.0, ge=0, description="Maximum position size")
    max_daily_loss: float = Field(10000.0, ge=0, description="Maximum daily loss")
    min_liquidity_volume: int = Field(100, ge=0, description="Minimum volume for liquidity")
    max_concentration: float = Field(0.3, ge=0, le=1.0, description="Maximum position concentration")
    
    # Time-based limits
    max_days_to_expiry: int = Field(90, ge=1, description="Maximum days to option expiry")
    min_days_to_expiry: int = Field(1, ge=1, description="Minimum days to option expiry")
    
    @field_validator('min_days_to_expiry')
    @classmethod
    def validate_expiry_range(cls, v, info):
        if 'max_days_to_expiry' in info.data and v >= info.data['max_days_to_expiry']:
            raise ValueError('min_days_to_expiry must be less than max_days_to_expiry')
        return v


class ParameterConstraint(BaseModel):
    """Parameter constraint definition."""
    model_config = ConfigDict(extra='forbid')
    
    min_value: Optional[float] = Field(None, description="Minimum allowed value")
    max_value: Optional[float] = Field(None, description="Maximum allowed value") 
    step: Optional[float] = Field(None, description="Step increment")
    choices: Optional[List[Any]] = Field(None, description="Valid choices")
    required: bool = Field(True, description="Whether parameter is required")
    

class ParameterDefinition(BaseModel):
    """Strategy parameter definition with metadata."""
    model_config = ConfigDict(extra='forbid')
    
    name: str = Field(..., description="Parameter name")
    display_name: str = Field(..., description="User-friendly display name")
    description: str = Field(..., description="Parameter description")
    parameter_type: str = Field(..., pattern="^(float|int|bool|str|choice)$", description="Parameter data type")
    default_value: Any = Field(..., description="Default parameter value")
    constraint: Optional[ParameterConstraint] = Field(None, description="Parameter constraints")
    category: str = Field("general", description="Parameter category")
    advanced: bool = Field(False, description="Whether this is an advanced parameter")


class StrategyParameterSet(BaseModel):
    """Complete parameter set for a strategy."""
    model_config = ConfigDict(extra='forbid')
    
    strategy_type: StrategyType = Field(..., description="Strategy type")
    parameter_definitions: List[ParameterDefinition] = Field(..., description="Parameter definitions")
    preset_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Preset configurations")


class StrategyConfig(BaseModel):
    """Strategy configuration."""
    model_config = ConfigDict(extra='forbid')
    
    type: StrategyType = Field(..., description="Strategy type")
    enabled: bool = Field(True, description="Enable this strategy")
    priority: int = Field(1, ge=1, le=10, description="Strategy priority")
    
    # Common parameters
    min_profit_threshold: float = Field(0.01, ge=0, description="Minimum profit threshold")
    max_risk_tolerance: float = Field(0.1, ge=0, description="Maximum risk tolerance")
    
    # Strategy-specific parameters
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    
    # Configuration metadata
    preset_name: Optional[str] = Field(None, description="Name of preset configuration used")
    custom_config: bool = Field(False, description="Whether this is a custom configuration")
    last_modified: datetime = Field(default_factory=datetime.now, description="Last modification timestamp")


class MonitoringConfig(BaseModel):
    """Monitoring and alerting configuration.""" 
    model_config = ConfigDict(extra='forbid')
    
    enabled: bool = Field(True, description="Enable monitoring")
    scan_interval: int = Field(60, ge=1, description="Scan interval in seconds")
    alert_threshold: float = Field(0.05, ge=0, description="Alert threshold for opportunities")
    
    # Notification settings
    email_enabled: bool = Field(False, description="Enable email notifications")
    email_recipients: List[str] = Field(default_factory=list, description="Email recipients")
    webhook_url: Optional[str] = Field(None, description="Webhook URL for notifications")


class PluginConfig(BaseModel):
    """Plugin management configuration."""
    model_config = ConfigDict(extra='forbid')
    
    enabled: bool = Field(True, description="Enable plugin system")
    plugin_directories: List[str] = Field(
        default_factory=lambda: ['src/strategies'], 
        description="Directories to search for plugins"
    )
    auto_reload: bool = Field(True, description="Enable automatic plugin reloading")
    reload_delay: float = Field(1.0, ge=0.1, description="Delay in seconds before reloading changed plugins")
    enable_hot_reload: bool = Field(True, description="Enable hot-reload during development")
    max_load_retries: int = Field(3, ge=1, description="Maximum plugin load retry attempts")
    retry_delay: float = Field(0.5, ge=0.1, description="Delay between retry attempts")
    validate_on_load: bool = Field(True, description="Validate plugins when loading")
    parallel_loading: bool = Field(True, description="Load plugins in parallel")
    max_load_workers: int = Field(4, ge=1, description="Maximum parallel loading workers")
    scan_interval: int = Field(30, ge=5, description="Plugin directory scan interval in seconds")
    
    # File patterns
    plugin_file_pattern: str = Field("*.py", description="File pattern for plugin discovery")
    exclude_patterns: List[str] = Field(
        default_factory=lambda: ['__pycache__', '*.pyc', 'test_*', '*_test.py'],
        description="Patterns to exclude from plugin discovery"
    )


class SystemConfig(BaseSettings):
    """Main system configuration loaded from environment."""
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='forbid'
    )
    
    # Application settings
    app_name: str = Field("Options Arbitrage Scanner", description="Application name")
    version: str = Field("1.0.0", description="Application version")
    environment: str = Field("development", pattern="^(development|staging|production)$")
    debug: bool = Field(False, description="Enable debug mode")
    log_level: LogLevel = Field(LogLevel.INFO, description="Logging level")
    
    # Server settings  
    host: str = Field("localhost", description="Server host")
    port: int = Field(8000, ge=1, le=65535, description="Server port")
    workers: int = Field(1, ge=1, description="Number of worker processes")
    
    # Security
    secret_key: str = Field(default="development_secret_key_32chars_min", min_length=32, description="Secret key for security")
    api_key: Optional[str] = Field(None, description="API key for external access")
    
    # Data source API keys
    tushare_token: Optional[str] = Field(None, description="Tushare Pro API token")
    
    # File paths
    data_dir: Path = Field(Path("data"), description="Data directory path")
    log_dir: Path = Field(Path("logs"), description="Log directory path")
    config_dir: Path = Field(Path("config"), description="Configuration directory path")
    
    # Component configurations
    database: Optional[DatabaseConfig] = Field(None, description="Database configuration")
    cache: CacheConfig = Field(default_factory=CacheConfig, description="Cache configuration")
    risk: RiskConfig = Field(default_factory=RiskConfig, description="Risk configuration") 
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig, description="Monitoring configuration")
    plugins: PluginConfig = Field(default_factory=PluginConfig, description="Plugin system configuration")
    
    # Data sources
    data_sources: Dict[str, DataSourceConfig] = Field(default_factory=dict, description="Data source configurations")
    
    # Strategies
    strategies: Dict[str, StrategyConfig] = Field(default_factory=dict, description="Strategy configurations")
    
    @field_validator('data_dir', 'log_dir', 'config_dir', mode='before')
    @classmethod
    def convert_paths(cls, v):
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v


class ArbitrageOpportunity(BaseModel):
    """Model for arbitrage opportunities."""
    model_config = ConfigDict(extra='forbid')
    
    id: str = Field(..., description="Unique opportunity identifier")
    strategy_type: StrategyType = Field(..., description="Strategy that found this opportunity")
    timestamp: datetime = Field(default_factory=datetime.now, description="Discovery timestamp")
    
    # Instruments involved
    instruments: List[str] = Field(..., description="Option codes involved")
    underlying: str = Field(..., description="Underlying asset code")
    
    # Profit analysis
    expected_profit: float = Field(..., description="Expected profit amount")
    profit_margin: float = Field(..., description="Expected profit margin")
    confidence_score: float = Field(..., ge=0, le=1.0, description="Confidence in opportunity")
    
    # Risk metrics
    max_loss: float = Field(..., description="Maximum potential loss")
    risk_score: float = Field(..., ge=0, le=1.0, description="Risk assessment score")
    days_to_expiry: int = Field(..., ge=0, description="Days until nearest expiry")
    
    # Market data
    market_prices: Dict[str, float] = Field(..., description="Current market prices")
    theoretical_prices: Dict[str, float] = Field(default_factory=dict, description="Theoretical prices")
    volumes: Dict[str, int] = Field(default_factory=dict, description="Trading volumes")
    
    # Action plan
    actions: List[Dict[str, Any]] = Field(..., description="Required trading actions")
    
    # Metadata
    data_source: str = Field(..., description="Data source used")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters used")


class ValidationError(BaseModel):
    """Configuration validation error."""
    model_config = ConfigDict(extra='forbid')
    
    field: str = Field(..., description="Field with validation error")
    message: str = Field(..., description="Error message")
    value: Any = Field(..., description="Invalid value")