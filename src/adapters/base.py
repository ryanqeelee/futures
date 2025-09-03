"""
Base classes and interfaces for data source adapters.
"""

from abc import ABC, abstractmethod
from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field

from ..config.models import DataSourceType
from ..strategies.base import OptionData, OptionType


class ConnectionStatus(str, Enum):
    """Data source connection status."""
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    ERROR = "ERROR"
    CONNECTING = "CONNECTING"


class DataQuality(str, Enum):
    """Data quality indicators."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    STALE = "STALE"


@dataclass
class ConnectionInfo:
    """Data source connection information."""
    source_name: str
    status: ConnectionStatus
    last_connected: Optional[datetime] = None
    last_error: Optional[str] = None
    latency_ms: Optional[float] = None
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None


@dataclass
class DataRequest:
    """Data request specification."""
    instruments: Optional[List[str]] = None  # Specific instruments, None for all
    underlying_assets: Optional[List[str]] = None  # Filter by underlying
    option_types: Optional[List[OptionType]] = None  # Filter by option type
    min_days_to_expiry: Optional[int] = None
    max_days_to_expiry: Optional[int] = None
    min_volume: Optional[int] = None
    include_greeks: bool = False
    include_iv: bool = False
    as_of_date: Optional[date] = None  # Historical data date, None for current


class DataResponse(BaseModel):
    """Data response container."""
    model_config = {'extra': 'forbid'}
    
    request: DataRequest = Field(..., description="Original request")
    data: List[OptionData] = Field(..., description="Retrieved option data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    source: str = Field(..., description="Data source identifier")
    quality: DataQuality = Field(DataQuality.HIGH, description="Data quality assessment")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @property
    def record_count(self) -> int:
        """Number of data records returned."""
        return len(self.data)
    
    @property
    def age_seconds(self) -> float:
        """Age of data in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()


class DataSourceError(Exception):
    """Base exception for data source errors."""
    pass


class ConnectionError(DataSourceError):
    """Connection related errors."""
    pass


class AuthenticationError(DataSourceError):
    """Authentication errors."""
    pass


class RateLimitError(DataSourceError):
    """Rate limit exceeded errors."""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class DataNotFoundError(DataSourceError):
    """Requested data not found."""
    pass


class BaseDataAdapter(ABC):
    """
    Abstract base class for data source adapters.
    
    Provides standardized interface for accessing different data sources
    like Tushare, Wind, etc. with connection management, caching, and
    error handling.
    """
    
    def __init__(self, config: Dict[str, Any], name: Optional[str] = None):
        """
        Initialize data adapter.
        
        Args:
            config: Adapter configuration
            name: Optional adapter name
        """
        self.config = config
        self.name = name or self.__class__.__name__
        self._connection_info = ConnectionInfo(
            source_name=self.name,
            status=ConnectionStatus.DISCONNECTED
        )
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, datetime] = {}
    
    @property
    @abstractmethod
    def data_source_type(self) -> DataSourceType:
        """Return the data source type."""
        pass
    
    @property
    def connection_info(self) -> ConnectionInfo:
        """Get current connection information."""
        return self._connection_info
    
    @property
    def is_connected(self) -> bool:
        """Check if adapter is connected."""
        return self._connection_info.status == ConnectionStatus.CONNECTED
    
    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to the data source.
        
        Raises:
            ConnectionError: If connection fails
            AuthenticationError: If authentication fails
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the data source."""
        pass
    
    @abstractmethod
    async def get_option_data(self, request: DataRequest) -> DataResponse:
        """
        Retrieve option data based on request.
        
        Args:
            request: Data request specification
            
        Returns:
            DataResponse: Retrieved data with metadata
            
        Raises:
            DataSourceError: If data retrieval fails
            RateLimitError: If rate limit exceeded
            DataNotFoundError: If no data found
        """
        pass
    
    @abstractmethod
    async def get_underlying_price(self, symbol: str, as_of_date: Optional[date] = None) -> Optional[float]:
        """
        Get underlying asset price.
        
        Args:
            symbol: Underlying asset symbol
            as_of_date: Date for historical price, None for current
            
        Returns:
            float: Underlying price or None if not found
        """
        pass
    
    async def health_check(self) -> bool:
        """
        Perform health check on the data source.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            if not self.is_connected:
                await self.connect()
            
            # Try a simple data request
            test_request = DataRequest(instruments=["TEST"])
            await self.get_option_data(test_request)
            return True
        except Exception:
            return False
    
    def validate_request(self, request: DataRequest) -> None:
        """
        Validate data request.
        
        Args:
            request: Request to validate
            
        Raises:
            ValueError: If request is invalid
        """
        if request.min_days_to_expiry is not None and request.max_days_to_expiry is not None:
            if request.min_days_to_expiry >= request.max_days_to_expiry:
                raise ValueError("min_days_to_expiry must be less than max_days_to_expiry")
        
        if request.min_volume is not None and request.min_volume < 0:
            raise ValueError("min_volume must be non-negative")
    
    def _cache_key(self, request: DataRequest) -> str:
        """Generate cache key for request."""
        import hashlib
        import json
        
        # Convert request to hashable format
        req_dict = {
            'instruments': sorted(request.instruments) if request.instruments else None,
            'underlying_assets': sorted(request.underlying_assets) if request.underlying_assets else None,
            'option_types': sorted([t.value for t in request.option_types]) if request.option_types else None,
            'min_days_to_expiry': request.min_days_to_expiry,
            'max_days_to_expiry': request.max_days_to_expiry,
            'min_volume': request.min_volume,
            'include_greeks': request.include_greeks,
            'include_iv': request.include_iv,
            'as_of_date': request.as_of_date.isoformat() if request.as_of_date else None
        }
        
        key_str = json.dumps(req_dict, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached_response(self, request: DataRequest, ttl_seconds: int = 300) -> Optional[DataResponse]:
        """
        Get cached response if available and not expired.
        
        Args:
            request: Data request
            ttl_seconds: Cache TTL in seconds
            
        Returns:
            DataResponse: Cached response or None
        """
        cache_key = self._cache_key(request)
        
        if cache_key in self._cache:
            cache_time = self._cache_ttl.get(cache_key)
            if cache_time and (datetime.now() - cache_time).total_seconds() < ttl_seconds:
                return self._cache[cache_key]
        
        return None
    
    def _cache_response(self, request: DataRequest, response: DataResponse) -> None:
        """
        Cache response data.
        
        Args:
            request: Original request
            response: Response to cache
        """
        cache_key = self._cache_key(request)
        self._cache[cache_key] = response
        self._cache_ttl[cache_key] = datetime.now()
    
    def _clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._cache_ttl.clear()
    
    def _update_connection_status(self, status: ConnectionStatus, error: Optional[str] = None) -> None:
        """
        Update connection status.
        
        Args:
            status: New connection status
            error: Error message if applicable
        """
        self._connection_info.status = status
        if status == ConnectionStatus.CONNECTED:
            self._connection_info.last_connected = datetime.now()
            self._connection_info.last_error = None
        elif error:
            self._connection_info.last_error = error
    
    def __str__(self) -> str:
        return f"{self.name}({self.data_source_type.value})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', status={self._connection_info.status})"


class AdapterRegistry:
    """
    Registry for data adapter implementations.
    
    Allows registration and discovery of adapter implementations.
    """
    
    _adapters: Dict[DataSourceType, type] = {}
    _instances: Dict[str, BaseDataAdapter] = {}
    
    @classmethod
    def register(cls, data_source_type: DataSourceType):
        """
        Decorator to register an adapter class.
        
        Args:
            data_source_type: Type of data source
            
        Returns:
            Decorator function
        """
        def decorator(adapter_class: type) -> type:
            if not issubclass(adapter_class, BaseDataAdapter):
                raise ValueError(f"Adapter {adapter_class} must inherit from BaseDataAdapter")
            
            cls._adapters[data_source_type] = adapter_class
            return adapter_class
        return decorator
    
    @classmethod
    def create_adapter(cls, data_source_type: DataSourceType, config: Dict[str, Any], name: Optional[str] = None) -> BaseDataAdapter:
        """
        Create an adapter instance.
        
        Args:
            data_source_type: Type of data source
            config: Adapter configuration
            name: Optional adapter name
            
        Returns:
            BaseDataAdapter: Adapter instance
            
        Raises:
            ValueError: If data source type not registered
        """
        if data_source_type not in cls._adapters:
            raise ValueError(f"Data source type {data_source_type} not registered")
            
        adapter_class = cls._adapters[data_source_type]
        return adapter_class(config, name)
    
    @classmethod
    def get_registered_adapters(cls) -> Dict[DataSourceType, type]:
        """
        Get all registered adapter types.
        
        Returns:
            Dict mapping data source types to their classes
        """
        return cls._adapters.copy()
    
    @classmethod
    def clear_registry(cls) -> None:
        """Clear the adapter registry (mainly for testing)."""
        cls._adapters.clear()
        cls._instances.clear()