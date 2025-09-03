"""
Adapters module for data source adapters.
Provides unified interface for multiple data sources with adapter pattern.
"""

# New interface-based imports
from .base import (
    BaseDataAdapter, DataRequest, DataResponse, DataQuality,
    ConnectionStatus, ConnectionInfo, AdapterRegistry,
    DataSourceError, ConnectionError, AuthenticationError,
    RateLimitError, DataNotFoundError
)
from .tushare_adapter import TushareAdapter

# Legacy compatibility imports (if available)
try:
    from .base_adapter import IDataAdapter, BaseAdapter as LegacyBaseAdapter
    from .adapter_factory import AdapterFactory
    _has_legacy_adapters = True
except ImportError:
    _has_legacy_adapters = False

__all__ = [
    # New interface classes
    'BaseDataAdapter', 'DataRequest', 'DataResponse', 'DataQuality',
    'ConnectionStatus', 'ConnectionInfo', 'AdapterRegistry',
    
    # Exceptions
    'DataSourceError', 'ConnectionError', 'AuthenticationError',
    'RateLimitError', 'DataNotFoundError',
    
    # Adapter implementations
    'TushareAdapter'
]

# Add legacy exports if available
if _has_legacy_adapters:
    __all__.extend([
        'IDataAdapter', 'LegacyBaseAdapter', 'AdapterFactory'
    ])