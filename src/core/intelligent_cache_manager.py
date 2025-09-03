"""
Complete intelligent cache manager implementation with multi-tier storage.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional, List, Any, Callable
from datetime import datetime

from .cache_manager import IntelligentCacheManager, CacheTier, DataType, CacheEntry
from ..cache.memory_cache import MemoryCache
from ..cache.disk_cache import DiskCache
from ..cache.redis_cache import RedisCache
from ..cache.cache_strategies import TradingAwareCacheStrategy


class TradingCacheManager(IntelligentCacheManager):
    """
    Complete trading-focused intelligent cache manager.
    
    Integrates all cache tiers with trading-aware strategies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trading cache manager.
        
        Args:
            config: Cache configuration
        """
        super().__init__(config)
        
        # Use trading-aware strategy
        self.strategy = TradingAwareCacheStrategy()
        
        # Initialize cache tiers based on configuration
        self._initialize_cache_tiers()
        
        self.logger.info("Trading cache manager initialized")
    
    def _initialize_cache_tiers(self) -> None:
        """Initialize cache storage tiers."""
        # L1 Memory Cache - Always available
        memory_config = self.config.get('memory', {})
        max_entries = memory_config.get('max_entries', 10000)
        max_size_mb = memory_config.get('max_size_mb', 512)
        
        self.storage_tiers[CacheTier.L1_MEMORY] = MemoryCache(
            max_entries=max_entries,
            max_size_mb=max_size_mb
        )
        self.logger.info(f"L1 Memory cache initialized: {max_entries} entries, {max_size_mb}MB")
        
        # L2 Disk Cache - If enabled
        if self.config.get('disk', {}).get('enabled', True):
            disk_config = self.config.get('disk', {})
            cache_dir = disk_config.get('cache_dir') or str(Path(tempfile.gettempdir()) / "options_cache")
            max_size_gb = disk_config.get('max_size_gb', 2.0)
            compression_enabled = disk_config.get('compression_enabled', True)
            
            try:
                self.storage_tiers[CacheTier.L2_DISK] = DiskCache(
                    cache_dir=cache_dir,
                    max_size_gb=max_size_gb,
                    compression_enabled=compression_enabled
                )
                self.logger.info(f"L2 Disk cache initialized: {cache_dir}, {max_size_gb}GB")
            except Exception as e:
                self.logger.warning(f"Failed to initialize disk cache: {e}")
        
        # L3 Redis Cache - If available and enabled
        if self.config.get('redis', {}).get('enabled', False):
            redis_config = self.config.get('redis', {})
            redis_url = redis_config.get('url', 'redis://localhost:6379')
            key_prefix = redis_config.get('key_prefix', 'options_cache:')
            
            try:
                self.storage_tiers[CacheTier.L3_REDIS] = RedisCache(
                    redis_url=redis_url,
                    key_prefix=key_prefix,
                    compression_enabled=redis_config.get('compression_enabled', True),
                    pool_size=redis_config.get('pool_size', 10)
                )
                self.logger.info(f"L3 Redis cache initialized: {redis_url}")
            except ImportError:
                self.logger.warning("Redis cache unavailable - install redis package")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Redis cache: {e}")
    
    async def initialize(self) -> None:
        """Initialize cache manager and strategy."""
        await super().initialize()
        
        # Initialize strategy
        if hasattr(self.strategy, 'initialize'):
            await self.strategy.initialize()
    
    async def shutdown(self) -> None:
        """Shutdown cache manager."""
        # Shutdown strategy
        if hasattr(self.strategy, 'shutdown'):
            await self.strategy.shutdown()
        
        await super().shutdown()
    
    async def get_with_loader(self, 
                              key: str, 
                              data_loader: Callable[[], Any],
                              data_type: DataType = DataType.REAL_TIME,
                              ttl_seconds: Optional[int] = None) -> Optional[Any]:
        """
        Get data with automatic loading and caching if not found.
        
        Args:
            key: Cache key
            data_loader: Function to load data if not cached
            data_type: Data type for strategy decisions
            ttl_seconds: Optional TTL override
            
        Returns:
            Cached or loaded data
        """
        # Try to get from cache first
        cached_data = await self.get(key, data_type)
        if cached_data is not None:
            # Record access for pattern analysis
            if hasattr(self.strategy, 'record_access'):
                await self.strategy.record_access(key)
            return cached_data
        
        # Load data
        try:
            data = await data_loader() if asyncio.iscoroutinefunction(data_loader) else data_loader()
            if data is not None:
                # Cache the loaded data
                await self.set(key, data, data_type, ttl_seconds)
                if hasattr(self.strategy, 'record_access'):
                    await self.strategy.record_access(key)
                return data
        except Exception as e:
            self.logger.error(f"Error loading data for key '{key}': {e}")
        
        return None
    
    async def bulk_get(self, keys: List[str], data_type: DataType = DataType.REAL_TIME) -> Dict[str, Any]:
        """
        Get multiple keys efficiently.
        
        Args:
            keys: List of cache keys
            data_type: Data type for strategy decisions
            
        Returns:
            Dict mapping keys to cached data (only includes found keys)
        """
        results = {}
        
        # Use asyncio.gather for concurrent lookups
        tasks = [self.get(key, data_type) for key in keys]
        cached_values = await asyncio.gather(*tasks, return_exceptions=True)
        
        for key, value in zip(keys, cached_values):
            if not isinstance(value, Exception) and value is not None:
                results[key] = value
        
        return results
    
    async def bulk_set(self, 
                       items: Dict[str, Any],
                       data_type: DataType = DataType.REAL_TIME,
                       ttl_seconds: Optional[int] = None) -> int:
        """
        Set multiple keys efficiently.
        
        Args:
            items: Dict mapping keys to data
            data_type: Data type for strategy decisions
            ttl_seconds: Optional TTL override
            
        Returns:
            Number of successfully cached items
        """
        tasks = [self.set(key, data, data_type, ttl_seconds) for key, data in items.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for result in results if result is True)
        return success_count
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching a pattern.
        
        Args:
            pattern: Key pattern (supports wildcards)
            
        Returns:
            Number of invalidated keys
        """
        invalidated_count = 0
        
        for storage in self.storage_tiers.values():
            try:
                keys = await storage.keys()
                matching_keys = [key for key in keys if self._matches_pattern(key, pattern)]
                
                for key in matching_keys:
                    if await storage.delete(key):
                        invalidated_count += 1
                        
            except Exception as e:
                self.logger.warning(f"Error invalidating pattern '{pattern}': {e}")
        
        return invalidated_count
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Simple pattern matching with wildcard support."""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    async def preload_strategies(self, strategies: List[str]) -> int:
        """
        Preload cache for specific trading strategies.
        
        Args:
            strategies: List of strategy names to preload
            
        Returns:
            Number of preloaded entries
        """
        if not self.preload_enabled:
            return 0
        
        preload_count = 0
        
        for strategy_name in strategies:
            try:
                # Generate preload keys based on strategy
                keys_to_preload = self._generate_strategy_keys(strategy_name)
                
                # Create a dummy loader that returns None (triggers actual loading elsewhere)
                async def dummy_loader():
                    return None
                
                # Warm cache with the keys
                preload_count += await self.warm_cache(dummy_loader, keys_to_preload)
                
            except Exception as e:
                self.logger.error(f"Error preloading strategy '{strategy_name}': {e}")
        
        return preload_count
    
    def _generate_strategy_keys(self, strategy_name: str) -> List[str]:
        """Generate cache keys for a trading strategy."""
        # This is a simplified implementation
        # In practice, this would integrate with strategy configurations
        base_key = f"strategy:{strategy_name}"
        
        keys = [
            f"{base_key}:options:current",
            f"{base_key}:options:near_expiry",
            f"{base_key}:underlying:prices",
            f"{base_key}:volatility:surface",
            f"{base_key}:greeks:current"
        ]
        
        return keys
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics including strategy analytics."""
        base_stats = self.get_statistics()
        
        # Add strategy analytics
        if hasattr(self.strategy, 'get_analytics'):
            base_stats['strategy_analytics'] = self.strategy.get_analytics()
        
        # Add tier-specific statistics
        tier_details = {}
        for tier_name, storage in self.storage_tiers.items():
            if hasattr(storage, 'get_statistics'):
                tier_details[tier_name.value] = storage.get_statistics()
        
        base_stats['tier_details'] = tier_details
        
        return base_stats
    
    async def health_check_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive health check for all cache tiers."""
        health_info = {
            'overall_status': 'healthy',
            'tiers': {},
            'strategy_status': 'active',
            'performance_metrics': self.get_statistics()
        }
        
        # Check each tier
        tier_issues = []
        for tier_name, storage in self.storage_tiers.items():
            tier_info = {'status': 'healthy', 'available': True}
            
            try:
                # Basic operations test
                test_key = f"health_check_{tier_name.value}"
                test_data = {'timestamp': datetime.now().isoformat()}
                
                # Test set/get/delete cycle
                test_entry = CacheEntry(
                    key=test_key,
                    data=test_data,
                    data_type=DataType.REFERENCE,
                    created_at=datetime.now(),
                    accessed_at=datetime.now(),
                    ttl_seconds=60
                )
                
                set_success = await storage.set(test_entry)
                if set_success:
                    get_result = await storage.get(test_key)
                    delete_success = await storage.delete(test_key)
                    
                    if not get_result or not delete_success:
                        tier_info['status'] = 'degraded'
                        tier_issues.append(f"{tier_name.value}: read/delete issues")
                else:
                    tier_info['status'] = 'error'
                    tier_issues.append(f"{tier_name.value}: write failed")
                
                # Add tier-specific health info
                if hasattr(storage, 'health_check'):
                    tier_health = await storage.health_check()
                    tier_info['specific_health'] = tier_health
                
            except Exception as e:
                tier_info['status'] = 'error'
                tier_info['error'] = str(e)
                tier_issues.append(f"{tier_name.value}: {str(e)}")
            
            health_info['tiers'][tier_name.value] = tier_info
        
        # Overall status assessment
        if tier_issues:
            if len(tier_issues) == len(self.storage_tiers):
                health_info['overall_status'] = 'critical'
            else:
                health_info['overall_status'] = 'degraded'
            health_info['issues'] = tier_issues
        
        return health_info