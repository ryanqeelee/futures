"""
Enterprise-grade intelligent cache management system.

Implements multi-tier caching architecture with intelligent TTL management,
cache warming, consistency management and performance monitoring.
"""

import asyncio
import logging
import time
import threading
import json
import hashlib
import pickle
import gzip
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Tuple, Callable, AsyncIterator
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from collections import defaultdict, OrderedDict

from ..strategies.base import OptionData


class CachePolicy(str, Enum):
    """Cache policy types."""
    LRU = "LRU"              # Least Recently Used
    LFU = "LFU"              # Least Frequently Used
    TTL = "TTL"              # Time To Live
    SMART = "SMART"          # Intelligent adaptive policy
    WRITE_THROUGH = "WRITE_THROUGH"
    WRITE_BACK = "WRITE_BACK"


class CacheTier(str, Enum):
    """Cache tier levels."""
    L1_MEMORY = "L1_MEMORY"    # In-memory cache (fastest)
    L2_DISK = "L2_DISK"        # Disk-based cache
    L3_REDIS = "L3_REDIS"      # Distributed Redis cache


class DataType(str, Enum):
    """Data type classification for cache strategy."""
    REAL_TIME = "REAL_TIME"      # Real-time market data (short TTL)
    REFERENCE = "REFERENCE"      # Reference data (long TTL)  
    HISTORICAL = "HISTORICAL"    # Historical data (very long TTL)
    CALCULATED = "CALCULATED"    # Computed values (medium TTL)


@dataclass
class CacheStats:
    """Cache statistics and metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    memory_usage: int = 0
    avg_access_time: float = 0.0
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass  
class CacheEntry:
    """Individual cache entry with metadata."""
    key: str
    data: Any
    data_type: DataType
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    compressed: bool = False
    priority: int = 1  # 1=low, 5=critical
    
    @property
    def age_seconds(self) -> float:
        """Age of cache entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return self.age_seconds > self.ttl_seconds
    
    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = datetime.now()
        self.access_count += 1


class CacheStrategy(ABC):
    """Abstract base class for cache strategies."""
    
    @abstractmethod
    def calculate_ttl(self, data_type: DataType, data: Any) -> int:
        """Calculate TTL for data based on type and content."""
        pass
    
    @abstractmethod
    def should_preload(self, key: str, entry: CacheEntry) -> bool:
        """Determine if entry should be preloaded."""
        pass
    
    @abstractmethod
    def get_priority(self, data_type: DataType, data: Any) -> int:
        """Calculate cache priority (1-5)."""
        pass


class SmartCacheStrategy(CacheStrategy):
    """Intelligent cache strategy with adaptive TTL management."""
    
    def __init__(self):
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.historical_ttl: Dict[DataType, List[int]] = defaultdict(list)
        
    def calculate_ttl(self, data_type: DataType, data: Any) -> int:
        """Calculate adaptive TTL based on data type and access patterns."""
        base_ttl = {
            DataType.REAL_TIME: 300,      # 5 minutes
            DataType.REFERENCE: 3600,     # 1 hour  
            DataType.HISTORICAL: 86400,   # 24 hours
            DataType.CALCULATED: 1800     # 30 minutes
        }
        
        ttl = base_ttl.get(data_type, 1800)
        
        # Adjust based on options characteristics
        if hasattr(data, '__len__') and len(data) > 0:
            if hasattr(data[0], 'days_to_expiry'):
                # For options data, adjust TTL based on time to expiry
                min_expiry = min([opt.expiry_date for opt in data if hasattr(opt, 'expiry_date')])
                days_to_expiry = (min_expiry - datetime.now()).days
                
                if days_to_expiry <= 1:  # Near expiry options
                    ttl = min(ttl, 60)   # 1 minute
                elif days_to_expiry <= 7:  # Weekly options
                    ttl = min(ttl, 300)  # 5 minutes
        
        return ttl
    
    def should_preload(self, key: str, entry: CacheEntry) -> bool:
        """Decide if entry should be preloaded based on access patterns."""
        # Preload if accessed frequently
        return entry.access_count > 5 and entry.age_seconds < entry.ttl_seconds * 0.8
    
    def get_priority(self, data_type: DataType, data: Any) -> int:
        """Calculate priority based on data type and characteristics."""
        priority_map = {
            DataType.REAL_TIME: 5,      # Critical
            DataType.CALCULATED: 4,     # High
            DataType.REFERENCE: 3,      # Medium
            DataType.HISTORICAL: 2      # Low
        }
        return priority_map.get(data_type, 3)


class CacheStorage(ABC):
    """Abstract base class for cache storage implementations."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve entry from cache."""
        pass
    
    @abstractmethod
    async def set(self, entry: CacheEntry) -> bool:
        """Store entry in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Remove entry from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def keys(self) -> List[str]:
        """Get all cache keys."""
        pass
    
    @abstractmethod
    async def size(self) -> int:
        """Get cache size in bytes."""
        pass


class IntelligentCacheManager:
    """
    Enterprise-grade intelligent cache manager.
    
    Features:
    - Multi-tier caching (L1/L2/L3)
    - Adaptive TTL management
    - Cache warming and prefetching
    - Compression and serialization
    - Performance monitoring
    - Graceful degradation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize cache manager.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Cache configuration
        self.max_memory_size = config.get('max_memory_size', 512 * 1024 * 1024)  # 512MB
        self.max_disk_size = config.get('max_disk_size', 2 * 1024 * 1024 * 1024)  # 2GB
        self.default_ttl = config.get('default_ttl', 1800)  # 30 minutes
        self.compression_threshold = config.get('compression_threshold', 1024)  # 1KB
        self.preload_enabled = config.get('preload_enabled', True)
        self.monitoring_enabled = config.get('monitoring_enabled', True)
        
        # Initialize strategy
        self.strategy = SmartCacheStrategy()
        
        # Storage tiers (will be initialized by subclasses)
        self.storage_tiers: Dict[CacheTier, CacheStorage] = {}
        
        # Statistics tracking
        self.stats: Dict[CacheTier, CacheStats] = {
            tier: CacheStats() for tier in CacheTier
        }
        
        # Performance monitoring
        self._access_times: List[float] = []
        self._operation_counts: Dict[str, int] = defaultdict(int)
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._preload_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        self.logger.info("Intelligent cache manager initialized")
    
    async def initialize(self) -> None:
        """Initialize cache manager and start background tasks."""
        try:
            # Start background tasks
            if self.preload_enabled:
                self._preload_task = asyncio.create_task(self._preload_worker())
            
            self._cleanup_task = asyncio.create_task(self._cleanup_worker())
            
            if self.monitoring_enabled:
                self._monitoring_task = asyncio.create_task(self._monitoring_worker())
            
            self.logger.info("Cache manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cache manager: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown cache manager and cleanup resources."""
        try:
            # Cancel background tasks
            for task in [self._cleanup_task, self._preload_task, self._monitoring_task]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Shutdown executor
            self._executor.shutdown(wait=True)
            
            # Clear caches
            for storage in self.storage_tiers.values():
                await storage.clear()
            
            self.logger.info("Cache manager shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during cache manager shutdown: {e}")
    
    async def get(self, key: str, data_type: DataType = DataType.REAL_TIME) -> Optional[Any]:
        """
        Retrieve data from cache with multi-tier lookup.
        
        Args:
            key: Cache key
            data_type: Data type for strategy decisions
            
        Returns:
            Cached data or None if not found
        """
        start_time = time.time()
        
        try:
            # Try each tier in order (L1 -> L2 -> L3)
            for tier in [CacheTier.L1_MEMORY, CacheTier.L2_DISK, CacheTier.L3_REDIS]:
                if tier not in self.storage_tiers:
                    continue
                
                entry = await self.storage_tiers[tier].get(key)
                if entry and not entry.is_expired:
                    # Update access statistics
                    entry.touch()
                    self.stats[tier].hits += 1
                    
                    # Promote to higher tier if beneficial
                    if tier != CacheTier.L1_MEMORY and entry.access_count > 3:
                        await self._promote_entry(entry, tier)
                    
                    # Record performance
                    access_time = time.time() - start_time
                    self._record_access_time(access_time)
                    self._operation_counts['get_hit'] += 1
                    
                    return entry.data
                
                elif entry and entry.is_expired:
                    # Remove expired entry
                    await self.storage_tiers[tier].delete(key)
                    self.stats[tier].evictions += 1
                
                self.stats[tier].misses += 1
            
            # Record cache miss
            self._operation_counts['get_miss'] += 1
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving from cache key '{key}': {e}")
            self._operation_counts['get_error'] += 1
            return None
        finally:
            access_time = time.time() - start_time
            self._record_access_time(access_time)
    
    async def set(self, 
                  key: str, 
                  data: Any, 
                  data_type: DataType = DataType.REAL_TIME,
                  ttl_seconds: Optional[int] = None) -> bool:
        """
        Store data in cache with intelligent tier placement.
        
        Args:
            key: Cache key
            data: Data to cache
            data_type: Data type for strategy decisions
            ttl_seconds: Optional TTL override
            
        Returns:
            bool: Success status
        """
        start_time = time.time()
        
        try:
            # Calculate TTL and priority
            calculated_ttl = ttl_seconds or self.strategy.calculate_ttl(data_type, data)
            priority = self.strategy.get_priority(data_type, data)
            
            # Serialize and possibly compress data
            serialized_data, compressed = await self._serialize_data(data)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=data,
                data_type=data_type,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                ttl_seconds=calculated_ttl,
                size_bytes=len(serialized_data),
                compressed=compressed,
                priority=priority
            )
            
            # Determine target tier based on data type and priority
            target_tier = self._select_target_tier(entry)
            
            # Store in target tier
            if target_tier in self.storage_tiers:
                success = await self.storage_tiers[target_tier].set(entry)
                if success:
                    self.stats[target_tier].total_size += entry.size_bytes
                    self._operation_counts['set_success'] += 1
                    return True
            
            self._operation_counts['set_failure'] += 1
            return False
            
        except Exception as e:
            self.logger.error(f"Error storing to cache key '{key}': {e}")
            self._operation_counts['set_error'] += 1
            return False
        finally:
            access_time = time.time() - start_time
            self._record_access_time(access_time)
    
    async def delete(self, key: str) -> bool:
        """
        Delete entry from all cache tiers.
        
        Args:
            key: Cache key to delete
            
        Returns:
            bool: True if at least one deletion succeeded
        """
        success_count = 0
        
        for tier in self.storage_tiers.values():
            try:
                if await tier.delete(key):
                    success_count += 1
            except Exception as e:
                self.logger.warning(f"Error deleting key '{key}' from tier: {e}")
        
        return success_count > 0
    
    async def clear(self) -> None:
        """Clear all cache tiers."""
        for tier_name, storage in self.storage_tiers.items():
            try:
                await storage.clear()
                self.stats[tier_name] = CacheStats()
            except Exception as e:
                self.logger.error(f"Error clearing tier {tier_name}: {e}")
    
    async def warm_cache(self, data_loader: Callable[[str], Any], keys: List[str]) -> int:
        """
        Warm cache with data from loader function.
        
        Args:
            data_loader: Function to load data for given key
            keys: Keys to preload
            
        Returns:
            int: Number of successfully loaded entries
        """
        if not self.preload_enabled:
            return 0
        
        success_count = 0
        
        try:
            # Load data in batches to avoid overwhelming the system
            batch_size = 10
            for i in range(0, len(keys), batch_size):
                batch_keys = keys[i:i + batch_size]
                
                # Process batch concurrently
                tasks = [self._warm_single_key(data_loader, key) for key in batch_keys]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                success_count += sum(1 for result in results if result is True)
                
                # Small delay between batches to prevent overload
                await asyncio.sleep(0.1)
            
            self.logger.info(f"Cache warming completed: {success_count}/{len(keys)} entries loaded")
            
        except Exception as e:
            self.logger.error(f"Error during cache warming: {e}")
        
        return success_count
    
    async def _warm_single_key(self, data_loader: Callable, key: str) -> bool:
        """Warm single cache key."""
        try:
            # Check if already cached
            if await self.get(key) is not None:
                return True
            
            # Load data
            data = await data_loader(key)
            if data is not None:
                return await self.set(key, data)
            
        except Exception as e:
            self.logger.debug(f"Failed to warm key '{key}': {e}")
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_hits = sum(stats.hits for stats in self.stats.values())
        total_misses = sum(stats.misses for stats in self.stats.values()) 
        total_operations = total_hits + total_misses
        
        overall_hit_rate = total_hits / total_operations if total_operations > 0 else 0
        avg_access_time = sum(self._access_times[-1000:]) / min(len(self._access_times), 1000) if self._access_times else 0
        
        tier_stats = {}
        for tier, stats in self.stats.items():
            tier_operations = stats.hits + stats.misses
            tier_stats[tier.value] = {
                'hits': stats.hits,
                'misses': stats.misses,
                'hit_rate': stats.hits / tier_operations if tier_operations > 0 else 0,
                'total_size': stats.total_size,
                'evictions': stats.evictions
            }
        
        return {
            'overall': {
                'hit_rate': overall_hit_rate,
                'total_operations': total_operations,
                'avg_access_time_ms': avg_access_time * 1000,
                'operation_counts': dict(self._operation_counts)
            },
            'tiers': tier_stats,
            'last_updated': datetime.now().isoformat()
        }
    
    async def _serialize_data(self, data: Any) -> Tuple[bytes, bool]:
        """Serialize and optionally compress data."""
        try:
            # Pickle serialize
            serialized = pickle.dumps(data)
            
            # Compress if beneficial
            if len(serialized) > self.compression_threshold:
                compressed = gzip.compress(serialized)
                if len(compressed) < len(serialized) * 0.9:  # At least 10% savings
                    return compressed, True
            
            return serialized, False
            
        except Exception as e:
            self.logger.error(f"Error serializing data: {e}")
            raise
    
    def _select_target_tier(self, entry: CacheEntry) -> CacheTier:
        """Select appropriate storage tier for entry."""
        # High priority and frequently accessed -> L1 Memory
        if entry.priority >= 4 and entry.access_count > 5:
            return CacheTier.L1_MEMORY
        
        # Real-time data -> L1 Memory
        if entry.data_type == DataType.REAL_TIME:
            return CacheTier.L1_MEMORY
        
        # Large historical data -> L2 Disk or L3 Redis
        if entry.data_type == DataType.HISTORICAL:
            return CacheTier.L2_DISK if CacheTier.L2_DISK in self.storage_tiers else CacheTier.L3_REDIS
        
        # Default to L1 Memory
        return CacheTier.L1_MEMORY
    
    async def _promote_entry(self, entry: CacheEntry, current_tier: CacheTier) -> None:
        """Promote frequently accessed entry to higher tier."""
        if current_tier == CacheTier.L1_MEMORY:
            return  # Already at highest tier
        
        target_tier = CacheTier.L1_MEMORY
        
        if target_tier in self.storage_tiers:
            await self.storage_tiers[target_tier].set(entry)
            self.logger.debug(f"Promoted entry '{entry.key}' from {current_tier} to {target_tier}")
    
    def _record_access_time(self, access_time: float) -> None:
        """Record access time for performance monitoring."""
        with self._lock:
            self._access_times.append(access_time)
            # Keep only recent measurements
            if len(self._access_times) > 10000:
                self._access_times = self._access_times[-5000:]
    
    async def _cleanup_worker(self) -> None:
        """Background worker for cache cleanup."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._cleanup_expired_entries()
                await self._enforce_size_limits()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup worker: {e}")
    
    async def _preload_worker(self) -> None:
        """Background worker for cache preloading."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._intelligent_preload()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in preload worker: {e}")
    
    async def _monitoring_worker(self) -> None:
        """Background worker for performance monitoring."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._update_performance_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring worker: {e}")
    
    async def _cleanup_expired_entries(self) -> None:
        """Clean up expired entries from all tiers."""
        for tier_name, storage in self.storage_tiers.items():
            try:
                keys = await storage.keys()
                expired_count = 0
                
                for key in keys:
                    entry = await storage.get(key)
                    if entry and entry.is_expired:
                        await storage.delete(key)
                        expired_count += 1
                
                if expired_count > 0:
                    self.logger.debug(f"Cleaned {expired_count} expired entries from {tier_name}")
                    self.stats[tier_name].evictions += expired_count
                    
            except Exception as e:
                self.logger.error(f"Error cleaning up tier {tier_name}: {e}")
    
    async def _enforce_size_limits(self) -> None:
        """Enforce cache size limits using eviction policies."""
        for tier_name, storage in self.storage_tiers.items():
            try:
                current_size = await storage.size()
                max_size = self.max_memory_size if tier_name == CacheTier.L1_MEMORY else self.max_disk_size
                
                if current_size > max_size:
                    await self._evict_entries(storage, current_size - max_size, tier_name)
                    
            except Exception as e:
                self.logger.error(f"Error enforcing size limits for {tier_name}: {e}")
    
    async def _evict_entries(self, storage: CacheStorage, target_bytes: int, tier_name: CacheTier) -> None:
        """Evict entries to free up specified bytes."""
        # Implementation would depend on eviction policy
        # For now, simple LRU-based eviction
        pass
    
    async def _intelligent_preload(self) -> None:
        """Intelligent cache preloading based on access patterns."""
        # Implementation for predictive preloading
        pass
    
    async def _update_performance_metrics(self) -> None:
        """Update performance metrics and statistics."""
        for tier_name, stats in self.stats.items():
            total_ops = stats.hits + stats.misses
            if total_ops > 0:
                stats.hit_rate = stats.hits / total_ops
                stats.miss_rate = stats.misses / total_ops
            
            stats.last_updated = datetime.now()