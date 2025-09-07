"""
Plugin Manager for Dynamic Strategy Loading System.

This module implements the PluginManager class that handles dynamic discovery,
loading, and management of arbitrage strategy plugins with enterprise-level
error handling, hot-reload capabilities, and configuration-driven management.

Key Features:
- Dynamic strategy plugin discovery and loading
- Hot-reload support for strategy updates
- Plugin priority and execution order control
- Configuration-driven plugin management
- Comprehensive error handling and logging
- Plugin health monitoring and diagnostics
"""

import asyncio
import importlib
import importlib.util
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Tuple, Set, Union
from dataclasses import dataclass, field
# Optional dependency for hot-reload functionality
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object

from ..config.models import StrategyType, StrategyConfig
from ..strategies.base import BaseStrategy, StrategyParameters, StrategyRegistry


@dataclass
class PluginInfo:
    """Information about a loaded plugin."""
    name: str
    strategy_type: StrategyType
    strategy_class: Type[BaseStrategy]
    module_path: str
    file_path: str
    last_modified: datetime
    load_time: datetime = field(default_factory=datetime.now)
    is_enabled: bool = True
    priority: int = 1
    load_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PluginLoadResult:
    """Result of plugin loading operation."""
    success: bool
    plugin_info: Optional[PluginInfo] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    load_time: float = 0.0


@dataclass
class PluginManagerConfig:
    """Configuration for the plugin manager."""
    plugin_directories: List[str] = field(default_factory=lambda: ['src/strategies'])
    auto_reload: bool = True
    reload_delay: float = 1.0  # Seconds to wait after file change
    max_load_retries: int = 3
    retry_delay: float = 0.5
    enable_hot_reload: bool = True
    validate_on_load: bool = True
    parallel_loading: bool = True
    max_load_workers: int = 4
    scan_interval: int = 30  # Seconds between plugin directory scans
    plugin_file_pattern: str = "*.py"
    exclude_patterns: List[str] = field(default_factory=lambda: ['__pycache__', '*.pyc', 'test_*', '*_test.py'])


class PluginFileWatcher(FileSystemEventHandler):
    """File system watcher for plugin hot-reload."""
    
    def __init__(self, plugin_manager: 'PluginManager'):
        """
        Initialize the file watcher.
        
        Args:
            plugin_manager: Reference to the plugin manager
        """
        super().__init__()
        self.plugin_manager = plugin_manager
        self.logger = logging.getLogger(__name__)
        self._pending_reloads: Set[str] = set()
        self._reload_timer: Optional[threading.Timer] = None
        self._reload_lock = threading.Lock()
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        file_path = event.src_path
        if not file_path.endswith('.py'):
            return
        
        # Check if it's a plugin file
        if not self.plugin_manager._is_plugin_file(file_path):
            return
        
        self.logger.info(f"Plugin file modified: {file_path}")
        
        with self._reload_lock:
            self._pending_reloads.add(file_path)
            
            # Cancel existing timer
            if self._reload_timer:
                self._reload_timer.cancel()
            
            # Schedule reload with delay to handle multiple rapid changes
            self._reload_timer = threading.Timer(
                self.plugin_manager.config.reload_delay,
                self._process_pending_reloads
            )
            self._reload_timer.start()
    
    def _process_pending_reloads(self):
        """Process all pending reload requests."""
        with self._reload_lock:
            if not self._pending_reloads:
                return
            
            files_to_reload = list(self._pending_reloads)
            self._pending_reloads.clear()
        
        # Reload plugins
        for file_path in files_to_reload:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.plugin_manager._reload_plugin_from_file(file_path),
                    self.plugin_manager._event_loop
                )
            except Exception as e:
                self.logger.error(f"Failed to reload plugin {file_path}: {e}")


class PluginManager:
    """
    Enterprise-grade plugin manager for dynamic strategy loading.
    
    Provides comprehensive plugin management capabilities including:
    - Dynamic discovery and loading of strategy plugins
    - Hot-reload support for development and updates
    - Configuration-driven plugin management
    - Plugin health monitoring and diagnostics
    - Error recovery and retry mechanisms
    - Thread-safe plugin operations
    """
    
    def __init__(self, config: Optional[PluginManagerConfig] = None):
        """
        Initialize the plugin manager.
        
        Args:
            config: Plugin manager configuration
        """
        self.config = config or PluginManagerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Plugin storage
        self._plugins: Dict[str, PluginInfo] = {}
        self._plugin_configs: Dict[str, StrategyConfig] = {}
        self._plugin_instances: Dict[str, BaseStrategy] = {}
        
        # Thread safety
        self._plugin_lock = threading.RLock()
        self._load_lock = threading.Lock()
        
        # Hot-reload support
        self._file_observer: Optional[Observer] = None
        self._file_watcher: Optional[PluginFileWatcher] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Parallel loading
        self._executor: Optional[ThreadPoolExecutor] = None
        if self.config.parallel_loading:
            self._executor = ThreadPoolExecutor(max_workers=self.config.max_load_workers)
        
        # Plugin directory paths
        self._plugin_paths: List[Path] = []
        for plugin_dir in self.config.plugin_directories:
            path = Path(plugin_dir).resolve()
            if path.exists() and path.is_dir():
                self._plugin_paths.append(path)
                self.logger.info(f"Added plugin directory: {path}")
        
        # Performance metrics
        self._load_stats = {
            'total_loads': 0,
            'successful_loads': 0,
            'failed_loads': 0,
            'reload_count': 0,
            'avg_load_time': 0.0
        }
    
    async def initialize(self) -> None:
        """
        Initialize the plugin manager and perform initial plugin discovery.
        """
        self.logger.info("Initializing PluginManager...")
        
        try:
            # Store event loop reference for hot-reload
            self._event_loop = asyncio.get_running_loop()
            
            # Perform initial plugin discovery and loading
            await self.discover_and_load_plugins()
            
            # Setup hot-reload if enabled
            if self.config.enable_hot_reload:
                await self._setup_hot_reload()
            
            self.logger.info(
                f"PluginManager initialized successfully. "
                f"Loaded {len(self._plugins)} plugins from {len(self._plugin_paths)} directories."
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PluginManager: {e}")
            raise
    
    async def discover_and_load_plugins(self) -> Dict[str, PluginLoadResult]:
        """
        Load plugins from StrategyRegistry instead of dynamic file loading.
        
        Returns:
            Dictionary mapping plugin names to load results
        """
        self.logger.info("Loading strategies from StrategyRegistry...")
        
        # Get strategies from registry instead of dynamic loading
        load_results = {}
        strategies = StrategyRegistry.get_registered_strategies()
        
        for strategy_type, strategy_class in strategies.items():
            strategy_name = strategy_class.__name__
            
            # Create plugin info for each registered strategy
            plugin_info = PluginInfo(
                name=strategy_name,
                strategy_type=strategy_type,
                strategy_class=strategy_class,
                module_path=strategy_class.__module__,
                file_path="",  # No file path for registry-loaded strategies
                last_modified=datetime.now(),
                load_time=datetime.now()
            )
            
            self._plugins[strategy_name] = plugin_info
            load_results[strategy_name] = PluginLoadResult(
                success=True,
                error_message="",
                load_time=0.0
            )
        
        successful_loads = len(strategies)
        failed_loads = 0
        
        self._load_stats['total_loads'] += successful_loads
        self._load_stats['successful_loads'] += successful_loads
        self._load_stats['failed_loads'] += failed_loads
        
        self.logger.info(
            f"Strategy loading completed: {successful_loads} strategies loaded from registry"
        )
        
        return load_results
    
    async def _discover_plugin_files(self) -> List[Tuple[str, Path]]:
        """
        Discover all Python files that could be plugins.
        
        Returns:
            List of (module_name, file_path) tuples
        """
        plugin_files = []
        
        for plugin_dir in self._plugin_paths:
            try:
                for file_path in plugin_dir.rglob(self.config.plugin_file_pattern):
                    if not file_path.is_file():
                        continue
                    
                    # Skip excluded patterns
                    if any(file_path.match(pattern) for pattern in self.config.exclude_patterns):
                        continue
                    
                    # Skip __init__.py files unless they contain strategies
                    if file_path.name == '__init__.py':
                        continue
                    
                    # Generate module name
                    relative_path = file_path.relative_to(plugin_dir)
                    module_name = str(relative_path.with_suffix('')).replace(os.sep, '.')
                    
                    plugin_files.append((module_name, file_path))
                    
            except Exception as e:
                self.logger.warning(f"Error scanning plugin directory {plugin_dir}: {e}")
        
        return plugin_files
    
    async def _load_plugins_parallel(self, plugin_files: List[Tuple[str, Path]]) -> Dict[str, PluginLoadResult]:
        """Load plugins in parallel using ThreadPoolExecutor."""
        load_results = {}
        
        # Submit all load tasks
        future_to_plugin = {}
        for module_name, file_path in plugin_files:
            future = self._executor.submit(self._load_single_plugin_sync, module_name, file_path)
            future_to_plugin[future] = (module_name, file_path)
        
        # Collect results
        for future in future_to_plugin:
            module_name, file_path = future_to_plugin[future]
            try:
                result = future.result(timeout=30)  # 30s timeout per plugin
                load_results[module_name] = result
                
                if result.success:
                    self.logger.debug(f"Successfully loaded plugin: {module_name}")
                else:
                    self.logger.warning(f"Failed to load plugin {module_name}: {result.error_message}")
                    
            except Exception as e:
                load_results[module_name] = PluginLoadResult(
                    success=False,
                    error_message=f"Parallel loading error: {e}"
                )
                self.logger.error(f"Parallel loading failed for {module_name}: {e}")
        
        return load_results
    
    async def _load_plugins_sequential(self, plugin_files: List[Tuple[str, Path]]) -> Dict[str, PluginLoadResult]:
        """Load plugins sequentially."""
        load_results = {}
        
        for module_name, file_path in plugin_files:
            result = await self._load_single_plugin(module_name, file_path)
            load_results[module_name] = result
            
            if result.success:
                self.logger.debug(f"Successfully loaded plugin: {module_name}")
            else:
                self.logger.warning(f"Failed to load plugin {module_name}: {result.error_message}")
        
        return load_results
    
    async def _load_single_plugin(self, module_name: str, file_path: Path) -> PluginLoadResult:
        """Load a single plugin asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._load_single_plugin_sync, module_name, file_path)
    
    def _load_single_plugin_sync(self, module_name: str, file_path: Path) -> PluginLoadResult:
        """
        Load a single plugin synchronously with comprehensive error handling.
        
        Args:
            module_name: Module name for the plugin
            file_path: Path to the plugin file
            
        Returns:
            PluginLoadResult with loading outcome
        """
        start_time = time.time()
        warnings = []
        
        try:
            with self._load_lock:
                # Check if already loaded
                if module_name in self._plugins:
                    return PluginLoadResult(
                        success=True,
                        plugin_info=self._plugins[module_name],
                        warnings=["Plugin already loaded"]
                    )
                
                # Load the module
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None or spec.loader is None:
                    return PluginLoadResult(
                        success=False,
                        error_message="Could not create module spec"
                    )
                
                module = importlib.util.module_from_spec(spec)
                
                # Execute the module
                try:
                    spec.loader.exec_module(module)
                except Exception as e:
                    return PluginLoadResult(
                        success=False,
                        error_message=f"Module execution failed: {e}",
                        load_time=time.time() - start_time
                    )
                
                # Find strategy classes in the module
                strategy_classes = self._find_strategy_classes(module)
                
                if not strategy_classes:
                    return PluginLoadResult(
                        success=False,
                        error_message="No strategy classes found in module",
                        load_time=time.time() - start_time
                    )
                
                # Register strategies and create plugin info
                plugins_loaded = []
                
                for strategy_class in strategy_classes:
                    try:
                        strategy_type = strategy_class.strategy_type
                        if hasattr(strategy_class, 'strategy_type'):
                            # Get strategy type from instance
                            temp_instance = strategy_class()
                            strategy_type = temp_instance.strategy_type
                        else:
                            warnings.append(f"Strategy class {strategy_class.__name__} missing strategy_type property")
                            continue
                        
                        # Create plugin info
                        plugin_info = PluginInfo(
                            name=strategy_class.__name__,
                            strategy_type=strategy_type,
                            strategy_class=strategy_class,
                            module_path=module_name,
                            file_path=str(file_path),
                            last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
                            load_time=datetime.now(),
                            load_count=1,
                            metadata={
                                'module_name': module_name,
                                'file_size': file_path.stat().st_size,
                                'docstring': strategy_class.__doc__ or ''
                            }
                        )
                        
                        # Validate plugin if required
                        if self.config.validate_on_load:
                            validation_warnings = self._validate_plugin(plugin_info)
                            warnings.extend(validation_warnings)
                        
                        # Register with strategy registry
                        StrategyRegistry.register(strategy_type)(strategy_class)
                        
                        # Store plugin info
                        with self._plugin_lock:
                            self._plugins[strategy_class.__name__] = plugin_info
                        
                        plugins_loaded.append(plugin_info)
                        
                        self.logger.info(f"Loaded strategy plugin: {strategy_class.__name__} ({strategy_type})")
                        
                    except Exception as e:
                        warnings.append(f"Failed to register strategy {strategy_class.__name__}: {e}")
                        continue
                
                if not plugins_loaded:
                    return PluginLoadResult(
                        success=False,
                        error_message="No strategies could be registered from module",
                        warnings=warnings,
                        load_time=time.time() - start_time
                    )
                
                # Return info for the first loaded plugin (main result)
                return PluginLoadResult(
                    success=True,
                    plugin_info=plugins_loaded[0],
                    warnings=warnings,
                    load_time=time.time() - start_time
                )
                
        except Exception as e:
            return PluginLoadResult(
                success=False,
                error_message=f"Plugin loading error: {e}",
                warnings=warnings,
                load_time=time.time() - start_time
            )
    
    def _find_strategy_classes(self, module) -> List[Type[BaseStrategy]]:
        """
        Find all strategy classes in a module.
        
        Args:
            module: Python module to search
            
        Returns:
            List of strategy class types
        """
        strategy_classes = []
        
        for name in dir(module):
            obj = getattr(module, name)
            
            # Check if it's a class and inherits from BaseStrategy
            if (isinstance(obj, type) and 
                issubclass(obj, BaseStrategy) and 
                obj != BaseStrategy):
                
                strategy_classes.append(obj)
        
        return strategy_classes
    
    def _validate_plugin(self, plugin_info: PluginInfo) -> List[str]:
        """
        Validate a plugin and return any warnings.
        
        Args:
            plugin_info: Plugin information to validate
            
        Returns:
            List of validation warnings
        """
        warnings = []
        
        try:
            # Create a test instance
            strategy_class = plugin_info.strategy_class
            test_instance = strategy_class()
            
            # Check required methods exist and are callable
            required_methods = ['scan_opportunities', 'calculate_profit', 'assess_risk']
            for method_name in required_methods:
                if not hasattr(test_instance, method_name):
                    warnings.append(f"Missing required method: {method_name}")
                elif not callable(getattr(test_instance, method_name)):
                    warnings.append(f"Method {method_name} is not callable")
            
            # Check strategy_type property
            if not hasattr(test_instance, 'strategy_type'):
                warnings.append("Missing strategy_type property")
            
            # Validate strategy type
            try:
                strategy_type = test_instance.strategy_type
                if not isinstance(strategy_type, StrategyType):
                    warnings.append(f"Invalid strategy_type: {strategy_type}")
            except Exception as e:
                warnings.append(f"Error accessing strategy_type: {e}")
            
        except Exception as e:
            warnings.append(f"Plugin validation failed: {e}")
        
        return warnings
    
    async def _setup_hot_reload(self) -> None:
        """Setup file system monitoring for hot-reload."""
        if not WATCHDOG_AVAILABLE:
            self.logger.debug("Watchdog not available - hot-reload disabled")
            return
            
        if not self._plugin_paths:
            self.logger.warning("No plugin directories to monitor for hot-reload")
            return
        
        try:
            self._file_watcher = PluginFileWatcher(self)
            self._file_observer = Observer()
            
            # Watch all plugin directories
            for plugin_dir in self._plugin_paths:
                self._file_observer.schedule(
                    self._file_watcher,
                    str(plugin_dir),
                    recursive=True
                )
                self.logger.info(f"Monitoring plugin directory for changes: {plugin_dir}")
            
            self._file_observer.start()
            self.logger.info("Hot-reload monitoring started")
            
        except Exception as e:
            self.logger.error(f"Failed to setup hot-reload: {e}")
    
    def _is_plugin_file(self, file_path: str) -> bool:
        """Check if a file path corresponds to a plugin file."""
        file_path_obj = Path(file_path)
        
        # Check if it's in any plugin directory
        for plugin_dir in self._plugin_paths:
            try:
                file_path_obj.relative_to(plugin_dir)
                return True
            except ValueError:
                continue
        
        return False
    
    async def _reload_plugin_from_file(self, file_path: str) -> bool:
        """
        Reload a plugin from its file path.
        
        Args:
            file_path: Path to the plugin file
            
        Returns:
            True if reload was successful
        """
        try:
            # Find the plugin that corresponds to this file
            plugin_to_reload = None
            for plugin_info in self._plugins.values():
                if plugin_info.file_path == file_path:
                    plugin_to_reload = plugin_info
                    break
            
            if not plugin_to_reload:
                self.logger.warning(f"Could not find plugin for file: {file_path}")
                return False
            
            # Reload the plugin
            result = await self._reload_plugin(plugin_to_reload.name)
            
            if result:
                self.logger.info(f"Successfully reloaded plugin: {plugin_to_reload.name}")
            else:
                self.logger.error(f"Failed to reload plugin: {plugin_to_reload.name}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error reloading plugin from file {file_path}: {e}")
            return False
    
    async def reload_plugin(self, plugin_name: str) -> bool:
        """
        Reload a specific plugin by name.
        
        Args:
            plugin_name: Name of the plugin to reload
            
        Returns:
            True if reload was successful
        """
        return await self._reload_plugin(plugin_name)
    
    async def _reload_plugin(self, plugin_name: str) -> bool:
        """
        Internal method to reload a plugin.
        
        Args:
            plugin_name: Name of the plugin to reload
            
        Returns:
            True if reload was successful
        """
        try:
            with self._plugin_lock:
                if plugin_name not in self._plugins:
                    self.logger.warning(f"Plugin {plugin_name} not found for reload")
                    return False
                
                plugin_info = self._plugins[plugin_name]
                
                # Clear from strategy registry first
                if plugin_info.strategy_type in StrategyRegistry._strategies:
                    del StrategyRegistry._strategies[plugin_info.strategy_type]
                
                # Remove from plugin instances
                if plugin_name in self._plugin_instances:
                    del self._plugin_instances[plugin_name]
                
                # Reload the module
                module_path = plugin_info.module_path
                file_path = Path(plugin_info.file_path)
                
                # Remove from sys.modules if present
                if module_path in sys.modules:
                    del sys.modules[module_path]
                
                # Load the plugin again
                result = await self._load_single_plugin(module_path, file_path)
                
                if result.success:
                    # Update statistics
                    self._load_stats['reload_count'] += 1
                    
                    self.logger.info(f"Plugin {plugin_name} reloaded successfully")
                    return True
                else:
                    # Reload failed, restore error info
                    plugin_info.error_count += 1
                    plugin_info.last_error = result.error_message
                    
                    self.logger.error(f"Plugin {plugin_name} reload failed: {result.error_message}")
                    return False
        
        except Exception as e:
            self.logger.error(f"Error reloading plugin {plugin_name}: {e}")
            return False
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """
        Get information about a specific plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Plugin information or None if not found
        """
        with self._plugin_lock:
            return self._plugins.get(plugin_name)
    
    def list_plugins(self) -> Dict[str, PluginInfo]:
        """
        List all loaded plugins.
        
        Returns:
            Dictionary mapping plugin names to plugin information
        """
        with self._plugin_lock:
            return self._plugins.copy()
    
    def get_strategies(self, enabled_only: bool = True) -> Dict[str, BaseStrategy]:
        """
        Get strategy instances managed by this plugin manager.
        
        Args:
            enabled_only: Only return enabled strategies
            
        Returns:
            Dictionary mapping strategy names to instances
        """
        strategies = {}
        
        with self._plugin_lock:
            for plugin_name, plugin_info in self._plugins.items():
                if enabled_only and not plugin_info.is_enabled:
                    continue
                
                # Create instance if not cached
                if plugin_name not in self._plugin_instances:
                    try:
                        # Get parameters from config if available
                        config = self._plugin_configs.get(plugin_name)
                        if config:
                            params = StrategyParameters(
                                min_profit_threshold=config.min_profit_threshold,
                                max_risk_tolerance=config.max_risk_tolerance,
                                **config.parameters
                            )
                        else:
                            params = StrategyParameters()
                        
                        # Create strategy instance
                        strategy_instance = plugin_info.strategy_class(params)
                        self._plugin_instances[plugin_name] = strategy_instance
                        
                    except Exception as e:
                        self.logger.error(f"Failed to create instance for plugin {plugin_name}: {e}")
                        continue
                
                strategies[plugin_name] = self._plugin_instances[plugin_name]
        
        return strategies
    
    def update_plugin_config(self, plugin_name: str, config: StrategyConfig) -> bool:
        """
        Update configuration for a specific plugin.
        
        Args:
            plugin_name: Name of the plugin
            config: New strategy configuration
            
        Returns:
            True if update was successful
        """
        try:
            with self._plugin_lock:
                if plugin_name not in self._plugins:
                    self.logger.warning(f"Plugin {plugin_name} not found for config update")
                    return False
                
                # Store the configuration
                self._plugin_configs[plugin_name] = config
                
                # Update plugin info
                plugin_info = self._plugins[plugin_name]
                plugin_info.is_enabled = config.enabled
                plugin_info.priority = config.priority
                
                # Clear cached instance to force recreation with new config
                if plugin_name in self._plugin_instances:
                    del self._plugin_instances[plugin_name]
                
                self.logger.info(f"Updated configuration for plugin: {plugin_name}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update config for plugin {plugin_name}: {e}")
            return False
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """
        Get plugin loading statistics.
        
        Returns:
            Dictionary with loading statistics
        """
        with self._plugin_lock:
            stats = self._load_stats.copy()
            stats.update({
                'total_plugins': len(self._plugins),
                'enabled_plugins': sum(1 for p in self._plugins.values() if p.is_enabled),
                'disabled_plugins': sum(1 for p in self._plugins.values() if not p.is_enabled),
                'plugins_with_errors': sum(1 for p in self._plugins.values() if p.error_count > 0)
            })
            return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of the plugin system.
        
        Returns:
            Dictionary with health status information
        """
        health_info = {
            'status': 'healthy',
            'plugins': {},
            'statistics': self.get_load_statistics(),
            'hot_reload_active': self._file_observer is not None and self._file_observer.is_alive() if self._file_observer else False,
            'plugin_directories': [str(p) for p in self._plugin_paths]
        }
        
        # Check individual plugins
        with self._plugin_lock:
            for plugin_name, plugin_info in self._plugins.items():
                plugin_health = {
                    'enabled': plugin_info.is_enabled,
                    'load_count': plugin_info.load_count,
                    'error_count': plugin_info.error_count,
                    'last_error': plugin_info.last_error,
                    'last_modified': plugin_info.last_modified.isoformat(),
                    'load_time': plugin_info.load_time.isoformat()
                }
                
                # Try to create instance to verify health
                try:
                    if plugin_name not in self._plugin_instances:
                        test_instance = plugin_info.strategy_class()
                        plugin_health['instance_creation'] = 'success'
                    else:
                        plugin_health['instance_creation'] = 'cached'
                except Exception as e:
                    plugin_health['instance_creation'] = f'failed: {e}'
                    health_info['status'] = 'degraded'
                
                health_info['plugins'][plugin_name] = plugin_health
        
        return health_info
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the plugin manager."""
        self.logger.info("Shutting down PluginManager...")
        
        # Stop file observer
        if self._file_observer:
            self._file_observer.stop()
            self._file_observer.join()
            self._file_observer = None
        
        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        
        # Clear plugin instances
        with self._plugin_lock:
            self._plugin_instances.clear()
            self._plugins.clear()
            self._plugin_configs.clear()
        
        self.logger.info("PluginManager shutdown complete")
    
    def __str__(self) -> str:
        return f"PluginManager(plugins={len(self._plugins)}, directories={len(self._plugin_paths)})"
    
    def __repr__(self) -> str:
        return (f"PluginManager(plugins={list(self._plugins.keys())}, "
                f"directories={[str(p) for p in self._plugin_paths]}, "
                f"hot_reload={self.config.enable_hot_reload})")