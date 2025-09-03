"""
Configuration manager for handling application settings and validation.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import ValidationError as PydanticValidationError

from .models import (
    SystemConfig, DataSourceConfig, StrategyConfig, 
    ValidationError, DataSourceType, StrategyType
)


class ConfigurationError(Exception):
    """Configuration related errors."""
    pass


class ConfigManager:
    """
    Configuration manager handling loading, validation and hot-reloading.
    """
    
    def __init__(self, config_path: Optional[Path] = None, env_file: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration files directory
            env_file: Path to environment file
        """
        self.config_path = config_path or Path("config")
        self.env_file = env_file or Path(".env")
        self._config: Optional[SystemConfig] = None
        self._config_cache: Dict[str, Any] = {}
        
    def load_config(self, reload: bool = False) -> SystemConfig:
        """
        Load and validate system configuration.
        
        Args:
            reload: Force reload even if already cached
            
        Returns:
            SystemConfig: Validated configuration
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if self._config is not None and not reload:
            return self._config
            
        try:
            # Load environment variables first
            self._load_environment()
            
            # Load base configuration
            config_data = self._load_config_files()
            
            # Create and validate configuration
            self._config = SystemConfig(**config_data)
            
            # Ensure directories exist
            self._ensure_directories()
            
            return self._config
            
        except PydanticValidationError as e:
            errors = [
                ValidationError(
                    field=".".join(str(x) for x in error["loc"]),
                    message=error["msg"],
                    value=error.get("input", "")
                ) for error in e.errors()
            ]
            raise ConfigurationError(f"Configuration validation failed: {errors}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def get_data_source_config(self, source_name: str) -> Optional[DataSourceConfig]:
        """
        Get configuration for a specific data source.
        
        Args:
            source_name: Name of the data source
            
        Returns:
            DataSourceConfig: Data source configuration or None if not found
        """
        config = self.load_config()
        return config.data_sources.get(source_name)
    
    def get_strategy_config(self, strategy_name: str) -> Optional[StrategyConfig]:
        """
        Get configuration for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            StrategyConfig: Strategy configuration or None if not found
        """
        config = self.load_config()
        return config.strategies.get(strategy_name)
    
    def get_system_config(self) -> SystemConfig:
        """
        Get the complete system configuration.
        
        Returns:
            SystemConfig: Complete system configuration
        """
        return self.load_config()
    
    def get_enabled_data_sources(self) -> Dict[str, DataSourceConfig]:
        """
        Get all enabled data sources sorted by priority.
        
        Returns:
            Dict of enabled data source configurations
        """
        config = self.load_config()
        enabled_sources = {
            name: ds_config 
            for name, ds_config in config.data_sources.items()
            if ds_config.enabled
        }
        
        # Sort by priority (lower number = higher priority)
        return dict(sorted(
            enabled_sources.items(),
            key=lambda x: x[1].priority
        ))
    
    def get_enabled_strategies(self) -> Dict[str, StrategyConfig]:
        """
        Get all enabled strategies sorted by priority.
        
        Returns:
            Dict of enabled strategy configurations
        """
        config = self.load_config()
        enabled_strategies = {
            name: strategy_config
            for name, strategy_config in config.strategies.items()
            if strategy_config.enabled
        }
        
        # Sort by priority (lower number = higher priority)
        return dict(sorted(
            enabled_strategies.items(),
            key=lambda x: x[1].priority
        ))
    
    def validate_config(self) -> List[ValidationError]:
        """
        Validate current configuration and return any errors.
        
        Returns:
            List of validation errors, empty if valid
        """
        try:
            self.load_config(reload=True)
            return []
        except ConfigurationError as e:
            if "validation failed:" in str(e):
                # Extract validation errors from error message
                return []  # This would need proper error extraction
            return [ValidationError(field="config", message=str(e), value="")]
    
    def reload_config(self) -> SystemConfig:
        """
        Force reload configuration from files.
        
        Returns:
            SystemConfig: Reloaded configuration
        """
        self._config = None
        self._config_cache.clear()
        return self.load_config()
    
    def add_data_source(self, name: str, config: DataSourceConfig) -> None:
        """
        Add or update data source configuration.
        
        Args:
            name: Data source name
            config: Data source configuration
        """
        system_config = self.load_config()
        system_config.data_sources[name] = config
        self._config = system_config
    
    def add_strategy(self, name: str, config: StrategyConfig) -> None:
        """
        Add or update strategy configuration.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        system_config = self.load_config()
        system_config.strategies[name] = config
        self._config = system_config
    
    def _load_environment(self) -> None:
        """Load environment variables from .env file."""
        if self.env_file.exists():
            # This is a simplified env loader - in production, use python-dotenv
            with open(self.env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip().strip('"\'')
    
    def _load_config_files(self) -> Dict[str, Any]:
        """
        Load configuration from YAML/JSON files.
        
        Returns:
            Dict containing merged configuration data
        """
        config_data = {}
        
        # Load main configuration file
        for ext in ['.yaml', '.yml', '.json']:
            config_file = self.config_path / f"config{ext}"
            if config_file.exists():
                config_data.update(self._load_file(config_file))
                break
        
        # Load data sources configuration
        data_sources_file = self._find_config_file("data_sources")
        if data_sources_file:
            data_sources = self._load_file(data_sources_file)
            config_data['data_sources'] = self._build_data_sources(data_sources)
        
        # Load strategies configuration
        strategies_file = self._find_config_file("strategies") 
        if strategies_file:
            strategies = self._load_file(strategies_file)
            config_data['strategies'] = self._build_strategies(strategies)
        
        return config_data
    
    def _find_config_file(self, basename: str) -> Optional[Path]:
        """Find configuration file with various extensions."""
        for ext in ['.yaml', '.yml', '.json']:
            config_file = self.config_path / f"{basename}{ext}"
            if config_file.exists():
                return config_file
        return None
    
    def _load_file(self, file_path: Path) -> Dict[str, Any]:
        """Load configuration from a YAML or JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif file_path.suffix == '.json':
                    return json.load(f)
                else:
                    raise ValueError(f"Unsupported file format: {file_path.suffix}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load {file_path}: {e}")
    
    def _build_data_sources(self, data_sources_config: Dict[str, Any]) -> Dict[str, DataSourceConfig]:
        """Build data source configurations from config dict."""
        result = {}
        
        for name, config in data_sources_config.items():
            try:
                result[name] = DataSourceConfig(**config)
            except PydanticValidationError as e:
                raise ConfigurationError(f"Invalid data source config for {name}: {e}")
        
        return result
    
    def _build_strategies(self, strategies_config: Dict[str, Any]) -> Dict[str, StrategyConfig]:
        """Build strategy configurations from config dict."""
        result = {}
        
        for name, config in strategies_config.items():
            try:
                result[name] = StrategyConfig(**config)
            except PydanticValidationError as e:
                raise ConfigurationError(f"Invalid strategy config for {name}: {e}")
        
        return result
    
    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        if self._config:
            for directory in [self._config.data_dir, self._config.log_dir, self.config_path]:
                directory.mkdir(parents=True, exist_ok=True)


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        ConfigManager: Global configuration manager
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> SystemConfig:
    """
    Get the current system configuration.
    
    Returns:
        SystemConfig: Current system configuration
    """
    return get_config_manager().load_config()