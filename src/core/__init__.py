"""
Core components of the arbitrage trading system.

This package contains the core functionality including the plugin management system,
configuration managers, and utility functions.
"""

from .plugin_manager import PluginManager, PluginManagerConfig, PluginInfo

__all__ = [
    'PluginManager',
    'PluginManagerConfig', 
    'PluginInfo'
]