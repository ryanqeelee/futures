"""
插件模块 - 可扩展插件系统
Plugins Module - Extensible Plugin System

支持动态加载的插件架构，用于扩展策略和数据源功能
"""

from .plugin_manager import PluginManager
from .plugin_interface import IPlugin

__all__ = ["PluginManager", "IPlugin"]