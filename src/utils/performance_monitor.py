#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance monitoring utilities for the arbitrage scanning system.

Provides comprehensive performance tracking, metrics collection, and
monitoring capabilities for all system components.
"""

import time
import psutil
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemResourceMetrics:
    """System resource utilization metrics."""
    timestamp: datetime
    memory_usage_mb: float
    memory_percent: float
    cpu_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str, monitor: 'PerformanceMonitor'):
        self.operation_name = operation_name
        self.monitor = monitor
        self.start_time = None
        self.metrics = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.metrics = PerformanceMetrics(
            operation_name=self.operation_name,
            start_time=self.start_time
        )
        return self.metrics
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        self.metrics.end_time = end_time
        self.metrics.duration = end_time - self.start_time
        
        # Capture resource usage
        try:
            process = psutil.Process()
            self.metrics.memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            self.metrics.cpu_usage = process.cpu_percent()
        except:
            pass
        
        if exc_type is not None:
            self.metrics.success = False
            self.metrics.error_message = str(exc_val)
        
        self.monitor._record_metrics(self.metrics)


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    
    Tracks operation performance, resource usage, and system health
    with thread-safe operations and configurable retention.
    """
    
    def __init__(self, max_history: int = 1000, 
                 enable_resource_monitoring: bool = True,
                 resource_monitor_interval: float = 1.0):
        """
        Initialize performance monitor.
        
        Args:
            max_history: Maximum number of metrics to retain
            enable_resource_monitoring: Enable system resource monitoring
            resource_monitor_interval: Interval between resource measurements (seconds)
        """
        self.max_history = max_history
        self.enable_resource_monitoring = enable_resource_monitoring
        self.resource_monitor_interval = resource_monitor_interval
        
        # Thread-safe storage
        self._lock = threading.Lock()
        self._metrics_history: deque = deque(maxlen=max_history)
        self._operation_stats = defaultdict(list)
        self._resource_history: deque = deque(maxlen=max_history)
        
        # Performance counters
        self._total_operations = 0
        self._failed_operations = 0
        self._start_time = time.time()
        
        # Resource monitoring thread
        self._resource_monitor_thread = None
        self._stop_resource_monitoring = False
        
        if enable_resource_monitoring:
            self._start_resource_monitoring()
        
        self.logger = logging.getLogger(f"{__name__}.PerformanceMonitor")
        self.logger.info("Performance monitor initialized")
    
    def time_operation(self, operation_name: str) -> PerformanceTimer:
        """
        Create a timer context manager for an operation.
        
        Args:
            operation_name: Name of the operation being timed
            
        Returns:
            PerformanceTimer context manager
        """
        return PerformanceTimer(operation_name, self)
    
    def _record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics (thread-safe)."""
        with self._lock:
            self._metrics_history.append(metrics)
            self._operation_stats[metrics.operation_name].append(metrics)
            
            self._total_operations += 1
            if not metrics.success:
                self._failed_operations += 1
    
    def _start_resource_monitoring(self):
        """Start background resource monitoring thread."""
        def monitor_resources():
            while not self._stop_resource_monitoring:
                try:
                    # Collect system metrics
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    
                    # Disk I/O
                    disk_io_before = psutil.disk_io_counters()
                    time.sleep(0.1)  # Brief pause for I/O measurement
                    disk_io_after = psutil.disk_io_counters()
                    
                    # Network I/O
                    net_io_before = psutil.net_io_counters()
                    time.sleep(0.1)  # Brief pause for I/O measurement
                    net_io_after = psutil.net_io_counters()
                    
                    metrics = SystemResourceMetrics(
                        timestamp=datetime.now(),
                        memory_usage_mb=memory_info.rss / 1024 / 1024,
                        memory_percent=process.memory_percent(),
                        cpu_percent=process.cpu_percent(),
                        disk_io_read_mb=(disk_io_after.read_bytes - disk_io_before.read_bytes) / 1024 / 1024,
                        disk_io_write_mb=(disk_io_after.write_bytes - disk_io_before.write_bytes) / 1024 / 1024,
                        network_io_sent_mb=(net_io_after.bytes_sent - net_io_before.bytes_sent) / 1024 / 1024,
                        network_io_recv_mb=(net_io_after.bytes_recv - net_io_before.bytes_recv) / 1024 / 1024
                    )
                    
                    with self._lock:
                        self._resource_history.append(metrics)
                    
                except Exception as e:
                    self.logger.warning(f"Resource monitoring error: {e}")
                
                # Wait for next measurement
                time.sleep(self.resource_monitor_interval)
        
        self._resource_monitor_thread = threading.Thread(
            target=monitor_resources, daemon=True
        )
        self._resource_monitor_thread.start()
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Dictionary containing operation statistics
        """
        with self._lock:
            metrics_list = self._operation_stats.get(operation_name, [])
            
            if not metrics_list:
                return {
                    "operation_name": operation_name,
                    "total_executions": 0,
                    "success_rate": 0.0,
                    "avg_duration": 0.0,
                    "min_duration": 0.0,
                    "max_duration": 0.0,
                    "avg_memory_usage": 0.0
                }
            
            successful_metrics = [m for m in metrics_list if m.success]
            durations = [m.duration for m in metrics_list if m.duration is not None]
            memory_usages = [m.memory_usage for m in metrics_list if m.memory_usage is not None]
            
            return {
                "operation_name": operation_name,
                "total_executions": len(metrics_list),
                "successful_executions": len(successful_metrics),
                "failed_executions": len(metrics_list) - len(successful_metrics),
                "success_rate": len(successful_metrics) / len(metrics_list) if metrics_list else 0,
                "avg_duration": sum(durations) / len(durations) if durations else 0,
                "min_duration": min(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
                "total_duration": sum(durations) if durations else 0,
                "avg_memory_usage": sum(memory_usages) / len(memory_usages) if memory_usages else 0,
                "peak_memory_usage": max(memory_usages) if memory_usages else 0
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system performance statistics."""
        with self._lock:
            uptime = time.time() - self._start_time
            
            # Operation statistics
            operations_per_second = self._total_operations / uptime if uptime > 0 else 0
            error_rate = self._failed_operations / self._total_operations if self._total_operations > 0 else 0
            
            # Resource statistics
            current_resource = None
            avg_memory = 0
            peak_memory = 0
            avg_cpu = 0
            peak_cpu = 0
            
            if self._resource_history:
                current_resource = self._resource_history[-1]
                memory_values = [r.memory_usage_mb for r in self._resource_history]
                cpu_values = [r.cpu_percent for r in self._resource_history]
                
                avg_memory = sum(memory_values) / len(memory_values)
                peak_memory = max(memory_values)
                avg_cpu = sum(cpu_values) / len(cpu_values)
                peak_cpu = max(cpu_values)
            
            return {
                "uptime_seconds": uptime,
                "total_operations": self._total_operations,
                "failed_operations": self._failed_operations,
                "operations_per_second": operations_per_second,
                "error_rate": error_rate,
                "current_memory_mb": current_resource.memory_usage_mb if current_resource else 0,
                "average_memory_mb": avg_memory,
                "peak_memory_mb": peak_memory,
                "current_cpu_percent": current_resource.cpu_percent if current_resource else 0,
                "average_cpu_percent": avg_cpu,
                "peak_cpu_percent": peak_cpu,
                "metrics_history_size": len(self._metrics_history),
                "resource_history_size": len(self._resource_history)
            }
    
    def get_recent_performance(self, operation_name: Optional[str] = None,
                             minutes: int = 5) -> List[PerformanceMetrics]:
        """
        Get recent performance metrics.
        
        Args:
            operation_name: Filter by operation name (optional)
            minutes: Number of minutes of history to return
            
        Returns:
            List of recent performance metrics
        """
        cutoff_time = time.time() - (minutes * 60)
        
        with self._lock:
            recent_metrics = [
                m for m in self._metrics_history 
                if m.start_time >= cutoff_time
            ]
            
            if operation_name:
                recent_metrics = [
                    m for m in recent_metrics 
                    if m.operation_name == operation_name
                ]
            
            return recent_metrics
    
    def get_resource_usage_trend(self, minutes: int = 5) -> List[SystemResourceMetrics]:
        """
        Get resource usage trend over time.
        
        Args:
            minutes: Number of minutes of history to return
            
        Returns:
            List of resource metrics
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self._lock:
            return [
                r for r in self._resource_history
                if r.timestamp >= cutoff_time
            ]
    
    def detect_performance_issues(self) -> List[Dict[str, Any]]:
        """
        Detect potential performance issues.
        
        Returns:
            List of detected issues with descriptions and recommendations
        """
        issues = []
        
        # Check system resources
        system_stats = self.get_system_stats()
        
        if system_stats["peak_memory_mb"] > 1024:  # 1GB
            issues.append({
                "type": "high_memory_usage",
                "severity": "warning",
                "description": f"Peak memory usage: {system_stats['peak_memory_mb']:.1f} MB",
                "recommendation": "Monitor memory usage and optimize data structures"
            })
        
        if system_stats["average_cpu_percent"] > 80:
            issues.append({
                "type": "high_cpu_usage",
                "severity": "warning", 
                "description": f"Average CPU usage: {system_stats['average_cpu_percent']:.1f}%",
                "recommendation": "Consider optimizing CPU-intensive operations"
            })
        
        if system_stats["error_rate"] > 0.05:  # 5% error rate
            issues.append({
                "type": "high_error_rate",
                "severity": "critical",
                "description": f"Error rate: {system_stats['error_rate']:.2%}",
                "recommendation": "Investigate and fix failing operations"
            })
        
        # Check slow operations
        with self._lock:
            for operation_name in self._operation_stats:
                stats = self.get_operation_stats(operation_name)
                
                if stats["avg_duration"] > 5.0:  # 5 seconds
                    issues.append({
                        "type": "slow_operation",
                        "severity": "warning",
                        "description": f"{operation_name} averages {stats['avg_duration']:.2f}s",
                        "recommendation": "Optimize this operation for better performance"
                    })
        
        return issues
    
    def export_metrics(self, filepath: str, format: str = "json"):
        """
        Export performance metrics to file.
        
        Args:
            filepath: Path to output file
            format: Export format ("json" or "csv")
        """
        with self._lock:
            if format.lower() == "json":
                data = {
                    "system_stats": self.get_system_stats(),
                    "operation_stats": {
                        op: self.get_operation_stats(op)
                        for op in self._operation_stats.keys()
                    },
                    "recent_metrics": [
                        {
                            "operation_name": m.operation_name,
                            "start_time": m.start_time,
                            "duration": m.duration,
                            "memory_usage": m.memory_usage,
                            "cpu_usage": m.cpu_usage,
                            "success": m.success,
                            "error_message": m.error_message
                        }
                        for m in list(self._metrics_history)
                    ]
                }
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            
            elif format.lower() == "csv":
                import pandas as pd
                
                # Convert metrics to DataFrame
                metrics_data = []
                for m in self._metrics_history:
                    metrics_data.append({
                        "operation_name": m.operation_name,
                        "start_time": m.start_time,
                        "duration": m.duration,
                        "memory_usage": m.memory_usage,
                        "cpu_usage": m.cpu_usage,
                        "success": m.success,
                        "error_message": m.error_message
                    })
                
                df = pd.DataFrame(metrics_data)
                df.to_csv(filepath, index=False)
    
    def reset_metrics(self):
        """Reset all collected metrics."""
        with self._lock:
            self._metrics_history.clear()
            self._operation_stats.clear()
            self._resource_history.clear()
            self._total_operations = 0
            self._failed_operations = 0
            self._start_time = time.time()
        
        self.logger.info("Performance metrics reset")
    
    def shutdown(self):
        """Shutdown the performance monitor."""
        if self._resource_monitor_thread:
            self._stop_resource_monitoring = True
            self._resource_monitor_thread.join(timeout=5)
        
        self.logger.info("Performance monitor shutdown")
    
    def __del__(self):
        """Cleanup resources."""
        self.shutdown()


# Example usage and testing functions
def example_usage():
    """Example of how to use the performance monitor."""
    monitor = PerformanceMonitor()
    
    # Time a simple operation
    with monitor.time_operation("test_calculation"):
        time.sleep(0.1)  # Simulate work
        result = sum(range(10000))
    
    # Time a potentially failing operation
    with monitor.time_operation("risky_operation") as timer:
        timer.metadata["input_size"] = 1000
        try:
            # Simulate work that might fail
            if False:  # Change to True to simulate failure
                raise ValueError("Simulated error")
            time.sleep(0.05)
        except Exception as e:
            timer.error_message = str(e)
            raise
    
    # Get statistics
    print("System Stats:", monitor.get_system_stats())
    print("Operation Stats:", monitor.get_operation_stats("test_calculation"))
    
    # Check for issues
    issues = monitor.detect_performance_issues()
    for issue in issues:
        print(f"Issue: {issue}")
    
    monitor.shutdown()


if __name__ == "__main__":
    example_usage()