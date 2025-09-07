"""
Comprehensive unit tests for unified exception handling framework.

Tests cover:
- Exception hierarchy and inheritance
- Error context creation and handling
- Decorator functionality
- Exception chaining and preservation
- Error handling patterns
"""

import pytest
import asyncio
import logging
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.core.exceptions import (
    TradingSystemError, DataSourceError, ArbitrageError, PricingError,
    RiskError, SystemError, ConfigurationError, ErrorContext,
    error_handler, async_error_handler, create_error_context,
    handle_data_source_error
)


class TestExceptionHierarchy:
    """Test the exception class hierarchy and basic functionality."""

    def test_trading_system_error_base(self):
        """Test base TradingSystemError functionality."""
        context = create_error_context("test_component", "test_operation")
        error = TradingSystemError("Test error", "TEST_ERROR", context)
        
        assert "Test error" in str(error)
        assert "TEST_ERROR" in str(error)
        assert error.context.component == "test_component"
        assert error.context.operation == "test_operation"

    def test_data_source_error(self):
        """Test DataSourceError specific functionality."""
        context = create_error_context("tushare_adapter", "fetch_data")
        error = DataSourceError("Connection failed", context)
        
        assert isinstance(error, TradingSystemError)
        assert error.context.component == "tushare_adapter"
        assert "DATA_ERROR" in str(error)

    def test_arbitrage_error(self):
        """Test ArbitrageError specific functionality."""
        context = create_error_context("pricing_strategy", "calculate_opportunity")
        error = ArbitrageError("Arbitrage calculation failed", context)
        
        assert isinstance(error, TradingSystemError)
        assert error.context.component == "pricing_strategy"

    def test_pricing_error(self):
        """Test PricingError specific functionality."""
        context = create_error_context("bs_engine", "calculate_option_price")
        error = PricingError("Pricing model failed", context)
        
        assert isinstance(error, ArbitrageError)

    def test_risk_error(self):
        """Test RiskError specific functionality."""
        context = create_error_context("risk_manager", "calculate_var")
        error = RiskError("Risk calculation failed", context)
        
        assert isinstance(error, TradingSystemError)

    def test_system_error(self):
        """Test SystemError specific functionality."""
        context = create_error_context("cache_manager", "read_cache")
        error = SystemError("System operation failed", context)
        
        assert isinstance(error, TradingSystemError)

    def test_configuration_error(self):
        """Test ConfigurationError specific functionality."""
        context = create_error_context("config_manager", "load_config")
        error = ConfigurationError("Invalid configuration", context)
        
        assert isinstance(error, SystemError)


class TestErrorContext:
    """Test error context creation and handling."""

    def test_create_error_context_basic(self):
        """Test basic error context creation."""
        context = create_error_context("test_component", "test_operation")
        
        assert isinstance(context, ErrorContext)
        assert context.component == "test_component"
        assert context.operation == "test_operation"
        assert isinstance(context.timestamp, float)

    def test_create_error_context_with_kwargs(self):
        """Test error context creation with keyword arguments."""
        context = create_error_context(
            "test_component", 
            "test_operation", 
            additional_info={"symbol": "TSLA", "price": 150.0}
        )
        
        assert context.component == "test_component"
        assert context.operation == "test_operation"
        assert context.additional_info["symbol"] == "TSLA"
        assert context.additional_info["price"] == 150.0

    def test_create_error_context_with_request_id(self):
        """Test error context creation with request ID."""
        context = create_error_context(
            "test_component", 
            "test_operation",
            request_id="req-123"
        )
        
        assert context.request_id == "req-123"


class TestErrorHandlerDecorator:
    """Test the error_handler decorator functionality."""

    def test_error_handler_success(self):
        """Test error_handler decorator with successful function execution."""
        logger = MagicMock()
        
        @error_handler(logger)
        def successful_function(x, y):
            return x + y
        
        result = successful_function(2, 3)
        assert result == 5

    def test_error_handler_exception_caught(self):
        """Test error_handler decorator catches and logs exceptions."""
        logger = MagicMock()
        
        @error_handler(logger)
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()
        
        # Verify logger was called
        logger.error.assert_called()

    def test_error_handler_logging_format(self):
        """Test error_handler logs with proper format."""
        logger = MagicMock()
        
        @error_handler(logger)
        def failing_function():
            raise ValueError("Test error message")
        
        with pytest.raises(ValueError):
            failing_function()
        
        # Check that logger.error was called with the error message
        logger.error.assert_called()
        call_args = logger.error.call_args[0][0]
        assert "Test error message" in call_args


class TestAsyncErrorHandlerDecorator:
    """Test the async_error_handler decorator functionality."""

    @pytest.mark.asyncio
    async def test_async_error_handler_success(self):
        """Test async_error_handler decorator with successful async function."""
        logger = MagicMock()
        
        @async_error_handler(logger)
        async def successful_async_function(x, y):
            await asyncio.sleep(0.001)  # Small delay to test async
            return x + y
        
        result = await successful_async_function(2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_async_error_handler_exception_caught(self):
        """Test async_error_handler decorator catches and logs exceptions."""
        logger = MagicMock()
        
        @async_error_handler(logger)
        async def failing_async_function():
            await asyncio.sleep(0.001)
            raise ValueError("Async error")
        
        with pytest.raises(ValueError):
            await failing_async_function()
        
        # Verify logger was called
        logger.error.assert_called()


class TestHandleDataSourceError:
    """Test the handle_data_source_error utility function."""

    def test_handle_data_source_error_basic(self):
        """Test basic data source error handling."""
        original_error = ConnectionError("Connection timeout")
        
        data_error = handle_data_source_error("tushare", "fetch_options_data", original_error)
        
        assert isinstance(data_error, DataSourceError)
        assert "Connection timeout" in str(data_error)
        assert data_error.context.component == "data_source_tushare"
        assert data_error.context.operation == "fetch_options_data"

    def test_handle_data_source_error_preserves_original(self):
        """Test data source error handling preserves original error info."""
        original_error = ValueError("Invalid API key")
        
        data_error = handle_data_source_error("wind", "authenticate", original_error)
        
        assert isinstance(data_error, DataSourceError)
        assert "Invalid API key" in str(data_error)
        assert data_error.context.component == "data_source_wind"


class TestErrorHandlingPatterns:
    """Test common error handling patterns and integration."""

    def test_exception_chaining(self):
        """Test that exception information is preserved in logs."""
        logger = MagicMock()
        
        @error_handler(logger)
        def level_2():
            try:
                raise ValueError("Level 1 error")
            except ValueError as e:
                # Re-raise with additional context
                raise RuntimeError("Level 2 error") from e
        
        with pytest.raises(RuntimeError):
            level_2()
        
        # Verify logging occurred
        logger.error.assert_called()

    def test_error_context_creation_performance(self):
        """Test that error context creation is efficient."""
        start_time = time.time()
        
        # Create many error contexts
        for i in range(1000):
            context = create_error_context(f"component_{i}", f"operation_{i}")
            assert isinstance(context, ErrorContext)
        
        end_time = time.time()
        
        # Should complete in reasonable time (less than 1 second)
        assert end_time - start_time < 1.0

    def test_error_hierarchy_inheritance(self):
        """Test that error inheritance works correctly."""
        context = create_error_context("test", "test")
        
        # Test that specific errors are instances of their parent classes
        data_error = DataSourceError("test", context)
        assert isinstance(data_error, TradingSystemError)
        
        arbitrage_error = ArbitrageError("test", context)  
        assert isinstance(arbitrage_error, TradingSystemError)
        
        pricing_error = PricingError("test", context)
        assert isinstance(pricing_error, ArbitrageError)
        assert isinstance(pricing_error, TradingSystemError)

    def test_error_codes_are_set(self):
        """Test that error codes are properly set for different exception types."""
        context = create_error_context("test", "test")
        
        data_error = DataSourceError("test", context)
        assert "DATA_ERROR" in str(data_error)
        
        system_error = SystemError("test", context)
        assert "SYSTEM_ERROR" in str(system_error)


if __name__ == "__main__":
    pytest.main([__file__])