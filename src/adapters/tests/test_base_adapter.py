"""
Basic tests for base adapter functionality.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.adapters.base import (
    DataRequest, DataResponse, BaseDataAdapter, ConnectionStatus
)


class TestDataRequest:
    """Test DataRequest data model."""

    def test_data_request_creation(self):
        """Test DataRequest can be created with basic parameters."""
        request = DataRequest(
            request_type="options_data",
            parameters={"symbol": "AAPL", "date": "2024-01-15"}
        )
        
        assert request.request_type == "options_data"
        assert request.parameters["symbol"] == "AAPL"
        assert isinstance(request.timestamp, datetime)

    def test_data_request_with_optional_fields(self):
        """Test DataRequest with all optional fields."""
        request = DataRequest(
            request_type="market_data",
            parameters={"symbols": ["AAPL", "GOOGL"]},
            priority=5,
            timeout=30
        )
        
        assert request.priority == 5
        assert request.timeout == 30


class TestDataResponse:
    """Test DataResponse data model."""

    def test_successful_data_response(self):
        """Test successful DataResponse creation."""
        test_data = {"price": 150.0, "volume": 1000}
        
        response = DataResponse(
            success=True,
            data=test_data,
            timestamp=datetime.now()
        )
        
        assert response.success is True
        assert response.data == test_data
        assert response.error_message is None

    def test_failed_data_response(self):
        """Test failed DataResponse creation."""
        response = DataResponse(
            success=False,
            data=None,
            error_message="Connection timeout",
            timestamp=datetime.now()
        )
        
        assert response.success is False
        assert response.data is None
        assert response.error_message == "Connection timeout"


class TestConnectionStatus:
    """Test ConnectionStatus enum."""

    def test_connection_status_values(self):
        """Test ConnectionStatus enum has expected values."""
        assert hasattr(ConnectionStatus, 'CONNECTED')
        assert hasattr(ConnectionStatus, 'DISCONNECTED')
        assert hasattr(ConnectionStatus, 'CONNECTING')
        
        # Test values are unique
        statuses = [ConnectionStatus.CONNECTED, ConnectionStatus.DISCONNECTED, ConnectionStatus.CONNECTING]
        assert len(set(statuses)) == len(statuses)


class TestDataQuality:
    """Test basic adapter functionality."""

    def test_connection_status_usage(self):
        """Test ConnectionStatus is properly defined and usable."""
        assert ConnectionStatus.CONNECTED != ConnectionStatus.DISCONNECTED
        assert isinstance(ConnectionStatus.CONNECTED, str)


class ConcreteDataAdapter(BaseDataAdapter):
    """Concrete implementation of BaseDataAdapter for testing."""
    
    def __init__(self):
        super().__init__()
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.last_request = None
    
    async def connect(self):
        """Mock connect implementation."""
        self.connection_status = ConnectionStatus.CONNECTED
    
    async def disconnect(self):
        """Mock disconnect implementation."""
        self.connection_status = ConnectionStatus.DISCONNECTED
    
    async def fetch_data(self, request: DataRequest) -> DataResponse:
        """Mock fetch_data implementation."""
        self.last_request = request
        
        # Mock successful response
        return DataResponse(
            success=True,
            data={"mock": "data", "request_type": request.request_type},
            timestamp=datetime.now()
        )
    
    def get_status(self):
        """Mock get_status implementation."""
        return "ACTIVE" if self.connection_status == ConnectionStatus.CONNECTED else "INACTIVE"
    
    def health_check(self):
        """Mock health_check implementation."""
        return {
            "status": "healthy" if self.connection_status == ConnectionStatus.CONNECTED else "unhealthy",
            "connection": self.connection_status.value
        }


class TestBaseDataAdapter:
    """Test BaseDataAdapter abstract functionality."""

    def setup_method(self):
        """Set up test fixtures.""" 
        self.adapter = ConcreteDataAdapter()

    @pytest.mark.asyncio
    async def test_adapter_connection_lifecycle(self):
        """Test adapter connection and disconnection."""
        # Initially disconnected
        assert self.adapter.connection_status == ConnectionStatus.DISCONNECTED
        assert self.adapter.get_status() == "INACTIVE"
        
        # Connect
        await self.adapter.connect()
        assert self.adapter.connection_status == ConnectionStatus.CONNECTED
        assert self.adapter.get_status() == "ACTIVE"
        
        # Disconnect
        await self.adapter.disconnect()
        assert self.adapter.connection_status == ConnectionStatus.DISCONNECTED
        assert self.adapter.get_status() == "INACTIVE"

    @pytest.mark.asyncio
    async def test_fetch_data_functionality(self):
        """Test data fetching functionality."""
        await self.adapter.connect()
        
        request = DataRequest(
            request_type="test_data",
            parameters={"test": "param"}
        )
        
        response = await self.adapter.fetch_data(request)
        
        assert isinstance(response, DataResponse)
        assert response.success is True
        assert response.data is not None
        assert "mock" in response.data
        assert response.data["request_type"] == "test_data"
        
        # Verify request was stored
        assert self.adapter.last_request == request

    def test_health_check(self):
        """Test health check functionality."""
        # When disconnected
        health = self.adapter.health_check()
        assert health["status"] == "unhealthy"
        
        # When connected (simulate)
        self.adapter.connection_status = ConnectionStatus.CONNECTED
        health = self.adapter.health_check()
        assert health["status"] == "healthy"

    def test_adapter_initialization(self):
        """Test adapter initializes properly."""
        new_adapter = ConcreteDataAdapter()
        
        assert new_adapter.connection_status == ConnectionStatus.DISCONNECTED
        assert new_adapter.last_request is None
        assert callable(new_adapter.connect)
        assert callable(new_adapter.disconnect)
        assert callable(new_adapter.fetch_data)


if __name__ == "__main__":
    pytest.main([__file__])