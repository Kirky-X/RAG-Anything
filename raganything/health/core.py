# Copyright (c) 2025 Kirky.X
# All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import List, Optional, Protocol


class ComponentStatus(Enum):
    """Status of a component."""

    HEALTHY = auto()
    UNHEALTHY = auto()
    UNKNOWN = auto()
    WARNING = auto()


@dataclass(frozen=True)
class HealthCheckResult:
    """Result of a health check execution."""

    component_name: str
    status: ComponentStatus
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""
    error: Optional[Exception] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        return self.status == ComponentStatus.HEALTHY


class HealthCheck(Protocol):
    """Protocol for component health checks."""

    @property
    def name(self) -> str:
        """Name of the component being checked."""
        ...

    async def check(self) -> HealthCheckResult:
        """Execute the health check."""
        ...


class Notifier(Protocol):
    """Protocol for notification strategies."""

    async def notify(self, result: HealthCheckResult) -> None:
        """Send notification about a health check result."""
        ...


class BaseHealthCheck(ABC):
    """Base implementation for health checks to reduce boilerplate."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    async def check_health(self) -> HealthCheckResult:
        """Implement actual health check logic here."""
        pass

    async def check(self) -> HealthCheckResult:
        """Wrapper to handle exceptions safely."""
        try:
            return await self.check_health()
        except Exception as e:
            return HealthCheckResult(
                component_name=self.name,
                status=ComponentStatus.UNHEALTHY,
                message=f"Health check failed unexpectedly: {str(e)}",
                error=e,
            )
