import asyncio
from typing import List, Dict
from .core import HealthCheck, Notifier, HealthCheckResult, ComponentStatus
from raganything.logger import logger

class ConsoleNotifier(Notifier):
    """Simple notifier that logs to console/logger."""
    
    async def notify(self, result: HealthCheckResult) -> None:
        if result.status == ComponentStatus.UNHEALTHY:
            logger.error(f"🚨 HEALTH CHECK FAILED [{result.component_name}]: {result.message}")
            if result.error:
                logger.error(f"   Details: {result.error}")
        elif result.status == ComponentStatus.WARNING:
            logger.warning(f"⚠️ HEALTH CHECK WARNING [{result.component_name}]: {result.message}")
        elif result.status == ComponentStatus.HEALTHY:
            logger.info(f"✅ Health Check Passed [{result.component_name}]: {result.message}")
        else:
            logger.info(f"❓ Health Check Unknown [{result.component_name}]: {result.message}")


class HealthMonitor:
    """Orchestrates health checks and notifications."""
    
    def __init__(self):
        self._checks: List[HealthCheck] = []
        self._notifiers: List[Notifier] = []
        self._results: Dict[str, HealthCheckResult] = {}

    def add_check(self, check: HealthCheck):
        self._checks.append(check)

    def add_notifier(self, notifier: Notifier):
        self._notifiers.append(notifier)

    async def run_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        for check in self._checks:
            try:
                result = await check.check()
            except Exception as e:
                # Should be caught by BaseHealthCheck, but safety net
                from .core import ComponentStatus
                result = HealthCheckResult(
                    component_name=check.name,
                    status=ComponentStatus.UNHEALTHY,
                    message=f"Monitor failed to run check: {e}",
                    error=e
                )
            
            self._results[check.name] = result
            results[check.name] = result
            
            for notifier in self._notifiers:
                await notifier.notify(result)
                
        return results

    async def start_monitoring(self, interval_seconds: int = 60):
        """Run checks periodically in an infinite loop."""
        logger.info(f"Starting health monitoring (interval: {interval_seconds}s)")
        while True:
            try:
                await self.run_checks()
            except Exception as e:
                logger.error(f"Error during health monitoring cycle: {e}")
            
            await asyncio.sleep(interval_seconds)
