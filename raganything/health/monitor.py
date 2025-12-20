# Copyright (c) 2025 Kirky.X
# All rights reserved.

import asyncio
from typing import Dict, List

from raganything.logger import get_i18n_logger

from .core import ComponentStatus, HealthCheck, HealthCheckResult, Notifier
from raganything.i18n import _


class ConsoleNotifier(Notifier):
    """Simple notifier that logs to console/logger."""

    def __init__(self):
        self.logger = get_i18n_logger()

    async def notify(self, result: HealthCheckResult) -> None:
        if result.status == ComponentStatus.UNHEALTHY:
            self.logger.error(
                _("🚨 HEALTH CHECK FAILED [{}]: {}").format(result.component_name, result.message)
            )
            if result.error:
                self.logger.error(_("   Details: {}").format(result.error))
        elif result.status == ComponentStatus.WARNING:
            self.logger.warning(
                _("⚠️ HEALTH CHECK WARNING [{}]: {}").format(result.component_name, result.message)
            )
        elif result.status == ComponentStatus.HEALTHY:
            self.logger.info(
                _("✅ Health Check Passed [{}]: {}").format(result.component_name, result.message)
            )
        else:
            self.logger.info(
                _("❓ Health Check Unknown [{}]: {}").format(result.component_name, result.message)
            )


class HealthMonitor:
    """Orchestrates health checks and notifications."""

    def __init__(self):
        self.logger = get_i18n_logger()
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
                    error=e,
                )

            self._results[check.name] = result
            results[check.name] = result

            for notifier in self._notifiers:
                await notifier.notify(result)

        return results

    async def start_monitoring(self, interval_seconds: int = 60):
        """Run checks periodically in an infinite loop."""
        self.logger.info(_("Starting health monitoring (interval: {}s)").format(interval_seconds))
        while True:
            try:
                await self.run_checks()
            except Exception as e:
                self.logger.error(_("Error during health monitoring cycle: {}").format(e))

            await asyncio.sleep(interval_seconds)
