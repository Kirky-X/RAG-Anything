# Copyright (c) 2025 Kirky.X
# All rights reserved.

from typing import List, Optional

from raganything.logger import logger

from .core import ComponentStatus, HealthCheckResult, Notifier


class ConsoleNotifier(Notifier):
    """Simple notifier that logs to console/logger."""

    async def notify(self, result: HealthCheckResult) -> None:
        if result.status == ComponentStatus.UNHEALTHY:
            logger.error(
                f"🚨 HEALTH CHECK FAILED [{result.component_name}]: {result.message}"
            )
            if result.error:
                logger.error(f"   Details: {result.error}")
        elif result.status == ComponentStatus.WARNING:
            logger.warning(
                f"⚠️ HEALTH CHECK WARNING [{result.component_name}]: {result.message}"
            )
        elif result.status == ComponentStatus.HEALTHY:
            logger.info(
                f"✅ Health Check Passed [{result.component_name}]: {result.message}"
            )
        else:
            logger.info(
                f"❓ Health Check Unknown [{result.component_name}]: {result.message}"
            )


class EmailNotifier(Notifier):
    """Notifier that sends emails via SMTP."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        sender: str,
        recipients: List[str],
        password: Optional[str] = None,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.sender = sender
        self.recipients = recipients
        self.password = password

    async def notify(self, result: HealthCheckResult) -> None:
        # Only notify on failure/warning to avoid spam
        if result.status not in [ComponentStatus.UNHEALTHY, ComponentStatus.WARNING]:
            return

        subject = f"[{result.status.name}] Health Check Alert: {result.component_name}"
        body = f"""
        Component: {result.component_name}
        Status: {result.status.name}
        Message: {result.message}
        Timestamp: {result.timestamp}
        Metadata: {result.metadata}
        """

        # Placeholder for actual SMTP logic (requires aiosmtplib)
        logger.info(
            f"📧 [EmailNotifier] Would send email to {self.recipients}: {subject}"
        )
