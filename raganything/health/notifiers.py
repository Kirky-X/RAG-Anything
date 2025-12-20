# Copyright (c) 2025 Kirky.X
# All rights reserved.

from typing import List, Optional

from raganything.i18n_logger import get_i18n_logger

from .core import ComponentStatus, HealthCheckResult, Notifier
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
        self.logger = get_i18n_logger()
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
        self.logger.info(
            f"📧 [EmailNotifier] Would send email to {self.recipients}: {subject}"
        )
