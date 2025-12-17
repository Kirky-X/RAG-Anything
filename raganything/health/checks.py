# Copyright (c) 2025 Kirky.X
# All rights reserved.

import tomllib
import httpx
import os
import psutil
import shutil
from typing import Optional
from .core import BaseHealthCheck, HealthCheckResult, ComponentStatus

class OllamaHealthCheck(BaseHealthCheck):
    """Checks the health of the Ollama service."""
    
    def __init__(self, config_path: str = "config.toml"):
        super().__init__("Ollama")
        self.config_path = config_path

    async def check_health(self) -> HealthCheckResult:
        # 1. Load config
        if not os.path.exists(self.config_path):
             return HealthCheckResult(
                component_name=self.name,
                status=ComponentStatus.UNKNOWN,
                message=f"Config file not found: {self.config_path}"
            )
        
        try:
            with open(self.config_path, "rb") as f:
                config = tomllib.load(f)
        except Exception as e:
             return HealthCheckResult(
                component_name=self.name,
                status=ComponentStatus.UNKNOWN,
                message=f"Failed to parse config: {e}",
                error=e
            )

        # 2. Extract settings
        vision_config = config.get("raganything", {}).get("vision", {})
        if vision_config.get("provider") != "ollama":
             return HealthCheckResult(
                component_name=self.name,
                status=ComponentStatus.UNKNOWN, 
                message="Ollama not configured as vision provider",
                metadata={"provider": vision_config.get("provider")}
            )

        api_base = vision_config.get("api_base")
        model = vision_config.get("model")
        
        if not api_base:
             return HealthCheckResult(
                component_name=self.name,
                status=ComponentStatus.UNHEALTHY,
                message="Ollama api_base not configured"
            )

        # 3. Check connectivity
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Some Ollama versions return 200 OK on root
                try:
                    resp = await client.get(api_base)
                except httpx.ConnectError as e:
                     return HealthCheckResult(
                        component_name=self.name,
                        status=ComponentStatus.UNHEALTHY,
                        message=f"Connection refused to {api_base}",
                        error=e,
                        metadata={"url": api_base}
                    )
                
                if resp.status_code != 200:
                     return HealthCheckResult(
                        component_name=self.name,
                        status=ComponentStatus.UNHEALTHY,
                        message=f"Ollama returned status {resp.status_code}",
                        metadata={"url": api_base, "response": resp.text[:100]}
                    )
                
                # Check if model exists via /api/tags
                tags_url = f"{api_base.rstrip('/')}/api/tags"
                try:
                    tags_resp = await client.get(tags_url)
                except Exception:
                    # Fallback if /api/tags fails but root worked
                    return HealthCheckResult(
                        component_name=self.name,
                        status=ComponentStatus.WARNING,
                        message="Ollama running but failed to list models",
                        metadata={"url": api_base}
                    )
                
                model_found = False
                available_models = []
                if tags_resp.status_code == 200:
                    data = tags_resp.json()
                    models = data.get("models", [])
                    available_models = [m.get("name") for m in models]
                    # Simple check: is 'model' substring of any available model name?
                    model_found = any(model in m_name for m_name in available_models)
                
                status = ComponentStatus.HEALTHY
                msg = "Ollama is running"
                if not model_found:
                    status = ComponentStatus.WARNING
                    msg = f"Ollama is running but model '{model}' not found"

                return HealthCheckResult(
                    component_name=self.name,
                    status=status,
                    message=msg,
                    metadata={
                        "url": api_base, 
                        "model_configured": model,
                        "available_models": available_models[:10]
                    }
                )

        except httpx.RequestError as e:
            return HealthCheckResult(
                component_name=self.name,
                status=ComponentStatus.UNHEALTHY,
                message=f"Connection failed: {str(e)}",
                error=e,
                metadata={"url": api_base}
            )

class SystemResourceCheck(BaseHealthCheck):
    """Checks system memory and disk usage."""
    
    def __init__(self, disk_path: str = "."):
        super().__init__("SystemResources")
        self.disk_path = disk_path

    async def check_health(self) -> HealthCheckResult:
        mem = psutil.virtual_memory()
        disk = shutil.disk_usage(self.disk_path)
        
        mem_usage = mem.percent
        disk_usage = (disk.used / disk.total) * 100
        
        status = ComponentStatus.HEALTHY
        msg_parts = []
        
        if mem_usage > 90:
            status = ComponentStatus.WARNING
            msg_parts.append(f"High Memory Usage: {mem_usage}%")
        
        if disk_usage > 90:
            status = ComponentStatus.WARNING
            msg_parts.append(f"High Disk Usage: {disk_usage:.1f}%")
            
        if not msg_parts:
            msg_parts.append("Resources OK")

        return HealthCheckResult(
            component_name=self.name,
            status=status,
            message=", ".join(msg_parts),
            metadata={
                "memory_percent": mem_usage,
                "disk_percent": disk_usage,
                "disk_free_gb": disk.free / (1024**3)
            }
        )
