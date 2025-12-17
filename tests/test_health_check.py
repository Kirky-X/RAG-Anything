import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from raganything.health import (
    HealthMonitor, 
    OllamaHealthCheck, 
    SystemResourceCheck, 
    ConsoleNotifier,
    ComponentStatus
)

# Configure logging
logging.basicConfig(level=logging.INFO)

async def main():
    print("Initializing Health Monitor...")
    monitor = HealthMonitor()
    
    # Add checks
    monitor.add_check(OllamaHealthCheck("config.toml"))
    monitor.add_check(SystemResourceCheck())
    
    # Add notifier
    monitor.add_notifier(ConsoleNotifier())
    
    print("Running checks...")
    results = await monitor.run_checks()
    
    print("\nResults Summary:")
    all_healthy = True
    for name, result in results.items():
        print(f"{name}: {result.status.name} - {result.message}")
        if result.status == ComponentStatus.UNHEALTHY:
            all_healthy = False
            
    if all_healthy:
        print("\n✅ All systems healthy!")
        sys.exit(0)
    else:
        print("\n❌ Some systems unhealthy!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
