"""Main daemon process for Symbiote."""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import psutil
import aiohttp
from aiohttp import web
from loguru import logger

from .config import Config
from .bus import get_event_bus
from .capture import CaptureService
from .search import SearchOrchestrator
from .indexers import AnalyticsIndexer
from .api import create_api_app


class SymbioteDaemon:
    """Main daemon coordinating all services."""
    
    def __init__(self, config: Config):
        self.config = config
        self.start_time = datetime.utcnow()
        
        # Core services
        self.event_bus = get_event_bus()
        self.capture_service = CaptureService(config)
        self.search_orchestrator = SearchOrchestrator(config)
        self.analytics_indexer = AnalyticsIndexer(config)
        
        # Statistics
        self.stats = {
            "capture_count": 0,
            "search_count": 0,
            "suggestion_count": 0
        }
        
        # HTTP API
        self.api_app = None
        self.api_runner = None
        self.api_site = None
    
    async def start(self) -> None:
        """Start all daemon services."""
        logger.info("Starting Symbiote daemon...")
        
        # Start event bus
        await self.event_bus.start()
        
        # Initialize services
        await self.capture_service.initialize()
        await self.analytics_indexer.initialize()
        
        # Subscribe to events for stats
        self.event_bus.subscribe("capture.written", self._on_capture)
        self.event_bus.subscribe("search.completed", self._on_search)
        
        # Start HTTP API
        await self._start_api()
        
        logger.info("Symbiote daemon started successfully")
    
    async def stop(self) -> None:
        """Stop all daemon services."""
        logger.info("Stopping Symbiote daemon...")
        
        # Stop API
        if self.api_site:
            await self.api_site.stop()
        if self.api_runner:
            await self.api_runner.cleanup()
        
        # Stop services
        await self.capture_service.close()
        await self.analytics_indexer.close()
        await self.event_bus.stop()
        
        logger.info("Symbiote daemon stopped")
    
    async def _start_api(self) -> None:
        """Start the HTTP API server."""
        self.api_app = create_api_app(self)
        self.api_runner = web.AppRunner(self.api_app)
        await self.api_runner.setup()
        
        self.api_site = web.TCPSite(
            self.api_runner,
            'localhost',
            8765
        )
        await self.api_site.start()
        
        logger.info("API server started on http://localhost:8765")
    
    async def _on_capture(self, event) -> None:
        """Handle capture events for stats."""
        self.stats["capture_count"] += 1
    
    async def _on_search(self, event) -> None:
        """Handle search events for stats."""
        self.stats["search_count"] += 1
    
    def get_status(self) -> dict:
        """Get daemon status and statistics."""
        process = psutil.Process()
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "status": "running",
            "version": "0.1.0",
            "uptime": f"{uptime:.0f}s",
            "stats": {
                "capture_count": self.stats["capture_count"],
                "search_count": self.stats["search_count"],
                "suggestion_count": self.stats["suggestion_count"],
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent()
            },
            "config": {
                "vault_path": str(self.config.vault_path),
                "indices": {
                    "fts": self.config.indices.fts,
                    "vector": self.config.indices.vector,
                    "analytics": self.config.indices.analytics
                }
            }
        }


async def main(config_path: Optional[str] = None):
    """Main entry point for the daemon."""
    # Setup logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Also log to file
    log_dir = Path.home() / ".local" / "share" / "symbiote" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "daemon.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )
    
    # Load configuration
    try:
        if config_path:
            config = Config.load(Path(config_path))
        else:
            config = Config.load()
    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    
    # Create and start daemon
    daemon = SymbioteDaemon(config)
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        asyncio.create_task(daemon.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await daemon.start()
        
        # Keep running until stopped
        while True:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.exception(f"Daemon error: {e}")
    finally:
        await daemon.stop()


if __name__ == "__main__":
    asyncio.run(main())