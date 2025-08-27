#!/usr/bin/env python3
"""Launch Symbiote in production mode with real implementations.

This script:
1. Checks all dependencies
2. Reports what features are available
3. Launches with appropriate configuration
4. Falls back gracefully for missing dependencies
"""

import sys
import asyncio
import argparse
from pathlib import Path
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from daemon.production_integration import (
    ProductionIntegration,
    DependencyChecker,
    run_production_mode
)
from loguru import logger


def check_environment():
    """Check environment and dependencies."""
    print("=" * 60)
    print("SYMBIOTE PRODUCTION LAUNCHER")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("ERROR: Python 3.8+ required")
        sys.exit(1)
    
    # Check dependencies
    print("\nChecking dependencies...")
    deps = DependencyChecker.check_all()
    
    all_available = True
    for name, status in deps.items():
        if status.available:
            print(f"  ✓ {name}: {status.version or 'available'}")
        else:
            print(f"  ✗ {name}: not available")
            all_available = False
    
    print("\n" + "=" * 60)
    
    if all_available:
        print("✅ ALL DEPENDENCIES AVAILABLE - Full production mode")
    else:
        print("⚠️  SOME DEPENDENCIES MISSING - Degraded mode")
        print("\nTo enable all features, install:")
        print("  pip install -r requirements_production.txt")
        print("  python -m spacy download en_core_web_sm")
    
    print("=" * 60)
    
    return deps


def install_missing_dependencies():
    """Attempt to install missing dependencies."""
    print("\nAttempting to install missing dependencies...")
    
    import subprocess
    
    try:
        # Install production requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", "requirements_production.txt"
        ])
        
        # Download spacy model
        subprocess.check_call([
            sys.executable, "-m", "spacy", 
            "download", "en_core_web_sm"
        ])
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Launch Symbiote in production mode"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("symbiote_production.yaml"),
        help="Path to configuration file"
    )
    parser.add_argument(
        "--vault",
        type=Path,
        help="Override vault path"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Attempt to install missing dependencies"
    )
    parser.add_argument(
        "--force-degraded",
        action="store_true",
        help="Force degraded mode even if all dependencies available"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    # Check environment
    deps = check_environment()
    
    # Install dependencies if requested
    if args.install_deps:
        if install_missing_dependencies():
            print("\nPlease restart the application")
            sys.exit(0)
    
    # Check if should continue
    all_available = all(d.available for d in deps.values())
    
    if not all_available and not args.force_degraded:
        print("\n⚠️  Not all dependencies available.")
        print("Options:")
        print("  1. Run with --install-deps to install missing dependencies")
        print("  2. Run with --force-degraded to continue anyway")
        print("  3. Install manually: pip install -r requirements_production.txt")
        
        response = input("\nContinue in degraded mode? [y/N]: ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(0)
    
    # Set environment
    os.environ['SYMBIOTE_MODE'] = 'production'
    
    if args.vault:
        os.environ['SYMBIOTE_VAULT'] = str(args.vault)
    
    # Launch production mode
    print("\n🚀 Launching Symbiote in production mode...")
    print(f"   Config: {args.config}")
    print(f"   Mode: {'FULL' if all_available else 'DEGRADED'}")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        await run_production_mode(args.config)
    except KeyboardInterrupt:
        print("\n\n👋 Symbiote shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())