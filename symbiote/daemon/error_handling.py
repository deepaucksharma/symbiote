"""Comprehensive error handling and recovery system.

This module implements robust error handling:
- Circuit breakers for failing services
- Exponential backoff with jitter
- Graceful degradation strategies
- Error recovery and self-healing
- Detailed error tracking and reporting
"""

import asyncio
import time
import random
from typing import Optional, Callable, Any, Dict, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import traceback

from loguru import logger


class ServiceState(Enum):
    """Service health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CIRCUIT_OPEN = "circuit_open"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ErrorEvent:
    """Represents an error event."""
    timestamp: datetime
    service: str
    error_type: str
    message: str
    severity: ErrorSeverity
    traceback: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'service': self.service,
            'error_type': self.error_type,
            'message': self.message,
            'severity': self.severity.value,
            'context': self.context
        }


@dataclass
class ServiceHealth:
    """Tracks health of a service."""
    name: str
    state: ServiceState = ServiceState.HEALTHY
    error_count: int = 0
    success_count: int = 0
    last_error: Optional[ErrorEvent] = None
    last_success: Optional[datetime] = None
    consecutive_failures: int = 0
    circuit_opened_at: Optional[datetime] = None
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        total = self.error_count + self.success_count
        if total == 0:
            return 0.0
        return self.error_count / total
    
    @property
    def is_available(self) -> bool:
        """Check if service is available."""
        return self.state not in [ServiceState.CIRCUIT_OPEN, ServiceState.UNHEALTHY]


class CircuitBreaker:
    """Circuit breaker for service protection."""
    
    def __init__(self,
                 name: str,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        """
        Initialize circuit breaker.
        
        Args:
            name: Service name
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type to catch
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.health = ServiceHealth(name=name)
        self.recovery_attempts = 0
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        # Check if circuit is open
        if self.health.state == ServiceState.CIRCUIT_OPEN:
            if self._should_attempt_recovery():
                logger.info(f"Circuit breaker {self.name}: Attempting recovery")
                self.recovery_attempts += 1
            else:
                raise Exception(f"Circuit breaker {self.name} is open")
        
        try:
            # Call function
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Record success
            self._record_success()
            
            return result
            
        except self.expected_exception as e:
            # Record failure
            self._record_failure(e)
            
            # Check if circuit should open
            if self.health.consecutive_failures >= self.failure_threshold:
                self._open_circuit()
            
            raise
    
    def _record_success(self):
        """Record successful call."""
        self.health.success_count += 1
        self.health.consecutive_failures = 0
        self.health.last_success = datetime.now()
        
        # Reset circuit if was open
        if self.health.state == ServiceState.CIRCUIT_OPEN:
            logger.info(f"Circuit breaker {self.name}: Circuit closed after recovery")
            self.health.state = ServiceState.HEALTHY
            self.recovery_attempts = 0
        elif self.health.state == ServiceState.DEGRADED:
            # Check if should return to healthy
            if self.health.error_rate < 0.1:
                self.health.state = ServiceState.HEALTHY
    
    def _record_failure(self, error: Exception):
        """Record failed call."""
        self.health.error_count += 1
        self.health.consecutive_failures += 1
        
        self.health.last_error = ErrorEvent(
            timestamp=datetime.now(),
            service=self.name,
            error_type=type(error).__name__,
            message=str(error),
            severity=ErrorSeverity.HIGH if self.health.consecutive_failures > 3 else ErrorSeverity.MEDIUM,
            traceback=traceback.format_exc()
        )
        
        # Update state
        if self.health.error_rate > 0.5:
            self.health.state = ServiceState.UNHEALTHY
        elif self.health.error_rate > 0.2:
            self.health.state = ServiceState.DEGRADED
    
    def _open_circuit(self):
        """Open the circuit."""
        logger.warning(f"Circuit breaker {self.name}: Opening circuit after {self.health.consecutive_failures} failures")
        self.health.state = ServiceState.CIRCUIT_OPEN
        self.health.circuit_opened_at = datetime.now()
    
    def _should_attempt_recovery(self) -> bool:
        """Check if should attempt recovery."""
        if not self.health.circuit_opened_at:
            return True
        
        elapsed = (datetime.now() - self.health.circuit_opened_at).total_seconds()
        
        # Use exponential backoff for recovery attempts
        backoff = self.recovery_timeout * (2 ** min(self.recovery_attempts, 5))
        
        return elapsed >= backoff
    
    def reset(self):
        """Reset circuit breaker."""
        self.health = ServiceHealth(name=self.name)
        self.recovery_attempts = 0


class RetryPolicy:
    """Retry policy with exponential backoff."""
    
    def __init__(self,
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        """
        Initialize retry policy.
        
        Args:
            max_retries: Maximum retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add jitter
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for attempt.
        
        Args:
            attempt: Attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        # Exponential backoff
        delay = self.base_delay * (self.exponential_base ** attempt)
        
        # Cap at max delay
        delay = min(delay, self.max_delay)
        
        # Add jitter
        if self.jitter:
            delay = delay * (0.5 + random.random())
        
        return delay
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry policy.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self.calculate_delay(attempt)
                    logger.debug(f"Retry {attempt + 1}/{self.max_retries} after {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All retries failed: {e}")
        
        raise last_exception


class ErrorRecovery:
    """Error recovery strategies."""
    
    def __init__(self):
        """Initialize error recovery."""
        self.recovery_strategies = {
            'file_not_found': self._recover_file_not_found,
            'permission_denied': self._recover_permission_denied,
            'connection_error': self._recover_connection_error,
            'timeout_error': self._recover_timeout_error,
            'memory_error': self._recover_memory_error,
            'disk_full': self._recover_disk_full
        }
    
    async def attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """
        Attempt to recover from error.
        
        Args:
            error: The error to recover from
            context: Error context
            
        Returns:
            True if recovery successful
        """
        error_type = self._classify_error(error)
        
        if error_type in self.recovery_strategies:
            strategy = self.recovery_strategies[error_type]
            return await strategy(error, context)
        
        return False
    
    def _classify_error(self, error: Exception) -> str:
        """Classify error type."""
        error_str = str(error).lower()
        
        if 'file not found' in error_str or isinstance(error, FileNotFoundError):
            return 'file_not_found'
        elif 'permission denied' in error_str or isinstance(error, PermissionError):
            return 'permission_denied'
        elif 'connection' in error_str or isinstance(error, ConnectionError):
            return 'connection_error'
        elif 'timeout' in error_str or isinstance(error, TimeoutError):
            return 'timeout_error'
        elif isinstance(error, MemoryError):
            return 'memory_error'
        elif 'no space left' in error_str:
            return 'disk_full'
        
        return 'unknown'
    
    async def _recover_file_not_found(self, error: Exception, context: Dict) -> bool:
        """Recover from file not found error."""
        file_path = context.get('file_path')
        
        if file_path:
            try:
                # Create parent directories
                from pathlib import Path
                path = Path(file_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create empty file if appropriate
                if context.get('create_if_missing'):
                    path.touch()
                    logger.info(f"Created missing file: {file_path}")
                    return True
                    
            except Exception as e:
                logger.error(f"Failed to create file {file_path}: {e}")
        
        return False
    
    async def _recover_permission_denied(self, error: Exception, context: Dict) -> bool:
        """Recover from permission denied error."""
        file_path = context.get('file_path')
        
        if file_path:
            try:
                # Try to fix permissions
                import os
                import stat
                
                os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP)
                logger.info(f"Fixed permissions for: {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to fix permissions for {file_path}: {e}")
        
        return False
    
    async def _recover_connection_error(self, error: Exception, context: Dict) -> bool:
        """Recover from connection error."""
        # Wait and retry
        await asyncio.sleep(5)
        return True  # Allow retry
    
    async def _recover_timeout_error(self, error: Exception, context: Dict) -> bool:
        """Recover from timeout error."""
        # Increase timeout for next attempt
        if 'timeout' in context:
            context['timeout'] *= 2
            logger.info(f"Increased timeout to {context['timeout']}s")
            return True
        
        return False
    
    async def _recover_memory_error(self, error: Exception, context: Dict) -> bool:
        """Recover from memory error."""
        try:
            # Trigger garbage collection
            import gc
            gc.collect()
            
            # Reduce batch size if applicable
            if 'batch_size' in context:
                context['batch_size'] = max(1, context['batch_size'] // 2)
                logger.info(f"Reduced batch size to {context['batch_size']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to recover from memory error: {e}")
        
        return False
    
    async def _recover_disk_full(self, error: Exception, context: Dict) -> bool:
        """Recover from disk full error."""
        try:
            # Clean up temporary files
            import tempfile
            import shutil
            
            temp_dir = tempfile.gettempdir()
            
            # Remove old temp files (older than 1 day)
            from pathlib import Path
            import time
            
            for temp_file in Path(temp_dir).glob("*"):
                try:
                    if time.time() - temp_file.stat().st_mtime > 86400:
                        if temp_file.is_file():
                            temp_file.unlink()
                        elif temp_file.is_dir():
                            shutil.rmtree(temp_file)
                except:
                    pass
            
            logger.info("Cleaned up temporary files")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clean up disk space: {e}")
        
        return False


class ErrorAggregator:
    """Aggregates and analyzes errors."""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize error aggregator.
        
        Args:
            window_size: Number of errors to keep
        """
        self.window_size = window_size
        self.errors: deque = deque(maxlen=window_size)
        self.error_counts = {}
        self.service_health = {}
    
    def record_error(self, error_event: ErrorEvent):
        """Record an error event."""
        self.errors.append(error_event)
        
        # Update counts
        key = f"{error_event.service}:{error_event.error_type}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        # Update service health
        if error_event.service not in self.service_health:
            self.service_health[error_event.service] = ServiceHealth(name=error_event.service)
        
        health = self.service_health[error_event.service]
        health.error_count += 1
        health.last_error = error_event
        health.consecutive_failures += 1
        
        # Check severity
        if error_event.severity == ErrorSeverity.CRITICAL:
            health.state = ServiceState.UNHEALTHY
        elif health.consecutive_failures > 3:
            health.state = ServiceState.DEGRADED
    
    def record_success(self, service: str):
        """Record a successful operation."""
        if service in self.service_health:
            health = self.service_health[service]
            health.success_count += 1
            health.consecutive_failures = 0
            health.last_success = datetime.now()
            
            # Update state
            if health.error_rate < 0.1:
                health.state = ServiceState.HEALTHY
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary."""
        now = datetime.now()
        
        # Recent errors (last hour)
        recent_errors = [
            e for e in self.errors
            if (now - e.timestamp).total_seconds() < 3600
        ]
        
        # Group by service
        by_service = {}
        for error in recent_errors:
            if error.service not in by_service:
                by_service[error.service] = []
            by_service[error.service].append(error)
        
        # Calculate statistics
        summary = {
            'total_errors': len(self.errors),
            'recent_errors': len(recent_errors),
            'error_rate_per_minute': len(recent_errors) / 60,
            'by_service': {},
            'by_severity': {
                'low': 0,
                'medium': 0,
                'high': 0,
                'critical': 0
            },
            'top_errors': [],
            'service_health': {}
        }
        
        # Service breakdown
        for service, errors in by_service.items():
            summary['by_service'][service] = {
                'count': len(errors),
                'types': list(set(e.error_type for e in errors))
            }
        
        # Severity breakdown
        for error in recent_errors:
            if error.severity == ErrorSeverity.LOW:
                summary['by_severity']['low'] += 1
            elif error.severity == ErrorSeverity.MEDIUM:
                summary['by_severity']['medium'] += 1
            elif error.severity == ErrorSeverity.HIGH:
                summary['by_severity']['high'] += 1
            elif error.severity == ErrorSeverity.CRITICAL:
                summary['by_severity']['critical'] += 1
        
        # Top errors
        top_errors = sorted(
            self.error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        summary['top_errors'] = [
            {'error': k, 'count': v} for k, v in top_errors
        ]
        
        # Service health
        for service, health in self.service_health.items():
            summary['service_health'][service] = {
                'state': health.state.value,
                'error_rate': health.error_rate,
                'consecutive_failures': health.consecutive_failures
            }
        
        return summary
    
    def should_alert(self) -> bool:
        """Check if should send alert."""
        # Alert if critical errors
        critical_errors = [
            e for e in self.errors
            if e.severity == ErrorSeverity.CRITICAL
            and (datetime.now() - e.timestamp).total_seconds() < 300
        ]
        
        if critical_errors:
            return True
        
        # Alert if service unhealthy
        unhealthy_services = [
            s for s, h in self.service_health.items()
            if h.state == ServiceState.UNHEALTHY
        ]
        
        if unhealthy_services:
            return True
        
        # Alert if high error rate
        recent_errors = [
            e for e in self.errors
            if (datetime.now() - e.timestamp).total_seconds() < 60
        ]
        
        if len(recent_errors) > 10:  # More than 10 errors per minute
            return True
        
        return False


class ResilientExecutor:
    """Execute operations with full error handling."""
    
    def __init__(self):
        """Initialize resilient executor."""
        self.circuit_breakers = {}
        self.retry_policy = RetryPolicy()
        self.error_recovery = ErrorRecovery()
        self.error_aggregator = ErrorAggregator()
    
    async def execute(self,
                     service: str,
                     func: Callable,
                     *args,
                     circuit_breaker: bool = True,
                     retry: bool = True,
                     recover: bool = True,
                     **kwargs) -> Any:
        """
        Execute function with resilience.
        
        Args:
            service: Service name
            func: Function to execute
            *args: Function arguments
            circuit_breaker: Use circuit breaker
            retry: Use retry policy
            recover: Attempt recovery
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        # Get or create circuit breaker
        if circuit_breaker and service not in self.circuit_breakers:
            self.circuit_breakers[service] = CircuitBreaker(service)
        
        try:
            # Execute with circuit breaker
            if circuit_breaker:
                cb = self.circuit_breakers[service]
                
                if retry:
                    # Retry with circuit breaker
                    result = await self.retry_policy.execute(
                        cb.call, func, *args, **kwargs
                    )
                else:
                    result = await cb.call(func, *args, **kwargs)
            else:
                # Execute directly
                if retry:
                    result = await self.retry_policy.execute(func, *args, **kwargs)
                else:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
            
            # Record success
            self.error_aggregator.record_success(service)
            
            return result
            
        except Exception as e:
            # Record error
            error_event = ErrorEvent(
                timestamp=datetime.now(),
                service=service,
                error_type=type(e).__name__,
                message=str(e),
                severity=self._classify_severity(e),
                traceback=traceback.format_exc(),
                context={'args': args, 'kwargs': kwargs}
            )
            
            self.error_aggregator.record_error(error_event)
            
            # Attempt recovery
            if recover:
                context = {'service': service, 'args': args, 'kwargs': kwargs}
                recovered = await self.error_recovery.attempt_recovery(e, context)
                
                if recovered:
                    # Retry after recovery
                    try:
                        if asyncio.iscoroutinefunction(func):
                            return await func(*args, **kwargs)
                        else:
                            return func(*args, **kwargs)
                    except:
                        pass
            
            # Check if should alert
            if self.error_aggregator.should_alert():
                logger.critical(f"System alert: High error rate detected")
            
            raise
    
    def _classify_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity."""
        if isinstance(error, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (FileNotFoundError, PermissionError)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        return {
            'circuit_breakers': {
                name: {
                    'state': cb.health.state.value,
                    'error_rate': cb.health.error_rate,
                    'consecutive_failures': cb.health.consecutive_failures
                }
                for name, cb in self.circuit_breakers.items()
            },
            'error_summary': self.error_aggregator.get_error_summary()
        }