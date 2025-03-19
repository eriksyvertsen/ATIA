"""
Enhanced error handling for ATIA.

This module provides improved error handling, validation, and exception management
for the ATIA system in Phase 4.
"""

import logging
import traceback
import functools
import time
from typing import Any, Callable, Dict, List, Type, TypeVar, Optional, Union
import asyncio

from atia.utils.analytics import track_error

# Type variable for return values
T = TypeVar('T')

logger = logging.getLogger(__name__)


class ATIAError(Exception):
    """Base exception class for all ATIA errors."""
    def __init__(self, message: str, component: str = "unknown", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.component = component
        self.details = details or {}
        self.timestamp = time.time()

        # Track the error
        track_error(component, self.__class__.__name__, message, metadata=details)


class NetworkError(ATIAError):
    """Error related to network operations."""
    pass


class APIError(ATIAError):
    """Error when interacting with external APIs."""
    pass


class AuthenticationError(ATIAError):
    """Error when authenticating with external services."""
    pass


class DocumentationProcessingError(ATIAError):
    """Error when processing API documentation."""
    pass


class ValidationError(ATIAError):
    """Error when validating data or parameters."""
    pass


class ConfigurationError(ATIAError):
    """Error related to system configuration."""
    pass


class LLMError(ATIAError):
    """Error when interacting with OpenAI or other LLM services."""
    pass


class ToolExecutionError(ATIAError):
    """Error when executing a tool."""
    pass


class StorageError(ATIAError):
    """Error with storage operations (file, database, etc.)."""
    pass


def async_retry(
    max_tries: int = 3,
    exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception,
    delay: float = 1.0,
    backoff: float = 2.0,
    logger_obj: Optional[logging.Logger] = None
) -> Callable:
    """
    Retry async function decorator with exponential backoff.

    Args:
        max_tries: Maximum number of attempts
        exceptions: Exception(s) to catch and retry on
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier
        logger_obj: Optional logger object

    Returns:
        Decorated function
    """
    log = logger_obj or logger

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tries = 0
            current_delay = delay

            while tries < max_tries:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    tries += 1
                    if tries >= max_tries:
                        log.error(f"Function {func.__name__} failed after {tries} attempts. Error: {e}")
                        raise

                    log.warning(f"Attempt {tries} failed in {func.__name__}. Retrying in {current_delay} seconds. Error: {e}")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

        return wrapper

    return decorator


def sync_retry(
    max_tries: int = 3,
    exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception,
    delay: float = 1.0,
    backoff: float = 2.0,
    logger_obj: Optional[logging.Logger] = None
) -> Callable:
    """
    Retry sync function decorator with exponential backoff.

    Args:
        max_tries: Maximum number of attempts
        exceptions: Exception(s) to catch and retry on
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier
        logger_obj: Optional logger object

    Returns:
        Decorated function
    """
    log = logger_obj or logger

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tries = 0
            current_delay = delay

            while tries < max_tries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    tries += 1
                    if tries >= max_tries:
                        log.error(f"Function {func.__name__} failed after {tries} attempts. Error: {e}")
                        raise

                    log.warning(f"Attempt {tries} failed in {func.__name__}. Retrying in {current_delay} seconds. Error: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff

        return wrapper

    return decorator


def catch_and_log(
    component: str,
    exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception,
    default_return: Any = None,
    raise_atia_error: bool = False,
    atia_error_class: Type[ATIAError] = ATIAError
) -> Callable:
    """
    Decorator to catch exceptions, log them, and optionally convert to ATIA errors.

    Args:
        component: Component name for tracking
        exceptions: Exception(s) to catch
        default_return: Default return value if an exception is caught
        raise_atia_error: Whether to raise an ATIA error after catching
        atia_error_class: ATIA error class to use if raising

    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                # Capture stack trace
                stack_trace = traceback.format_exc()

                # Log the error
                logger.error(f"Error in {func.__name__} ({component}): {e}")
                logger.debug(f"Stack trace: {stack_trace}")

                # Track the error
                track_error(
                    component=component,
                    error_type=e.__class__.__name__,
                    error_message=str(e),
                    metadata={"stack_trace": stack_trace, "function": func.__name__}
                )

                # Raise appropriate ATIA error or return default
                if raise_atia_error:
                    raise atia_error_class(
                        message=str(e),
                        component=component,
                        details={"original_error": e.__class__.__name__, "stack_trace": stack_trace}
                    ) from e

                return default_return

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                # Capture stack trace
                stack_trace = traceback.format_exc()

                # Log the error
                logger.error(f"Error in {func.__name__} ({component}): {e}")
                logger.debug(f"Stack trace: {stack_trace}")

                # Track the error
                track_error(
                    component=component,
                    error_type=e.__class__.__name__,
                    error_message=str(e),
                    metadata={"stack_trace": stack_trace, "function": func.__name__}
                )

                # Raise appropriate ATIA error or return default
                if raise_atia_error:
                    raise atia_error_class(
                        message=str(e),
                        component=component,
                        details={"original_error": e.__class__.__name__, "stack_trace": stack_trace}
                    ) from e

                return default_return

        # Return the appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def validate_argument(arg_name: str, validation_func: Callable[[Any], bool], 
                     error_message: str = "Invalid argument") -> Callable:
    """
    Decorator to validate function arguments.

    Args:
        arg_name: Name of the argument to validate
        validation_func: Function that returns True if the argument is valid
        error_message: Error message to use if validation fails

    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Get the argument value
            arg_value = kwargs.get(arg_name) if arg_name in kwargs else None

            # If the argument is not in kwargs, try to find it in positional args
            if arg_value is None:
                # Get function parameter names
                import inspect
                param_names = inspect.signature(func).parameters.keys()
                param_names_list = list(param_names)

                # Find the index of the argument
                if arg_name in param_names_list:
                    arg_index = param_names_list.index(arg_name)
                    if arg_index < len(args):
                        arg_value = args[arg_index]

            # Validate the argument
            if arg_value is not None and not validation_func(arg_value):
                raise ValidationError(
                    message=f"{error_message}: {arg_name}",
                    component=func.__module__,
                    details={"argument": arg_name, "value": str(arg_value)}
                )

            return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get the argument value
            arg_value = kwargs.get(arg_name) if arg_name in kwargs else None

            # If the argument is not in kwargs, try to find it in positional args
            if arg_value is None:
                # Get function parameter names
                import inspect
                param_names = inspect.signature(func).parameters.keys()
                param_names_list = list(param_names)

                # Find the index of the argument
                if arg_name in param_names_list:
                    arg_index = param_names_list.index(arg_name)
                    if arg_index < len(args):
                        arg_value = args[arg_index]

            # Validate the argument
            if arg_value is not None and not validation_func(arg_value):
                raise ValidationError(
                    message=f"{error_message}: {arg_name}",
                    component=func.__module__,
                    details={"argument": arg_name, "value": str(arg_value)}
                )

            return await func(*args, **kwargs)

        # Return the appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def validate_return(validation_func: Callable[[Any], bool], 
                  error_message: str = "Invalid return value") -> Callable:
    """
    Decorator to validate function return values.

    Args:
        validation_func: Function that returns True if the return value is valid
        error_message: Error message to use if validation fails

    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return_value = func(*args, **kwargs)

            if not validation_func(return_value):
                raise ValidationError(
                    message=error_message,
                    component=func.__module__,
                    details={"function": func.__name__, "return_value": str(return_value)}
                )

            return return_value

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return_value = await func(*args, **kwargs)

            if not validation_func(return_value):
                raise ValidationError(
                    message=error_message,
                    component=func.__module__,
                    details={"function": func.__name__, "return_value": str(return_value)}
                )

            return return_value

        # Return the appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def timer(component: str, method_name: Optional[str] = None, 
         track_analytics: bool = True) -> Callable:
    """
    Decorator to time function execution.

    Args:
        component: Component name for tracking
        method_name: Optional method name override
        track_analytics: Whether to track timing in analytics

    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                func_name = method_name or func.__name__

                # Log the timing
                logger.debug(f"{component}.{func_name} executed in {duration_ms:.2f}ms")

                # Track in analytics if requested
                if track_analytics:
                    from atia.utils.analytics import track_method_timing
                    track_method_timing(component, func_name, duration_ms)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                func_name = method_name or func.__name__

                # Log the timing
                logger.debug(f"{component}.{func_name} executed in {duration_ms:.2f}ms")

                # Track in analytics if requested
                if track_analytics:
                    from atia.utils.analytics import track_method_timing
                    track_method_timing(component, func_name, duration_ms)

        # Return the appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Add a track_method_timing function to analytics.py
def track_method_timing(component: str, method_name: str, duration_ms: float, 
                       metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Track method execution timing.

    Args:
        component: Component name
        method_name: Method name
        duration_ms: Duration in milliseconds
        metadata: Additional metadata
    """
    if metadata is None:
        metadata = {}

    metadata["method"] = method_name

    from atia.utils.analytics import track_event, UsageEvent
    from datetime import datetime

    event = UsageEvent(
        event_type="method_timing",
        timestamp=datetime.now(),
        component=component,
        duration_ms=duration_ms,
        metadata=metadata
    )

    from atia.utils.analytics import dashboard
    dashboard.track_event(event)


def handle_exceptions_middleware(app):
    """
    Middleware to handle exceptions in FastAPI/Starlette apps.

    Args:
        app: FastAPI/Starlette app

    Returns:
        Middleware function
    """
    @app.middleware("http")
    async def exception_middleware(request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            # Log the error
            logger.error(f"Error processing request {request.url}: {e}")

            # Track the error
            track_error(
                component="api",
                error_type=e.__class__.__name__,
                error_message=str(e),
                metadata={"url": str(request.url), "method": request.method}
            )

            # Return error response
            from starlette.responses import JSONResponse
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "message": str(e)}
            )

    return app


class ErrorContext:
    """
    Context manager for error handling.

    Example:
        with ErrorContext("api_discovery", "Search failed"):
            result = api_discovery.search(query)
    """

    def __init__(self, component: str, message: str, 
                error_class: Type[ATIAError] = ATIAError,
                default_return: Any = None):
        """
        Initialize the error context.

        Args:
            component: Component name
            message: Error message
            error_class: ATIA error class to use
            default_return: Default return value if an error occurs
        """
        self.component = component
        self.message = message
        self.error_class = error_class
        self.default_return = default_return

    def __enter__(self):
        """Enter the context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context, handling any errors."""
        if exc_type is not None:
            # Log the error
            logger.error(f"Error in {self.component}: {self.message} - {exc_val}")

            # Track the error
            track_error(
                component=self.component,
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                metadata={"context_message": self.message}
            )

            # If not an ATIA error, wrap it
            if not isinstance(exc_val, ATIAError):
                raise self.error_class(
                    message=f"{self.message}: {exc_val}",
                    component=self.component,
                    details={"original_error": exc_type.__name__}
                ) from exc_val

            return True if self.default_return is not None else False

        return False


class AsyncErrorContext:
    """
    Async context manager for error handling.

    Example:
        async with AsyncErrorContext("api_discovery", "Search failed"):
            result = await api_discovery.search(query)
    """

    def __init__(self, component: str, message: str, 
                error_class: Type[ATIAError] = ATIAError,
                default_return: Any = None):
        """
        Initialize the async error context.

        Args:
            component: Component name
            message: Error message
            error_class: ATIA error class to use
            default_return: Default return value if an error occurs
        """
        self.component = component
        self.message = message
        self.error_class = error_class
        self.default_return = default_return

    async def __aenter__(self):
        """Enter the context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context, handling any errors."""
        if exc_type is not None:
            # Log the error
            logger.error(f"Error in {self.component}: {self.message} - {exc_val}")

            # Track the error
            track_error(
                component=self.component,
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                metadata={"context_message": self.message}
            )

            # If not an ATIA error, wrap it
            if not isinstance(exc_val, ATIAError):
                raise self.error_class(
                    message=f"{self.message}: {exc_val}",
                    component=self.component,
                    details={"original_error": exc_type.__name__}
                ) from exc_val

            return True if self.default_return is not None else False

        return False