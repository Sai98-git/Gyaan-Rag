import time
import logging
from typing import Callable, TypeVar, Any, Tuple, Type

logger = logging.getLogger(__name__)

T = TypeVar("T")

def execute_with_retry(
    operation: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    max_delay: float = 4.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    operation_name: str = "External API call"
) -> T:
    """
    Executes a callable with bounded exponential backoff retries.
    
    Args:
        operation: The zero-argument function to execute.
        max_retries: Maximum number of retry attempts after the first failure.
        initial_delay: Initial delay in seconds before first retry.
        backoff_factor: Multiplier for consecutive delay intervals.
        max_delay: Cap on maximum retry delay in seconds.
        retryable_exceptions: Tuple of exception classes that should trigger a retry.
        operation_name: Human-readable name for logging.
        
    Returns:
        The result of the operation.
        
    Raises:
        The last encountered exception if all retries are exhausted or non-retryable.
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(1, max_retries + 2):
        try:
            return operation()
        except retryable_exceptions as exc:
            last_exception = exc
            if attempt <= max_retries:
                logger.warning(
                    f"[{operation_name}] Attempt {attempt}/{max_retries + 1} failed: {exc}. "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                logger.error(
                    f"[{operation_name}] Exhausted all {max_retries + 1} attempts. Last error: {exc}"
                )
                break
                
    if last_exception:
        raise last_exception
    raise RuntimeError(f"[{operation_name}] Operation failed with unknown error.")
