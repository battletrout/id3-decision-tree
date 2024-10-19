import logging
from functools import wraps
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_function_call(func):
    """
    A decorator that logs the function call with its name.

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: The wrapped function that logs its call before execution.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        function_name = func.__name__
        logger.info(f"Function '{function_name}' called")
        return func(*args, **kwargs)
    return wrapper

def debug_log(log_enabled: bool, log_string: str) -> None:
    """
    Output formatted log entries to stdout if logging is enabled.

    Args:
        log_enabled (bool): Whether logging is enabled.
        log_string (str): The message to be logged.

    Returns:
        None
    """
    if not log_enabled:
        return
    else:
        datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output_string = f"{datetime_str:>15}: {log_string}"
        print(output_string)

@log_function_call
def example_function(a: str, b: int, c: int) -> None:
    """
    An example function to demonstrate the usage of the log_function_call decorator.
    """
    debug_log(True,"This is an example function")

if __name__ == "__main__":
    example_function("a", 2, 4)