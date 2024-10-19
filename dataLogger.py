import logging
from functools import wraps
from datetime import datetime

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_function_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        function_name = func.__name__
        #current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Function '{function_name}' called")
        return func(*args, **kwargs)
    return wrapper

def debug_log(log_enabled,log_string:str):
    '''
    DEBUG FUNCTION
    Output formatted log entries to stdout if "enable_log" is true
    '''
    if not log_enabled:
        exit
    else:
        output_string = ""
        datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output_string = f"{datetime_str:>15}: {log_string}"
        print(output_string)
    # return log_string

# Example usage
@log_function_call
def example_function(cat, bird, mouse):
    print("This is an example function")

if __name__ == "__main__":
    example_function("a", 2, 4)