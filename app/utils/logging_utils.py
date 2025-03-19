import logging
import traceback
import functools
import inspect

logger = logging.getLogger(__name__)

def log_exceptions(func):
    """
    Decorator to log exceptions with detailed location information.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get calling information
            frame = inspect.currentframe().f_back
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            function_name = frame.f_code.co_name
            
            # Get class name if applicable
            if args and hasattr(args[0], '__class__'):
                class_name = args[0].__class__.__name__
            else:
                class_name = "Module"
            
            # Log the exception with details
            logger.error(
                f"Exception in {filename}:{lineno} - {class_name}.{function_name}(): {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            
            # Re-raise the exception
            raise
    
    return wrapper