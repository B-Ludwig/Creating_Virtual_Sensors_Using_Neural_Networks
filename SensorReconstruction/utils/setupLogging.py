import logging

def setup_logging(level=logging.INFO):
    """
    Configures the logging settings.
    Logs messages with the specified level and higher to the console.

    Parameters:
    - level (int): The logging level (e.g., logging.DEBUG, logging.INFO).
                   Defaults to logging.INFO.
    """
    # Create a logger object
    logger = logging.getLogger()
    logger.setLevel(level)  # Set the minimum log level based on the parameter

    # Remove any existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a console handler to output logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)  # Set handler log level based on the parameter

    # Define a log message format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)