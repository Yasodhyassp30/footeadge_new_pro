import logging

def configure_logger(level=logging.INFO):
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create a stream handler (outputs to console)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    # Add the stream handler to the logger
    logger.addHandler(stream_handler)

    return logger
