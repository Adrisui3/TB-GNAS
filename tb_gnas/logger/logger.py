import logging


class Logger:
    def __init__(self, log_level=logging.DEBUG):
        # Configure the logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=None,
            filemode='w'
        )

        self.logger = logging.getLogger(__name__)

    def info(self, message: str):
        self.logger.info(message)

    def debug(self, message: str):
        self.logger.debug(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)
