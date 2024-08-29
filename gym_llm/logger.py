import os
import logging

DEFAULT_LOGGER_NAME = 'gym_llm_logger'
LOGGER_NAME = os.getenv('LOGGER_NAME', DEFAULT_LOGGER_NAME)


def get_logger():
    if LOGGER_NAME not in logging.Logger.manager.loggerDict.keys():
        logger = logging.getLogger(LOGGER_NAME)

        if LOGGER_NAME == DEFAULT_LOGGER_NAME:
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                ch = logging.StreamHandler()
                ch.setLevel(logging.INFO)
                formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
                ch.setFormatter(formatter)
                logger.addHandler(ch)
    else:
        logger = logging.getLogger(LOGGER_NAME)

    return logger


class EmptyLogger:
    def info(self, message):
        pass

    def error(self, message):
        pass

    def warning(self, message):
        pass

    def debug(self, message):
        pass