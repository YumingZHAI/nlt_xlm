# Yuming learns how to log both to the console and to a file
import logging
from logging.handlers import RotatingFileHandler
import datetime

e = datetime.datetime.now()
date = e.strftime("%d-%m-%Y")

logger = logging.getLogger()
# debug level, write all
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
# append mode, 1 backup, max size of 2Mo
file_handler = RotatingFileHandler('./path/' + date + ".log", 'a', 2000000, 1)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)

logger.info('Hello')
logger.warning('Testing %s', 'foo')
