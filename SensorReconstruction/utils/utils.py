import os
import logging

logger = logging.getLogger(__name__)

def create_dir(path, dir_name):
    path = os.path.join(path, dir_name)

    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f'{path} was created.')
