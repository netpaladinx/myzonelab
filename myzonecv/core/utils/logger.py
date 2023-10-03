import logging

import torch.distributed as dist

LOGGER_NAMES = set()

DASH_LINE = '-' * 60 + '\n'

LOGGER_GLOBAL_VARS = {}


def get_root_logger(log_file=None, log_level=logging.INFO):
    log_name = __name__.split('.')[0]
    return get_logger(log_name, log_file=log_file, log_level=log_level)


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    if log_file is None:
        log_file = LOGGER_GLOBAL_VARS.get('log_file')
    else:
        LOGGER_GLOBAL_VARS['log_file'] = log_file

    logger = logging.getLogger(name)
    if name.split('.')[0] in LOGGER_NAMES:
        return logger

    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    LOGGER_NAMES.add(name)
    return logger
