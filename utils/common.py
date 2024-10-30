import logging


def get_logger(name, file_name):
    log_format = "%(asctime)s | %(message)s"
    logging.basicConfig(
        level=logging.DEBUG, format=log_format, filename=file_name, filemode="w"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(name).addHandler(console)
    return logging.getLogger(name)
