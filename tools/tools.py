"""
various helper functions to make life easier
"""

import time
import subprocess
import os
from pathlib import Path
import sys
from datetime import datetime
import logging
import re
import string
import random


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def us2mc(x):
    """
    underscore to mixed-case notation
    from https://www.safaribooksonline.com/library/view/python-cookbook/0596001673/ch03s16.html
    """
    return re.sub(r'_([a-z])', lambda m: (m.group(1).upper()), x)


def us2cw(x):
    """
    underscore to capwords notation
    from https://www.safaribooksonline.com/library/view/python-cookbook/0596001673/ch03s16.html
    """
    s = us2mc(x)
    return s[0].upper()+s[1:]


def import_class(package_path, class_name):
    """
    dynamic import of a class from a given package
    :param package_path: path to the package
    :param class_name: class to be dynamically loaded
    :return: dynamically loaded class
    """
    try:
        logging.info(f"Loading {package_path}.{class_name} ...")
        module = __import__(f"{package_path}.{class_name}", fromlist=[class_name])
        return getattr(module, us2cw(class_name))
    except ModuleNotFoundError as exc:
        logging.error(f"{package_path}.{class_name} could not be found")
        exit(1)


def setup_logger():
    """
    setup the logging mechanism where log messages are saved in time-encoded txt files as well as to the terminal
    :return: directory path in which logs are saved
    """
    directory_path = f"logs/{datetime.now():%Y-%m-%d@%H:%M:%S}_{id_generator()}"

    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)

    log_format = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] [%(pathname)s:%(lineno)04d] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(f"{directory_path}/{datetime.now():%Y-%m-%d@%H:%M:%S}_{id_generator()}.log")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    file_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    logging.root = logger

    logging.info('START LOGGING')
    logging.info(f"Current Git Version: {git_version()}")

    return directory_path


def pretty_print_arguments(args):
    """
    return a nicely formatted list of passed arguments
    :param args: arguments passed to the program via terminal
    :return: None
    """
    longest_key = max([len(key) for key in vars(args)])

    print('Program was launched with the following arguments:')

    for key, item in vars(args).items():
        print("~ {0:{s}} \t {1}".format(key, item, s=longest_key))

    print('')
    # Wait a bit until program execution continues
    time.sleep(0.1)


def git_version():
    """
    return git revision such that it can be also logged to keep track of results
    :return: git revision hash
    """
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        git_revision = out.strip().decode('ascii')
    except OSError:
        git_revision = "Unknown"

    return git_revision
