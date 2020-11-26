from configparser import ConfigParser


def get_config(file_path):
    config = ConfigParser()
    config.read(file_path)
    return config


def is_verbose(key, config):
    return config.getint(key, 'verbose')  # todo get bool
