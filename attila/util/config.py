from pathlib import Path
from configparser import ConfigParser


def get_config(file_path):
    config = ConfigParser()
    config.read(file_path)
    return config


def is_verbose(key, config):
    return config.getboolean(key, 'verbose')


def check_input(config):
    def _is_square_img():
        return config.getint('image', 'width') == config.getint('image', 'height')

    def _is_integer(x):
        return x % 1 == 0

    def _is_good_input():
        # todo b = config.getint('image', 'width')  # input size
        # d = 
        # gamma = 
        # pool_size = 
        # a = (b - (2 ** d - 1) * gamma) / 2 ** d
        assert _is_integer(0.0)

    assert _is_square_img()  # only square images are supported


def get_env(root_folder):
    config = get_config(root_folder / './config.ini')
    check_input(config)


    data_path = root_folder / config.get('data', 'folder')
    data_path = data_path.resolve()

    out_path = Path(config.get('experiments', 'output folder')).resolve()

    models_config = root_folder / 'experiments.json'

    return config, data_path, out_path, models_config
