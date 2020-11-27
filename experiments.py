from pathlib import Path
import json

from attila.data.parse import parse_data, get_data
from attila.util.config import get_config
from attila.experiments.do import do_batch_experiments
from attila.experiments.tools import create_tex_experiments

_here = Path('.').resolve()


def check_input(config):
    def _is_square_img():
        return config.getint('image', 'width') == config.getint('image', 'height')

    def _is_integer(x):
        return x % 1 == 0

    def _is_good_input():
        # b = config.getint('image', 'width')  # input size
        # d = 
        # gamma = 
        # pool_size = 
        # a = (b - (2 ** d - 1) * gamma) / 2 ** d
        assert _is_integer(0.0)

    assert _is_square_img()  # only square images are supported



def main():
    config = get_config(_here / './config.ini')
    check_input(config)

    config = get_config(_here / 'config.ini')

    data_path = _here / config.get('data', 'folder')
    data_path = data_path.resolve()

    out_path = Path(config.get('experiments', 'output folder')).resolve()
    out_path.mkdir(parents=True, exist_ok=True)  # rm and mkdir if existing

    images_path = data_path / config.get('data', 'images')
    masks_path = data_path / config.get('data', 'masks')

    raw = get_data(images_path, masks_path)
    X, y = parse_data(
        raw,
        (config.getint('image', 'width'), config.getint('image', 'height'))
    )

    experiments_file = _here / 'experiments.json'
    with open(experiments_file, 'r') as fp:
        experiments = json.load(fp)

    do_batch_experiments(experiments, (X, y), config, out_path)
    create_tex_experiments(config, out_path, out_path / 'tables.tex')


if __name__ == '__main__':
  main()
