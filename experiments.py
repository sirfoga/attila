from pathlib import Path

from attila.util.config import get_config
from attila.data.parse import parse_data, get_data
from attila.experiments.data import load_experiments, save_experiments
from attila.experiments.do import do_experiments

_here = Path('.').resolve()


def main():
  config = get_config(_here / './config.ini')

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

  experiments_file = _here / config.get('experiments', 'output file')
  experiments = load_experiments(experiments_file)
  do_experiments(experiments, (X, y), config, out_path)

  save_experiments(experiments, out_path / config.get('experiments', 'output file'))


if __name__ == '__main__':
  main()
