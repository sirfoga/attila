from pathlib import Path

from attila.data.parse import parse_data, get_data
from attila.experiments.do import do_batch_experiments
from attila.util.config import get_env
from attila.util.io import load_json

_here = Path('.').resolve()


def main():
    config, data_path, out_path, models_config_path = get_env(_here)
    out_path = out_path / 'big'
    out_path.mkdir(parents=True, exist_ok=True)  # rm and mkdir if existing

    images_path = data_path / config.get('data', 'images')
    masks_path = data_path / config.get('data', 'masks')

    raw = get_data(images_path, masks_path)
    X, y = parse_data(
        raw,
        (config.getint('image', 'width'), config.getint('image', 'height'))
    )

    models_config = load_json(models_config_path)
    do_batch_experiments(models_config, (X, y), config, out_path)


if __name__ == '__main__':
  main()
