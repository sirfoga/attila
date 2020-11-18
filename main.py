from pathlib import Path

from util.config import get_config
from util.plots import plot_sample

from data.parse import parse_data, get_data

from nn.models.unet import build as build_model
from nn.metrics import mean_IoU, DSC
from nn.core import do_training, do_inference, do_evaluation

from data.experiments import save as save_exp
from data.prepare import train_validate_test_split, get_weights_file

_here = Path('.').resolve()


def get_default_args(config):
  conv_kernel_size = 3
  pool_size = 2
  
  model_args = {
    'n_filters': config.getint('unet', 'n filters'),
    'n_layers': config.getint('unet', 'n layers'),
    'kernel_size': conv_kernel_size,
    'pool_size': pool_size,
    'n_classes': 1,  # the other is 1 - ... (because it's a probability distribution)
    'final_activation': config.get('unet', 'final activation'),
    'dropout': config.getfloat('unet', 'dropout'),
    'batchnorm': config.getboolean('unet', 'batchnorm')
  }

  compile_args = {
    'optimizer': config.get('training', 'optimizer'),
    'loss': config.get('training', 'loss'),
    'metrics': ['accuracy', mean_IoU, DSC]
  }

  return model_args, compile_args


def main():
  config = get_config(_here / './config.ini')

  data_path = _here / config.get('data', 'folder')
  data_path = data_path.resolve()

  out_path = Path(config.get('experiments', 'output folder')).resolve()
  out_path.mkdir(parents=True, exist_ok=True)  # rm and mkdir if existing

  # todo use experiments_file = out_path / config.get('experiment', 'output file')

  images_path = data_path / config.get('data folder', 'images')
  masks_path = data_path / config.get('data folder', 'masks')
  raw = get_data(images_path, masks_path)
  X, y = parse_data(
    raw,
    (config.getint('image', 'width'), config.getint('image', 'height'))
  )
  X_train, X_val, X_test, y_train, y_val, y_test = train_validate_test_split(
    X,
    y,
    config.getfloat('experiment', 'val size'),
    config.getfloat('experiment', 'test size')
  )

  print('X ~ {}, y ~ {}'.format(X.shape, y.shape))
  plot_sample(X, y, cmap=config.get('image', 'cmap'), out_folder=_here)

  model_args, compile_args = get_default_args(config)
  args = {
    **model_args,
    'padding': 'same',
    'use_skip_conn': True,
    'use_se_block': False
  }
  model = build_model(**args)
  weights_file = str(get_weights_file(out_path, 'wow'))
  history = do_training(
    model,
    X_train,
    X_val,
    y_train,
    y_val,
    weights_file,
    config.getint('training', 'batch size'),
    config.getint('training', 'epochs'),
    compile_args,
    config.getint('experiments', 'verbose')
  )
  preds = do_inference(
    model,
    weights_file,
    X_val,
    config.getint('training', 'batch size'),
    config.getint('experiments', 'verbose')
  )

  # todo eval


if __name__ == '__main__':
  main()
