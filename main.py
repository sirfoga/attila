from pathlib import Path
import numpy as np

from util.config import get_config
from util.plots import plot_preds, plot_history

from data.parse import parse_data, get_data
from data.experiments import load_experiments, save_experiments

from nn.models.unet import calc_out_size, build as build_model
from nn.metrics import mean_IoU, DSC
from nn.core import do_training, do_evaluation

from data.prepare import train_validate_test_split, get_weights_file, get_model_output_folder, describe
from data.trans import crop_center_transformation, apply_transformations

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


def do_experiment(experiment, data, config, out_path):  # todo refactor
  def _fix_data_shape(img_out_shape):
    def _f(x):
      output_shape = (*img_out_shape, config.getint('image', 'depth'))

      transformations = [
        crop_center_transformation(output_shape),
      ]
      x = apply_transformations(x, transformations)  # reduce output size
      x = np.array(x)
      return x

    return _f


  def _prepare_data(data):
    (X_train, X_val, X_test, y_train, y_val, y_test) = data  # unpack

    img_shape = y_train.shape[1: 2 + 1]  # width, height of input images
    img_out_shape = calc_out_size(
      config.getint('unet', 'n layers'),
      2,
      3,
      2,
      experiment['padding']
    )(img_shape)

    y_train = _fix_data_shape(img_out_shape)(y_train)
    y_val = _fix_data_shape(img_out_shape)(y_val)
    y_test = _fix_data_shape(img_out_shape)(y_test)

    return (X_train, X_val, X_test, y_train, y_val, y_test)


  (X_train, X_val, X_test, y_train, y_val, y_test) = _prepare_data(data)

  if config.getint('experiments', 'verbose'):
    describe(X_train, X_val, X_test, y_train, y_val, y_test)

  model_args, compile_args = get_default_args(config)
  args = {
    **model_args,
    'padding': experiment['padding'],
    'use_skip_conn': experiment['use_skip_conn'],
    'use_se_block': experiment['use_se_block']
  }
  model = build_model(**args)
  weights_file = str(get_weights_file(out_path, experiment['name']))

  results = do_training(
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

  stats, preds = do_evaluation(
    model,
    weights_file,
    X_test,
    y_test,
    config.getint('training', 'batch size'),
    config.getint('experiments', 'verbose')
  )

  plot_preds(
    X_test,
    y_test,
    preds,
    cmap=config.get('image', 'cmap'),
    title='model: {}'.format(experiment['name']),
    out_folder=get_model_output_folder(out_path, experiment['name'])
  )

  return results, stats


def do_experiments(experiments, data, config, out_path):  # todo refactor
  if config.getint('experiments', 'verbose'):
    print('ready to perform {} experiments'.format(len(experiments)))

  X, y = data  # unpack
  X_train, X_val, X_test, y_train, y_val, y_test = train_validate_test_split(
    X,
    y,
    config.getfloat('experiment', 'val size'),
    config.getfloat('experiment', 'test size')
  )

  for i, experiment in enumerate(experiments):
    if config.getint('experiments', 'verbose'):
      print('=== experiment # {} / {}: {}'.format(i + 1, len(experiments), experiment['name']))

    data = (X_train, X_val, X_test, y_train, y_val, y_test)
    results, eval_stats = do_experiment(experiment, data, config, out_path)

    experiments[i]['history'] = results.history
    experiments[i]['eval'] = eval_stats

  last_epochs = int(config.getint('training', 'epochs') * 0.8)
  plot_history(experiments, out_path / 'history.png', last=last_epochs)

  return experiments


def main():
  config = get_config(_here / './config.ini')

  data_path = _here / config.get('data', 'folder')
  data_path = data_path.resolve()

  out_path = Path(config.get('experiments', 'output folder')).resolve()
  out_path.mkdir(parents=True, exist_ok=True)  # rm and mkdir if existing

  images_path = data_path / config.get('data folder', 'images')
  masks_path = data_path / config.get('data folder', 'masks')
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
