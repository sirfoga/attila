from sklearn.model_selection import train_test_split


def train_validate_test_split(X, y, valid_size, test_size, *args, **kwargs):
  X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=valid_size + test_size, *args, **kwargs
  )
  test_size = test_size / (test_size + valid_size)  # wrt to valid
  X_val, X_test, y_val, y_test = train_test_split(
    X_val, y_val, test_size=test_size, **kwargs
  )
  return X_train, X_val, X_test, y_train, y_val, y_test


def describe(X_train, X_val, X_test, y_train, y_val, y_test):
  print('= dataset training: X ~ {}, y ~ {}'.format(X_train.shape, y_train.shape))
  print('= dataset validation: X ~ {}, y ~ {}'.format(X_val.shape, y_val.shape))
  print('= dataset test (not used): X ~ {}, y ~ {}'.format(X_test.shape, y_test.shape))


def get_model_output_folder(out_folder, model_name, mkdir=True):
  out_path = out_folder / model_name  # a folder for each model
  if mkdir:
    out_path.mkdir(parents=True, exist_ok=True)  # mkdir -p, removes if existing

  return out_path


def get_weights_file(out_folder, model_name, extension='.h5'):
  file_name = 'model' + extension  # same filename for each model
  return get_model_output_folder(out_folder, model_name) / file_name
