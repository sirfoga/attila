import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateful_random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.util.tf_export import keras_export

ResizeMethod = image_ops.ResizeMethod

_RESIZE_METHODS = {
    'bilinear': ResizeMethod.BILINEAR,
    'nearest': ResizeMethod.NEAREST_NEIGHBOR,
    'bicubic': ResizeMethod.BICUBIC,
    'area': ResizeMethod.AREA,
    'lanczos3': ResizeMethod.LANCZOS3,
    'lanczos5': ResizeMethod.LANCZOS5,
    'gaussian': ResizeMethod.GAUSSIAN,
    'mitchellcubic': ResizeMethod.MITCHELLCUBIC
}

H_AXIS = 1
W_AXIS = 2


class CenterCrop(Layer):
  """Crop the central portion of the images to target height and width.
  Input shape:
    4D tensor with shape:
    `(samples, height, width, channels)`, data_format='channels_last'.
  Output shape:
    4D tensor with shape:
    `(samples, target_height, target_width, channels)`.
  If the input height/width is even and the target height/width is odd (or
  inversely), the input image is left-padded by 1 pixel.
  Arguments:
    height: Integer, the height of the output shape.
    width: Integer, the width of the output shape.
    name: A string, the name of the layer.
  """

  def __init__(self, height, width, name=None, **kwargs):
    self.target_height = height
    self.target_width = width
    self.input_spec = InputSpec(ndim=4)
    super(CenterCrop, self).__init__(name=name, **kwargs)

  def call(self, inputs):
    inputs_shape = array_ops.shape(inputs)
    img_hd = inputs_shape[H_AXIS]
    img_wd = inputs_shape[W_AXIS]
    img_hd_diff = img_hd - self.target_height
    img_wd_diff = img_wd - self.target_width
    checks = []
    checks.append(
        check_ops.assert_non_negative(
            img_hd_diff,
            message='The crop height {} should not be greater than input '
            'height.'.format(self.target_height)))
    checks.append(
        check_ops.assert_non_negative(
            img_wd_diff,
            message='The crop width {} should not be greater than input '
            'width.'.format(self.target_width)))
    with ops.control_dependencies(checks):
      bbox_h_start = math_ops.cast(img_hd_diff / 2, dtypes.int32)
      bbox_w_start = math_ops.cast(img_wd_diff / 2, dtypes.int32)
      bbox_begin = array_ops.stack([0, bbox_h_start, bbox_w_start, 0])
      bbox_size = array_ops.stack(
          [-1, self.target_height, self.target_width, -1])
      outputs = array_ops.slice(inputs, bbox_begin, bbox_size)
      return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    return tensor_shape.TensorShape(
        [input_shape[0], self.target_height, self.target_width, input_shape[3]])

  def get_config(self):
    config = {
        'height': self.target_height,
        'width': self.target_width,
    }
    base_config = super(CenterCrop, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
