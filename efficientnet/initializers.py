import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.initializers import Initializer
from tensorflow.keras.utils.generic_utils import get_custom_objects

class EfficientConv2DKernelInitializer(Initializer):
    def __call__(self, shape, dtype=K.floatx(), **kwargs):
        kernel_height, kernel_width, _, out_filters = shape
        fan_out = int(kernel_height * kernel_width * out_filters)
        return tf.random_normal(
            shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype
        )

class EfficientDenseKernelInitializer(Initializer):
    def __call__(self, shape, dtype=K.floatx(), **kwargs):
        init_range = 1.0 / np.sqrt(shape[1])
        return tf.random_normal(shape, -init_range, init_range, dtype=dtype)

conv_kernel_initializer = EfficientConv2DKernelInitializer()
dense_kernel_initializer = EfficientDenseKernelInitializer()

get_custom_objects().update({
    'EfficientDenseKernelInitializer': EfficientDenseKernelInitializer, 
    'EfficientConv2DKernelInitializer': EfficientDenseKernelInitializer
})
