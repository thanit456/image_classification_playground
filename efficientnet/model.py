# REF
# https://mc.ai/how-to-do-transfer-learning-with-efficientnet/
# https://github.com/Tony607/efficientnet_keras_transfer_learning/blob/master/Keras_efficientnet_transfer_learning.ipynb

from __future__ import absolute_import, division, print_function

import collections 
import math
import numpy as np
import six
from six.moves import xrange
import tensorflow as tf

import tensorflow.keras.backend as K
import tensorflow.keras.models as KM
import tensorflow.keras.layers as KL
from tensorflow.keras.utils import get_file
from tensorflow.keras.initializers import Initializer
from .layers import Swish, DropConnect
from .params import get_model_params, IMAGENET_WEIGHTS
from .initializers import conv_kernel_initializer, dense_kernel_initializer

__all__ = ['EfficientNet', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
        'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7'
]

class ConvKernelInitializer(Initializer):
    def __call__(self, shape, dtype=K.floatx(), partition_info=None):
        del partition_info
        kernel_height, kernel_width, _, out_filters = shape
        fan_out = int(kernel_height * kernel_width * out_filters)
        return tf.random_uniform(
            shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype
        )
    
class DenseKernelInitializer(Initializer):
    def __call__(self, shape, dtype=K.floatx(), partition_info=None):
        del partition_info
        init_range = 1.0 / np.sqrt(shape[1])
        return tf.random_uniform(
            shape, -init_range, init_range, dtype=dtype
        )

def round_filters(filters, global_params):
    orig_f = filters
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters
    
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, global_params):
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))
    
def SEBlock(block_args, global_params):
    num_reduced_filters = max(
        1, int(block_args.input_filters * block_args.se_ratio))
    if global_params.data_format == 'channel_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]
    
    def block(inputs):
        x = inputs
        x = KL.Lambda(lambda a: K.mean(a, axis=spatial_dims, keepdims=True))(x)
        x = KL.Conv2D(
            num_reduced_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=ConvKernelInitializer(),
            padding='same',
            use_bias=True
        )(x)
        x = Swish()(x)
        x = KL.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=ConvKernelInitializer(),
            padding='same',
            use_bias=True
        )(x)
        x = KL.Activation('sigmoid')(x)
        out = KL.Multiply()([x, inputs])
        return out
    
    return block

def MBConvBlock(block_args, global_params, drop_connect_rate=None):
    batch_norm_momentum = global_params.batch_norm_momentum
    batch_norm_epsilon = global_params.batch_norm_epsilon
    
    if global_params.data_format == 'channel_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]
    
    has_se = (block_args.se_ratio is not None) and (
        (block_args.se_ratio > 0) and (block_args.se_ratio <= 1)
    )
    
    filters = block_args.input_filters * block_args.expand_ratio
    kernel_size = block_args.kernel_size

    def block(inputs):
        if block_args.expand_ratio != 1:
            x = KL.Conv2D(
                filters, 
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=ConvKernelInitializer(),
                padding='same',
                use_bias=False
            )(inputs)
            x = KL.BatchNormalization(
                axis=channel_axis,
                momentum=batch_norm_momentum,
                epsilon=batch_norm_epsilon
            )(x)
            x = Swish()
        else:
            x = inputs
        
        x = KL.DepthwiseConv2D(
            [kernel_size, kernel_size],
            strides=block_args.strides,
            depthwise_initializer=ConvKernelInitializer(),
            padding='same',
            use_bias=False
        )(x)
        x = KL.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon
        )(x)
        x = Swish()(x)
        
        if has_se:
            x = SEBlock(block_args, global_params)(x)

        x = KL.Conv2D(
            filters, 
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=ConvKernelInitializer(),
            padding='same',
            use_bias=False
        )(x)
        x = KL.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon
        )(x)

        if block_args.id_skip:
            if all(
                s == 1 for s in block_args.strides
            ) and block_args.input_filters == block_args.output_filters:
                if drop_connect_rate:
                    x = DropConnect(drop_connect_rate)(x)
                x = KL.Add()([x, inputs])
        return x

    return block

def EfficientNet(input_shape, block_args_list, global_params, include_top=True, pooling=None):
    batch_norm_momentum = global_params.batch_norm_momentum
    batch_norm_epsilon = global_params.batch_norm_epsilon
    if global_params.data_format == 'channel_first':
        channel_axis = 1
    else:
        channel_axis = -1

    inputs = KL.Input(shape=input_shape)
    x = KL.Conv2D(
        filters=round_filters(32, global_params),
        kernel_size=[3, 3],
        strides=[2, 2],
        kernel_initializer=ConvKernelInitializer(),
        padding='same',
        use_bias=False
    )(inputs)
    x = KL.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon
    )(x)
    x = Swish()(x)

    block_idx = 1
    n_blocks = sum([block_args.num_repeat for block_args in block_args_list])
    drop_rate = global_params.drop_connect_rate or 0
    drop_rate_dx = drop_rate / n_blocks

    for block_args in block_args_list:
        assert block_args.num_repeat > 0
        block_args = block_args._replace(
            input_filters=round_filters(block_args.input_filters, global_params)
            output_filters=round_filters(block_args.output_filters, global_params)
            num_repeat=round_filters(block_args.num_repeat, global_params)
        )
        

