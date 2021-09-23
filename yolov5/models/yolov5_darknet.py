#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLOv5 Darknet Model Defined in Keras."""

import os
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras.utils import get_source_inputs, get_file
from tensorflow.keras.layers import Add, ZeroPadding2D, UpSampling2D, Concatenate
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, AveragePooling2D, GlobalMaxPooling2D, Reshape, \
    Flatten, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from yolov5.models.layers import *


def csp_resblock_body(x, num_filters, num_blocks, depth_multiple, width_multiple):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Swish(make_divisible(num_filters * width_multiple, 8), (3, 3), strides=(2, 2))(x)

    x = bottleneck_csp_block(x, num_filters, num_blocks, depth_multiple, width_multiple, shortcut=True)
    return x


def yolov5_darknet_body(x, depth_multiple, width_multiple):
    '''A modified darknet body for YOLOv5'''
    # x = ZeroPadding2D(((3,0),(3,0)))(x)
    # x = DarknetConv2D_BN_Swish(make_divisible(64*width_multiple, 8), (5,5), strides=(2,2))(x)
    x = focus_block(x, 64, width_multiple, kernel=3)

    x = csp_resblock_body(x, 128, 3, depth_multiple, width_multiple)
    x = csp_resblock_body(x, 256, 9, depth_multiple, width_multiple)
    # f3: 52 x 52 x (256*width_multiple)
    f3 = x

    x = csp_resblock_body(x, 512, 9, depth_multiple, width_multiple)
    # f2: 26 x 26 x (512*width_multiple)
    f2 = x

    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Swish(make_divisible(1024 * width_multiple, 8), (3, 3), strides=(2, 2))(x)
    # different with ultralytics PyTorch version, we will try to leave
    # the SPP & BottleneckCSP block to head part

    # f1 = x: 13 x 13 x (1024*width_multiple)
    return x, f2, f3


def yolov5_body(inputs, num_anchors, num_classes, depth_multiple=1.0, width_multiple=1.0, weights_path=None):
    """Create YOLOv5 model CNN body in Keras."""
    # due to depth_multiple, we need to get feature tensors from darknet
    # body function:
    # f1: 13 x 13 x (1024*width_multiple)
    # f2: 26 x 26 x (512*width_multiple)
    # f3: 52 x 52 x (256*width_multiple)
    f1, f2, f3 = yolov5_darknet_body(inputs, depth_multiple, width_multiple)
    darknet = Model(inputs, f1)

    print('backbone layers number: {}'.format(len(darknet.layers)))
    if weights_path is not None:
        darknet.load_weights(weights_path, by_name=True)
        print('Load weights {}.'.format(weights_path))

    f1_channel_num = int(1024 * width_multiple)
    f2_channel_num = int(512 * width_multiple)
    f3_channel_num = int(256 * width_multiple)

    y1, y2, y3 = yolov5_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors,
                                    num_classes, depth_multiple, width_multiple)

    return Model(inputs, [y1, y2, y3])
