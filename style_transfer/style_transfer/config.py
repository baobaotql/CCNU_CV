# -*- coding: utf-8 -*-

import numpy as np


class CONFIG:
    num_labels = 4
    IMAGE_WIDTH = 400
    IMAGE_HEIGHT = 300
    COLOR_CHANNELS = 3
    NOISE_RATIO = 0.6
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) #颜色通道均值
    VGG_MODEL = 'pretrained-model/imagenet-vgg-verydeep-19.mat'
    alpha=10
    beta=40
    num_iterations_1 = 50
    num_iterations_2 = 20
    learning_rate=2.0
    STYLE_LAYERS = [
        ('conv1_1', 0.2),
        ('conv2_1', 0.2),
        ('conv3_1', 0.2),
        ('conv4_1', 0.2),
        ('conv5_1', 0.2)]
    STYLE_LAYERS_2 = [
        ('block1_conv1', 0.2),
        ('block2_conv1', 0.2),
        ('block3_conv1', 0.2),
        ('block4_conv1', 0.2),
        ('block5_conv1', 0.2)]
		