#!/usr/bin/env python
# Created by "Thieu" at 09:41, 18/01/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import tensorflow as tf
import math


def expand_chebyshev(x, n_funcs):
    x1 = x
    x2 = 2 * tf.pow(x, 2) - 1
    x3 = 4 * tf.pow(x, 3) - 3 * x
    x4 = 8 * tf.pow(x, 4) - 8 * tf.pow(x, 2) + 1
    x5 = 16 * tf.pow(x, 5) - 20 * tf.pow(x, 3) + 5 * x
    my_list = [x1, x2, x3, x4, x5]
    return tf.concat(my_list[:n_funcs], axis=1)


def expand_legendre(x, n_funcs):
    x1 = x
    x2 = 1 / 2 * (3 * tf.pow(x, 2) - 1)
    x3 = 1 / 2 * (5 * tf.pow(x, 3) - 3 * x)
    x4 = 1 / 8 * (35 * tf.pow(x, 4) - 30 * tf.pow(x, 2) + 3)
    x5 = 1 / 40 * (9 * tf.pow(x, 5) - 350 * tf.pow(x, 3) + 75 * x)
    my_list = [x1, x2, x3, x4, x5]
    return tf.concat(my_list[:n_funcs], axis=1)


def expand_laguerre(x, n_funcs):
    x1 = -x + 1
    x2 = 1 / 2 * (tf.pow(x, 2) - 4 * x + 2)
    x3 = 1 / 6 * (-tf.pow(x, 3) + 9 * tf.pow(x, 2) - 18 * x + 6)
    x4 = 1 / 24 * (tf.pow(x, 4) - 16 * tf.pow(x, 3) + 72 * tf.pow(x, 2) - 96 * x + 24)
    x5 = 1 / 120 * (-tf.pow(x, 5) + 25 * tf.pow(x, 4) - 200 * tf.pow(x, 3) + 600 * tf.pow(x, 2) - 600 * x + 120)
    my_list = [x1, x2, x3, x4, x5]
    return tf.concat(my_list[:n_funcs], axis=1)


def expand_power(x, n_funcs):
    x1 = x
    x2 = x1 + tf.pow(x, 2)
    x3 = x2 + tf.pow(x, 3)
    x4 = x3 + tf.pow(x, 4)
    x5 = x4 + tf.pow(x, 5)
    my_list = [x1, x2, x3, x4, x5]
    return tf.concat(my_list[:n_funcs], axis=1)


def expand_trigonometric(x, n_funcs):
    x1 = x
    x2 = tf.sin(tf.constant(math.pi) * x) + tf.cos(tf.constant(math.pi) * x)
    x3 = tf.sin(2 * tf.constant(math.pi) * x) + tf.cos(2 * tf.constant(math.pi) * x)
    x4 = tf.sin(3 * tf.constant(math.pi) * x) + tf.cos(3 * tf.constant(math.pi) * x)
    x5 = tf.sin(4 * tf.constant(math.pi) * x) + tf.cos(4 * tf.constant(math.pi) * x)
    my_list = [x1, x2, x3, x4, x5]
    return tf.concat(my_list[:n_funcs], axis=1)

