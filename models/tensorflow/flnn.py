#!/usr/bin/env python
# Created by "Thieu" at 09:41, 18/01/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

"""
The implementation of Functional Linked Neural Network
"""

from keras import Model
from keras.layers import Dense, Layer
from models.tensorflow.utils import math_util


#################################### Functional Style ########################################




#################################### OOP Style ###############################################

class ExpandLayer(Layer):
    def __init__(self, expand_name, n_funcs):
        super(ExpandLayer, self).__init__()
        self.expand_name = expand_name
        self.expand_func = getattr(math_util, f"expand_{self.expand_name}")
        self.n_funcs = n_funcs

    def call(self, inputs, **kwargs):
        data = self.expand_func(inputs, self.n_funcs)
        return data


class FLNN(Model):
    def __init__(self, expand_name, n_funcs, act):
        super(FLNN, self).__init__()
        self.dense1 = ExpandLayer(expand_name, n_funcs)
        self.dense2 = Dense(1, activation=act)

    def call(self, input_tensor, training=False, **kwargs):
        x = self.dense1(input_tensor)
        return self.dense2(x)
