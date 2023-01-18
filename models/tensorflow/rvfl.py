#!/usr/bin/env python
# Created by "Thieu" at 09:42, 18/01/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

"""
The implementation of Random Vector Functional Link (RVFL) network
"""


import tensorflow as tf
from keras import Input, Model
from keras.layers import Dense, Add


#################################### Functional Style ########################################

def create_RVFL_network(size_input, size_hidden, activation, initializer):
    input = Input(shape=(size_input,))
    hidden = Dense(size_hidden, activation=activation, kernel_initializer=initializer, bias_initializer=initializer, trainable=False)(input)
    output1 = Dense(1, activation=None)(hidden)
    output2 = Dense(1, activation=None)(input)
    output = Add()([output1, output2])
    return Model(input, output)




#################################### OOP Style ###############################################

class RVFL(Model):
    def __init__(self, n_units, act, initializer):
        super(RVFL, self).__init__()
        self.dense1 = Dense(n_units, activation=act, kernel_initializer=initializer, bias_initializer=initializer, trainable=False)
        self.output1 = Dense(1, activation=None)
        self.output2 = Dense(1, activation=None)

    def call(self, input_tensor, training=False, **kwargs):
        x = Input(input_tensor)
        hidden = self.dense1(x)
        output1 = self.output1(hidden)
        output2 = self.output2(x)
        return tf.add(output1, output2)


