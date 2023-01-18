#!/usr/bin/env python
# Created by "Thieu" at 09:42, 18/01/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import tensorflow as tf
import numpy as np


class Anfis(tf.keras.Model):
    def __init__(self, n_inputs, m_rules, **kwargs):
        super(Anfis, self).__init__(**kwargs)
        self.n_inputs = n_inputs
        self.m_rules = m_rules
        self.mus = tf.Variable(shape=(self.m_rules * self.n_inputs), dtype=np.float32,
                               initial_value=tf.random.uniform([self.m_rules * self.n_inputs], -1.0, 1.0, seed=1))
        self.sigmas = tf.Variable(shape=(self.m_rules * self.n_inputs), dtype=np.float32,
                                  initial_value=tf.random.uniform([self.m_rules * self.n_inputs], -1.0, 1.0, seed=1))
        self.fxs = tf.Variable(shape=(1, self.m_rules), dtype=np.float32,
                               initial_value=tf.random.uniform([1, self.m_rules], -1.0, 1.0, seed=1))

    # def get_config(self):
    #     config = super(Anfis, self).get_config()
    #     config.update({"n_inputs": self.n_inputs, "m_rules": self.m_rules})
    #     return config

    def get_config(self):
        return {
            "n_inputs": self.n_inputs, "m_rules": self.m_rules, "weights": self.get_weights()
        }

    def get_membership_information(self):
        return {
            "mus": self.mus.numpy(),
            "sigmas": self.sigmas.numpy(),
            "y_output": self.fxs.numpy(),
            "m_rules": self.m_rules,
            "n_inputs": self.n_inputs
        }

    def call(self, args, **kwargs):
        t2 = tf.subtract(tf.tile(args, (1, self.m_rules)), self.mus)
        fuzzy_layer = tf.exp(-0.5 * tf.square(t2) / tf.square(self.sigmas))
        prod_layer = tf.reduce_prod(tf.reshape(fuzzy_layer, (-1, self.m_rules, self.n_inputs)), axis=2)
        normalized_nodes = tf.clip_by_value(tf.reduce_sum(prod_layer, axis=1), 1e-12, 1e12)
        defuzzy = tf.reduce_sum(tf.multiply(prod_layer, self.fxs), axis=1)
        output = tf.divide(defuzzy, normalized_nodes)
        return output


