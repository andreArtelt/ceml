# -*- coding: utf-8 -*-
import tensorflow as tf


def create_tensor(x):
    return tf.constant(x)


def create_mutable_tensor(x):
    return tf.Variable(x)
