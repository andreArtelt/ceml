# -*- coding: utf-8 -*-
import tensorflow as tf


def min_of_list(x):
    return tf.reduce_min(x)

def loglikelihood(y_pred, y_target):
    return tf.math.log(y_pred)[y_target]

def negloglikelihood(y_pred, y_target):
    return -1. * loglikelihood(y_pred, y_target)

def l1(x, x_orig):
    return tf.reduce_sum(tf.abs(x - x_orig))

def l2(x, x_orig):
    return tf.reduce_sum(tf.pow(x - x_orig, 2))

def lmad(x, x_orig, mad):
    return tf.reduce_sum(tf.divide(tf.abs(x_orig - x), mad))
