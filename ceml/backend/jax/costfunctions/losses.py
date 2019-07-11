# -*- coding: utf-8 -*-
import jax.numpy as npx


def min_of_list(x):
    return npx.min(x)

def loglikelihood(y_pred, y_target):
    return npx.log(y_pred)[y_target]

def negloglikelihood(y_pred, y_target):
    return -1. * loglikelihood(y_pred, y_target)

def custom_dist(x, x_orig, omega):
    d = npx.array(x - x_orig)
    return npx.dot(d, npx.dot(omega, d))

def l1(x, x_orig):
    return npx.sum(npx.abs(x - x_orig))

def l2(x, x_orig):
    return npx.sum(npx.square(x - x_orig))

def lmad(x, x_orig, mad):
    return npx.sum(npx.divide(npx.abs(x_orig - x), mad))
