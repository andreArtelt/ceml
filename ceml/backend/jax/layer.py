# -*- coding: utf-8 -*-
import jax
from jax.config import config; config.update("jax_enable_x64", True)    # Use 64 bit floating point numbers
import jax.numpy as npx


def create_tensor(x):
    return npx.array(x, dtype=npx.float64)

def affine(x, w, b):
    return npx.dot(w, x) + b

def softmax(x):
    return npx.exp(x) / npx.sum(npx.exp(x), axis=0)

def normal_distribution(x, mean, variance):
    return npx.exp(-.5 * npx.square(x - mean) / variance) / npx.sqrt(2. * npx.pi * variance)

def log_normal_distribution(x, mean, variance):
    return -.5 * npx.square(x - mean) / variance - .5 * (2. + npx.pi + variance)

def log_multivariate_normal(x, mean, sigma_inv, k):
    return .5 * npx.log(npx.linalg.det(sigma_inv)) - .5 * k * npx.log(2. * npx.pi) - .5 * npx.dot(x - mean, npx.dot(sigma_inv, (x - mean)))
