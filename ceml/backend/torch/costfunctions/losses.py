# -*- coding: utf-8 -*-
import torch


def min_of_list(x):
    return torch.min(x)

def loglikelihood(y_pred, y_target):
    return torch.log(y_pred)[y_target]

def negloglikelihood(y_pred, y_target):
    return -1. * loglikelihood(y_pred, y_target)

def l1(x, x_orig):
    return torch.sum(torch.abs(x - x_orig))

def l2(x, x_orig):
    return torch.sum(torch.pow(x - x_orig, 2))

def lmad(x, x_orig, mad):
    return torch.sum(torch.div(torch.abs(x_orig - x), mad))
