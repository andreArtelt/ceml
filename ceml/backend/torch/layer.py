# -*- coding: utf-8 -*-
import torch


def create_tensor(x, device=torch.device('cpu')):
    return torch.Tensor(x, device=device)
