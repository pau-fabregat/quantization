# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torch.nn as nn


class Swish(nn.Module):
    """
    Wrapper for the Swish activation function.
    """
    def __init__(self):
        super().__init__()
        self.ff = torch.ao.nn.quantized.FloatFunctional()

    def forward(self, x):
        y = self.ff.mul(x, torch.sigmoid(x))
        return y


class SwishFunction(torch.autograd.Function):
    """
    Implementation of the Swish activation function: x * sigmoid(x).

    Searching for activation functions. Ramachandran, Prajit and Zoph, Barret
    and Le, Quoc V. 2017
    """

    @staticmethod
    def forward(ctx, x, y):
        #result = x * torch.sigmoid(x)
        #result = self.bypass(x)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))


class Bypass(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = torch.ao.nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.ff.mul(x, torch.sigmoid(x))