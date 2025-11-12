#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: DJ
@file: lenet5.py
@time: 2021/05/09
@desc:
"""

import torch
from torch import nn
import torch.nn.functional as F


class Lenet5(nn.Module):
    """
    for cifa-10 dataset
    """

    def __init__(self):
        super(Lenet5, self).__init__()

        self.model = nn.Sequential(
            # x: [b, 3, 32, 32] => [b, 6, ]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Sigmoid(),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        """

        :param x: [b, 3, 32, 32]
        :return:
        """
        out = self.model(x)
        return out


def main():
    """
    test to check the input_channel required for first Linear layer
    :return:
    """
    net = Lenet5()
    # mimicking data point and error will occur
    # the error will tell the correct in_chnl for the first linear layer
    temp = torch.randn(2, 3, 32, 32)
    out = net(temp)
    print(out.shape)


if __name__ == '__main__':
    main()
