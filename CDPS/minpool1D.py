"""
minpool.py
This module implements multivariate minpooling
inspired from:
https://github.com/reachtarunhere/pytorch-snippets/blob/master/min_pool1d.py

Raises:
    ValueError: [description]

Returns:
    [type]: [description]
"""
import torch
import torch.nn as nn


class MinPool1d(nn.Module):

    def __init__(self, kernel_size=3, type_='DEP'):
        super(MinPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.type_ = type_

    def forward(self, x_input):
        """
        minpooling, either dependent or independent of time axis
        Args:
            x_input (tensor): input batch to perform minpoolin on
        """
        _, d, _, _ = [x_input.size(i) for i in range(4)]
        if self.type_ == 'INDEP':
            x_input = torch.stack([x_input[:, i, :, :].min(dim=2)[0]
                                  for i in range(d)])
            return x_input.transpose(1, 0).sum(1)
        elif self.type_ == "DEP":
            x_input = torch.sum(x_input, 1)
            return x_input.min(dim=2)[0]
        else:
            raise ValueError("Unsupported value for 'type_'. "
                             "Choose either 'INDEP' or 'DEP'.")
