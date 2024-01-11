"""
ShapeletLayer.py

Impelenetation of the Shapelet blocks as explained
in the paper and inspired from the work of LDPS by
Romain Tavenard.
"""

import numpy
import torch
import torch.nn as nn


class ShapeletLayer(nn.Module):
    """
    Shapelet layer.
    [i, 0] == pairs[i, 1]:
    pairs[i, 1] = np.random.choice(Trdata.shape[0])
    Computes sliding window distances between a set of time series and a set
    of shapelets.

    Parameters
    ----------
    in_channels : int
        Number of input channels (modalities of the time series)
    out_channels: int
        Number of output channels (number of shapelets)
    kernel_size: int
        Shapelet length

    Examples
    --------
    >>> time_series = torch.Tensor([[1. ,  2.,  3.,  4.,  5.],
                                    [-4., -5., -6., -7., -8.]]
                                   ).view(2, 1, 5)
    >>> shapelets = torch.Tensor([[1., 2.],
                                  [3., 4.],
                                  [5., 6.]])
    >>> layer = ShapeletLayer(in_channels=1, out_channels=3, kernel_size=2)
    >>> layer.weight.data = shapelets
    >>> dists = layer.forward(time_series)
    >>> dists.shape
    torch.Size([2, 3, 4])
    >>> dists[0]
    tensor([[ 0.,  1.,  4.,  9.],
            [ 4.,  1.,  0.,  1.],
            [16.,  9.,  4.,  1.]], grad_fn=<SelectBackward>)
    """

    def __init__(self, in_channels, out_channels, kernel_size, device, type_):
        super(ShapeletLayer, self).__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.type_ = type_
        self.false_conv_layer = nn.Conv1d(in_channels=1,
                                          out_channels=1,
                                          kernel_size=kernel_size,
                                          groups=1,
                                          bias=False)

        data = torch.Tensor(numpy.eye(kernel_size)).to(device)
        data_view = data.view(kernel_size, 1, kernel_size)
        self.false_conv_layer.weight.data = data_view
        for p in self.false_conv_layer.parameters():
            p.requires_grad = False
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels,
                                                kernel_size).to(device))

    def forward(self, x_input):
        """
        calculate the distance between the shapelet candidates and
        the batch time series.

        Args:
            x_input (tensor): batch of time series

        Returns:
            tensor : distance between all possible subsequence
                    and candidate shapelets
        """
        n, _, d = x_input.shape
        reshaped_x = torch.stack([self.false_conv_layer(
                                x_input[:, i, :].view(n, 1, d))
                                    for i in range(self.in_channels)])
        reshaped_x = torch.transpose(torch.transpose(reshaped_x, 0, 1), 2, 3)
        distances = torch.stack([self.pairwise_distances(
                                reshaped_x[:, i, :, :].contiguous().view(
                                   -1, self.kernel_size), self.weight[:, i, :])
                                        for i in range(self.in_channels)])
        distances = distances.view(self.in_channels, n, -1,
                                   self.out_channels).transpose(1, 0)
        return torch.transpose(distances, 2, 3)

    @classmethod
    def pairwise_distances(cls, x, y):
        """Computes pairwise distances between vectors in x and those in y.

        Computed distances are normalized (i.e. divided) by the dimension of
        the space in which vectors lie.
        Assumes x is 2d (n, d) and y is 2d (l, d) and returns
        a tensor of shape (n, l).

        Parameters
        ----------
        x : Tensor of shape=(n, d)
        y : Tensor of shape=(l, d)
        import time
        Returns
        -------
            A 2d Tensor of shape (n, l)
        """
        len_ = y.size(-1)
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, 0.0, numpy.inf) / len_
