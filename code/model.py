################################################################################
#                                                                              #
# This code is part of the following publication:                              #
#                                                                              #
# F. Kratzert, M. Herrnegger, D. Klotz, S. Hochreiter, G. Klambauer            #
# "Do internals of neural networks make sense in the context of hydrology?"    #
# presented at 2018 Fall Meeting, AGU, Washington D.C., 10-14 Dec.             #
#                                                                              #
# Corresponding author: Frederik Kratzert (f.kratzert(at)gmail.com)            #
#                                                                              #
################################################################################

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        """Initialize Wrapper for LSTM
        """
        super(Model, self).__init__()
        self.lstm = LSTM(input_size=5, hidden_size=10)
        self.fc = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass.

        :param x: Tensor with dynamic input features.
        :return: Tensor containing the model predictions.
        """
        h_n, c_n = self.lstm(x)

        # make prediction from last time step
        out = self.fc(h_n[:, -1, :])
        return out, h_n, c_n


class LSTM(nn.Module):
    """This class implements a single LSTM layer with various options.

    The basic LSTM is defined by the following equations.

    .. math::
        f_t = \sigma(W_{hf} h_{t-1} + W_{xf} x_t + b_f)
        i_t = \sigma(W_{hi} h_{t-1} + W_{xi} x_t + b_i)
        o_t = \sigma(W_{ho} h_{t-1} + W_{xo} x_t + b_o)
        c_t = f_t \odot c_{t-1} + tanh(W_{hc} h_t + W_{xc} x_t + b_c)
        h_t = o_t \odot tanh(c_t)

    """
    def __init__(self, input_size: int, hidden_size: int,
                 batch_first: bool=True):
        """Class implementing a single LSTM layer.

        :param input_size: Number of input features.
        :param hidden_size: Number of hidden units.
        :param batch_first: If True, the input must have the shape
            (batch, seq_len, n_features), else (seq_len, batch, n_features).

        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # create tensors of learnable parameters
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size,
                                                        4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size,
                                                        4 * hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all learnable parameters of the LSTM"""
        nn.init.orthogonal_(self.weight_ih.data)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 4)
        self.weight_hh.data.set_(weight_hh_data)
        nn.init.constant_(self.bias.data, val=0)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Forward pass to the LSTM layer.

        :param x: Tensor containing the input data. If batch_first is set to
            True, x should be of shape (batch, seq_len, n_features), otherwise
            (seq_len, batch, n_features).
        :return: h_n and c_n, two tensors containing the hidden and cell state
            of each time step. The shape of this tensor is also defined by
            the batch_first argument.
        """
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.size()

        # Initialize hidden and cell state to zeros
        h_0 = x.data.new(self.hidden_size).zero_()
        h_0 = h_0.unsqueeze(0).expand(batch_size, *h_0.size())
        c_0 = x.data.new(self.hidden_size).zero_()
        c_0 = c_0.unsqueeze(0).expand(batch_size, *c_0.size())
        h_x = (h_0, c_0)

        # empty lists to temporally store all intermediate hidden/cell states
        h_n, c_n = [], []

        # expand bias vectors to batch size
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))

        # perform forward steps over input sequence
        for t in range(seq_len):
            h_0, c_0 = h_x

            # calculate gates depending on bias and peephole option
            gates = (torch.addmm(bias_batch, h_0, self.weight_hh) +
                     torch.mm(x[t], self.weight_ih))

            f, i, o, g = gates.chunk(4, 1)
            c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)

            # store intermediate hidden/cell state in list
            h_n.append(h_1)
            c_n.append(c_1)

            h_x = (h_1, c_1)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        if self.batch_first:
            h_n = h_n.transpose(0, 1)
            c_n = c_n.transpose(0, 1)

        return h_n, c_n