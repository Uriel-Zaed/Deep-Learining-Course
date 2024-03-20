################################################################################
#
# BGU IEM Introduction to Deep Learning Course | 2023b
#
################################################################################
"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, use_batch_norm=False):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
          use_batch_norm: If True, add a Batch-Normalization layer in between
                          each Linear and ELU layer.

        TODO:
        Implement module setup of the network.
        The linear layer have to initialized according to the Kaiming initialization.
        Add the Batch-Normalization _only_ if use_batch_norm is True.
        
        Hint: No softmax layer is needed here. Look at the CrossEntropyLoss module for loss calculation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        super(MLP, self).__init__()

        layers = OrderedDict()

        # Add input layer
        if n_hidden:
            layers['input'] = nn.Linear(n_inputs, n_hidden[0])
            if use_batch_norm:
                layers['batch_norm_input'] = nn.BatchNorm1d(n_hidden[0])
            layers['activation_input'] = nn.ReLU(inplace=True)
        else:
            layers['input'] = nn.Linear(n_inputs, n_classes)

        # Add hidden layers
        if len(n_hidden) > 1:
            for i in range(len(n_hidden) - 1):
                layers['hidden_{}'.format(i)] = nn.Linear(n_hidden[i], n_hidden[i + 1])
                if use_batch_norm:
                    layers['batch_norm_{}'.format(i)] = nn.BatchNorm1d(n_hidden[i + 1])
                layers['activation_{}'.format(i)] = nn.ReLU(inplace=True)

        # Add output layer
        if n_hidden:
            layers['output'] = nn.Linear(n_hidden[-1], n_classes)
        else:
            layers['output'] = nn.Linear(n_inputs, n_classes)

        self.layers = nn.Sequential(layers)

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        out = self.layers(x)

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device
    
