################################################################################
#
# BGU IEM Introduction to Deep Learning Course | 2023b
#
################################################################################

"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logsistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = []

        if len(n_hidden) > 0:
            linear_layer = LinearModule(n_inputs, n_hidden[0], input_layer=True)
            self.layers.append(linear_layer)
            activation_layer = RELUModule()
            self.layers.append(activation_layer)
            last_dim = n_hidden[0]
            for h_layer in n_hidden:
                linear_layer = LinearModule(last_dim, h_layer, input_layer=False)
                self.layers.append(linear_layer)
                last_dim = h_layer
                activation_layer = RELUModule()
                self.layers.append(activation_layer)

            output_layer = LinearModule(h_layer, n_classes, input_layer=False)
            self.layers.append(output_layer)
            activation_layer = SoftMaxModule()
            self.layers.append(activation_layer)

        else:
            # logistic regression
            output_layer = LinearModule(n_inputs, n_classes, input_layer=True)
            self.layers.append(output_layer)
            activation_layer = SoftMaxModule()
            self.layers.append(activation_layer)
        pass

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:s
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        out = x
        for l in self.layers:
            out = l.forward(out)

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss with respec to the network output

        TODO:
        Implement backward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        step_back = dout

        for l in self.layers[::-1]:
            step_back = l.backward(step_back)

        pass
        #######################
        # END OF YOUR CODE    #
        #######################

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Iterate over modules and call the 'clear_cache' function.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        for l in self.layers:
            l.clear_cache()
        pass

        #######################
        # END OF YOUR CODE    #
        #######################

    def update_weights(self, lr):
        # Update weights and biases
        for l in self.layers:
            if isinstance(l, LinearModule):
                l.weights -= lr * l.grads['weights']
                l.biases -= lr * l.grads['biases']
