################################################################################
#
# BGU IEM Introduction to Deep Learning Course | 2023b
#
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os

from numpy.distutils.system_info import xft_info
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
import matplotlib.pyplot as plt

import torch


def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        accuracy 

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    total_correct = 0
    total_samples = 0

    for inputs, labels in data_loader:
        inputs = np.array(inputs)
        labels = np.array(labels)
        logits = model.forward(inputs.reshape(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3]))
        predicted_labels = np.argmax(logits, axis=1)
        correct = np.sum(predicted_labels == labels)
        total_correct += correct
        total_samples += len(labels)

    accuracy = total_correct / total_samples

    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    print(
        'hidden dims - {}, learning rate - {}, batch size - {}, epochs - {}, seed - {}'
        .format(hidden_dims, lr, batch_size, epochs, seed))

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Initialize model and loss module
    model = MLP(n_inputs=3072, n_hidden=hidden_dims, n_classes=10)

    loss_module = CrossEntropyModule()

    # Track the best validation accuracy and the corresponding model
    best_val_accuracy = 0.0
    best_model = None

    # TODO: Training loop including validation
    val_accuracies = []
    val_losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} / {epochs}")

        train_losses = []

        for inputs, labels in tqdm(cifar10_loader['train']):
            # Forward pass
            inputs = inputs.reshape([batch_size, -1])
            outputs = model.forward(inputs)
            loss = loss_module.forward(outputs, labels)
            d_loss = loss_module.backward(outputs, labels)
            model.backward(d_loss)
            model.update_weights(lr)
            train_losses.append(np.nan_to_num(loss))

        avg_train_loss = np.mean(train_losses)
        print(f"Train Loss: {avg_train_loss}")

        # Validation phase
        val_accuracy = evaluate_model(model, cifar10_loader['validation'])

        print(f"Validation Accuracy: {val_accuracy}")

        # Keep track of validation accuracies for plotting
        val_accuracies.append([val_accuracy])
        val_losses.append([avg_train_loss])

        # Check if current model has the best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Use deepcopy to create a new copy of the model (if necessary)
            best_model = deepcopy(model)

    # TODO: Test best model
    test_accuracy = evaluate_model(best_model, cifar10_loader['test'])

    # TODO: Add any information you might want to save for plotting
    logging_info = {'accuracy': val_accuracies, 'losses': val_losses}

    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_info


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    # plot Q3

    model, val_accuracies, test_accuracy, logging_info = train(**kwargs)

    X = np.array(range(1, args.epochs+1))

    plt.plot(X, np.array(logging_info['losses']), marker='o', linestyle='-', color='y')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.title('Plot of Loss vs Epoch')
    plt.xticks(np.arange(min(X), max(X) + 1, 1))
    plt.show()

    print("")
    print("Test Accuracy: " + str(test_accuracy))



