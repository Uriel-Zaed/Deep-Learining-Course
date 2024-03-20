################################################################################
#
# BGU IEM Introduction to Deep Learning Course | 2023b
#
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy

from torch import device
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


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

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)

            outputs = model(inputs.reshape(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3]))
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    #######################
    # END OF YOUR CODE    #
    #######################
    return accuracy


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
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
        'hidden dims - {}, learning rate - {}, batch normalization - {}, batch size - {}, epochs - {}, seed - {}'
        .format(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed))

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Initialize the MLP model
    model = MLP(3072, hidden_dims, 10, use_batch_norm)
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Variables to store best model and accuracy
    best_model = None
    best_accuracy = 0.0

    # Lists to store validation accuracies and test accuracy
    val_accuracies = []
    val_losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} / {epochs}")

        model.train()
        total_loss = 0.0
        i = 0

        for inputs, labels in tqdm(cifar10_loader['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs.reshape(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3]))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            i += 1

        val_losses.append(total_loss / i)

        # Compute validation accuracy
        accuracy = evaluate_model(model, cifar10_loader['validation'])
        val_accuracies.append(accuracy)

        print(
            'Epoch {}/{} - Loss: {:.4f} - Validation Accuracy: {:.4f}'.format(epoch + 1, epochs, total_loss / i,
                                                                              accuracy))

        # Update best model if validation accuracy is improved
        if accuracy > best_accuracy:
            best_model = deepcopy(model)
            best_accuracy = accuracy

    # Evaluate the best model on the test set
    test_accuracy = evaluate_model(best_model, cifar10_loader['test'], num_classes=10)

    logging_info = {'accuracy': val_accuracies, 'losses': val_losses}  # Placeholder for additional logging information

    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_info


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, '
                             'use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')

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

    # Plot Q4a

    model, val_accuracies, test_accuracy, logging_info = train(**kwargs)

    X = np.array(range(1, args.epochs+1))

    plt.plot(X, np.array(logging_info['losses']), marker='o', linestyle='-', color='r')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.title('Plot of Loss vs Epoch')
    plt.xticks(np.arange(min(X), max(X) + 1, 1))
    plt.show()

    print("")
    print("Test Accuracy: " + str(test_accuracy))

    # Plot Q4b iv

    lr_array = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10]
    accuracies = []
    losses_curves = []
    X = np.array(range(1, args.epochs + 1))

    for lr in lr_array:
        kwargs['lr'] = lr
        kwargs['use_batch_norm'] = True
        model, val_accuracies, test_accuracy, logging_info = train(**kwargs)
        accuracies.append(test_accuracy)
        losses_curves.append(logging_info['losses'])

    plt.plot(lr_array, accuracies, marker='o', linestyle='-', color='g')
    plt.xlabel('lr')
    plt.ylabel('accuracy')
    plt.title('Plot of Accuracy vs Learning Rate')
    plt.show()

    i = 1
    for l in range(len(losses_curves)):
        plt.plot(X, losses_curves[l], label='lr ' + str(lr_array[l]))
        i += 1

    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.title('Multiple curves of Losses vs Epochs')
    plt.xticks(np.arange(min(X), max(X) + 1, 1))
    plt.legend()
    plt.show()
