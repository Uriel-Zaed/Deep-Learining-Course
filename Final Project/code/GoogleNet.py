import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR
from sklearn.metrics import classification_report
import csv

from GoogleNet_Dependencies import *

batch_size=128
num_of_epochs=100
Learning_rate = 0.1

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size, shuffle=False, num_workers=2)

# Define GoogLeNet model
model = googlenet(num_classes=10, init_weights=True).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=Learning_rate, momentum=0.9)


losses = []
learning_rates = []

# Training loop
for epoch in range(num_of_epochs):
    running_loss = 0.0

    for data in trainloader:
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs, _, _ = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()

        losses.append(running_loss)
        learning_rates.append(Learning_rate)

    print(f'Epoch: {epoch+1}, Loss: {running_loss/100:.3f}')
    running_loss = 0.0


# Save losses and learning rate values to a CSV file
csv_data = list(zip(losses, learning_rates))
csv_headers = ['Loss', 'Learning Rate']

with open('losses_and_lr.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_headers)
    writer.writerows(csv_data)

# Evaluate the model
correct = 0
total = 0
predicted_labels = []
true_labels = []
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs, _, _ = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

# Calculate Precision, Recall, Specificity, and F1-Score
report = classification_report(true_labels, predicted_labels, target_names=testset.classes)
print(report)

