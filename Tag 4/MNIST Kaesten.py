# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 09:37:58 2025

@author: TAKO
"""

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# --- Hyperparameter ---
learning_rate = 0.01
training_epochs = 15
batch_size = 64

# --- Device wählen (GPU wenn verfügbar) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MNIST Dataset ---
mnist_train = dsets.MNIST(root='../data/MNIST/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='../data/MNIST/', train=False, transform=transforms.ToTensor(), download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)

# --- Model ---
model = torch.nn.Linear(28 * 28, 10).to(device)

# --- Loss & Optimizer ---
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# --- Accuracy Funktion ---
def accuracy():
    with torch.no_grad():
        X_test = mnist_test.data.view(-1, 28 * 28).float().to(device)
        Y_test = mnist_test.targets.to(device)
        preds = model(X_test)
        correct = (torch.argmax(preds, 1) == Y_test).float().sum()
        acc = correct / len(Y_test)
        print(f"Accuracy: {acc:.2%}")

# --- Training ---
for epoch in range(training_epochs):
    model.train()
    for X, Y in data_loader:
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, Y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{training_epochs}, Loss: {loss.item():.4f}")
    accuracy()

# --- Gewichte visualisieren ---
weights = model.weight.detach().cpu().numpy().reshape(10, 28, 28)

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(weights[i], cmap='gray')
    plt.title(f"Digit {i}")
    plt.axis('off')
plt.show()
