# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 09:03:46 2025

@author: TAKO
"""

#!/usr/local/bin/python3
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# hyperparameter
learning_rate = 0.01 # standard: bis .00001
training_epochs = 15 # 60% accuracy in a few seconds
# training_epochs = 20 #  Anzahl Trainingsdurchläufe 90% accuracy in 1 minute not bad
batch_size = 60000    # Anzahl der Trainingsdaten

# MNIST dataset
mnist_train = dsets.MNIST(root='../data/MNIST/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='../data/MNIST/', train=False, transform=transforms.ToTensor(), download=True)
data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

# minimal 'Perceptron':
# just a linear model of matrix multiplication!
# vector of 784 = 28 * 28 pixels (rows * columns)
# and 10 classes representing digits 0-9
# y = M*x + bias

# Dense Layer, Perceptron, Fully connected layer (linear + activation function) in pytorch extra
model = torch.nn.Linear( 28 * 28 , 10, bias=True) # most trivial

# define optimizer for cost/loss ≈ error ≈ distance
# loss = torch.nn.MSELoss()  # (activation - target) ** 2
loss = torch.nn.CrossEntropyLoss()    # Classification! Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def accuracy():
	# Test the model using test sets
	with torch.no_grad(): # no need to calculate gradient, just evaluate
		X_test = mnist_test.data.view(-1, 28 * 28).float()
		prediction = model(X_test)
		correct_prediction = torch.argmax(prediction, 1) == mnist_test.targets
		print('Accuracy:', float(correct_prediction.sum()) / len(prediction))

# def train() # so wie tensorflow … als wrapper für expliziten Ansatz
# …
for epoch in range(training_epochs):
	accuracy()
	for X, Y in data_loader: #
		optimizer.zero_grad() # reset gradients to zero
		Q = model(X.view(-1, 28 * 28)) # reshape input image into [batch_size by 784]
		# hypothesis = activation / prediction
		# Y = target label
		cost = loss(Q, Y) # Verlust-Funktion error
		cost.backward() # backpropagation, ziehe Gradienten von Gewichten ab
		optimizer.step() # gehe Gradienten in Richtung des Minimums

	print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(cost), flush=True)


# Visualisierung eines MNIST Bildes
import matplotlib.pyplot as plt
idx = 0 # random.randint(0, len(mnist_test)-1)
img = mnist_test.data[idx].numpy()
plt.imshow(img, cmap='gray')
plt.show()
weights = model.state_dict()['weight'].cpu().numpy()
weights = weights.reshape(10, 28, 28)