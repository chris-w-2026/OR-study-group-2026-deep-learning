import numpy as np

#manually define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#set up a test array
x = np.array([1,0])

#manually assign the weights
weights = np.array([2.0, -1.0])

#manually assign the balance
bias = 0.5

#calculate the prediction
z = np.dot(x, weights) + bias

#apply the sigmoid function (prevents small changes to weights and bias having profound effect on final prediction)
output = sigmoid(z)

print(output)