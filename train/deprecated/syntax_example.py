from tracemalloc import start

import numpy as np
import json
import pprint as pp
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidPrime(x):
    return x * (1 - x)

def delta(t,
          y,
          W2):
    """
    Args:
        t: ith output layer value
        y: ith target value
        W2: Weights for the output layer
    """

    # Need to figure out how to calculate the delta gradient, d, for the output layer
    d = 0
    u = 0
    for k in range(len(W2)):
        u += W2[k]*t
    d = -2*(y-t)*sigmoidPrime(u)
    return d

def sigma(D, x, y, W1, W2):
    """
    Args:
        D: delta for the output layer
        x: ith input value
        y: ith target value
        W1: Weights for the hidden layer
        W2: Weights for the output layer
    """
    # Need to figure out how to calculate the sigma gradient, s, for the hidden layer
    S = np.zeros(len(W1[0]))
    t = 0
    for j in range(len(W1[0])):
        for k in range(len(W1[1])):
            t += W1[j][k]*x[j]
    t = W1[]
    for j in range(len(S)):
        s = 0
        for l in range(len(W2)):
            s += W2[l]
        s += D * s
        s *= sigmoidPrime(x)
    return s

# A function to calculate the hidden perceptron layer values, z_m 
def get_hidden(W1,
               b1,
               X,
               Z,
               M):
    """
    Args: 
        W1: Weights for the hidden layer
        b1: Biases for the hidden layer
        X: Input data
        Z: Hidden layer values
        M: Number of hidden neurons
    """
    # Need to figure out how to get the hidden layer values, z_m
    for m in range(M):
        if m == 0:
            for i in range(X.shape[0]):
                for k in range(W1.shape[1]):
                    Z[k] = b1[k]
                    for j in range(X.shape[1]):
                        Z[k] += X[i][j]*W1[j][k]
                Z[k] = sigmoid(Z[k])
        # add case of more than one hidden layer (M > 1)
    return 1

def get_output(W2, Z, K, M):
    # Need to figure out how to get the output layer values, y_i
    Y = np.zeros((Z.shape[0], K))
    for i in range(Y.shape[0]):
        for k in range(K):
            Y[i][k] = 0
            for m in range(M):
                Y[i][k] += Z[i][m]*W2[m][k] # Modify weights array to contain submatrices, W_i, for each layer

    return 1

def get_backprop(W1, b1, W2, b2, D, Z, S, X, lr):
    # Need to figure out how to implement the backpropagation function
    W1 -= lr * np.dot(X.T, S)
    b1 -= lr * np.sum(S, axis=0, keepdims=True)
    W2 -= lr * np.dot(Z.T, D)
    b2 -= lr * np.sum(D, axis=0, keepdims=True)

    return 1

def make_sample_batch(X, Y, batch_size):
    # Need to figure out how to implement the make_batch function
    batch_X = []
    batch_Y = []
    for i in range(batch_size):
        randomIndex = np.random.randint(0, X.shape[0] - 1)
        batch_X.append(X[randomIndex][:])
        batch_Y.append(Y[randomIndex][:])
    return np.array(batch_X), np.array(batch_Y)

def epoch(M, X, Y, Z, W_hidden, b_hidden, W_output, b_output, lr, totalLoss, numSteps, batch_size):
    # Need to figure out how to implement the epoch function

    loss_list = np.array(np.zeros(numSteps))

    for step in range(numSteps):

        batch_X, batch_Y = make_sample_batch(X, Y, batch_size)

        # Forward Pass:
        Z = get_hidden(W_hidden, b_hidden, batch_X, Z, M)
        T = get_output(W_output, Z, batch_Y.shape[1], M)

        # Backward Pass:
        delta = delta(T, batch_Y, W_output)
        sigma = sigma(delta, batch_X, batch_Y, W_hidden, W_output)

        W_output -= lr * np.dot(Z.T, delta)
        b_output -= lr * np.sum(delta, axis=0, keepdims=True)
        W_hidden -= lr * np.dot(batch_X.T, sigma)
        b_hidden -= lr * np.sum(sigma, axis=0, keepdims=True)

        # Prediction:
        Z = get_hidden(W_hidden, b_hidden, batch_X, Z, M)
        Y_pred = get_output(W_output, Z, batch_Y.shape[1], M)

        # Loss:
        loss = np.mean((Y_pred - batch_Y) ** 2)
        totalLoss += loss
        loss_list[step] = loss

    # visualize loss over epochs:
    plt.plot(loss_list)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

    return 1

def main():
    # Need to figure out how to implement the main function
    trainName = input("Enter the name of the training data file (with extension): ")
    netName = input("Enter the name of the neural network: ")
    weightFilename = netName + ".json"
    dimInputs = int(input("Enter the number of input dimensions: "))
    dimOutputs = int(input("Enter the number of output dimensions: "))
    numHiddenNeurons = int(input("Enter the number of hidden neurons: "))
    numEpochs = int(input("Enter the number of epochs: "))      
    numBatches = int(input("Enter the number of batches: "))
    batch_size = int(input("Enter the batch size: "))
    numSteps = np.floor(numEpochs / numBatches)
    totalLoss = 0.0
    with open(weightFilename, "w", encoding="utf-8") as f:
        json.dump({"dimInputs": dimInputs, "dimOutputs": dimOutputs, "numHiddenNeurons": numHiddenNeurons, "numEpochs": numEpochs}, f, indent=4)
    with open(trainName, "r", encoding="utf-8") as f:
        trainingData = json.load(f)
    X = trainingData['INPUTS']
    Y = trainingData['TARGETS']
    Z = np.array(np.zeros((numHiddenNeurons, 1)))
    W_hidden = np.random.rand(dimInputs, numHiddenNeurons)
    b_hidden = np.random.rand(numHiddenNeurons, 1)
    W_output = np.random.rand(numHiddenNeurons, 1)
    b_output = 0.0
    lr = 1.0
    for i in range(numEpochs):
        epoch(numHiddenNeurons, X, Y, Z, W_hidden, b_hidden, W_output, b_output, lr, totalLoss, numSteps, batch_size)


if __name__ == "__main__":
    main()

