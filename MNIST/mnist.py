import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt

data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.

_, m_train = X_train.shape

def init_params(): 
    W1 = np.random.randn(64, 784) * np.sqrt(2. / 784)
    b1 = np.zeros((64, 1))
    
    W2 = np.random.randn(10, 64) * np.sqrt(2. / 64)
    b2 = np.zeros((10, 1))
    
    return W1, b1, W2, b2


def ReLu(Z): 
    return np.maximum(0, Z)

def softMax(Z): 
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # For numerical stability
    return expZ / np.sum(expZ, axis=0, keepdims=True)
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLu(Z1)
    
    Z2 = W2.dot(A1) + b2
    A2 = softMax(Z2)
    
    return Z1, A1, Z2, A2

def one_hot(Y): 
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    
    return one_hot_Y

def ReLU_deriv(Z):
    return (Z > 0).astype(float)

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    m = Y.size  # Number of examples

    dZ2 = A2 - one_hot_Y  # Shape: (10, m)
    dW2 = 1 / m * dZ2.dot(A1.T)  # Shape: (10, hidden_size)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)  # Shape: (10, 1)
    
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)  # Shape: (hidden_size, m)
    dW1 = 1 / m * dZ1.dot(X.T)  # Shape: (hidden_size, 784)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)  # Shape: (hidden_size, 1)
    
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha): 
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    
    return W1, b1, W2, b2

def get_predictions(A2): 
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y): 
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha): 
    W1, b1, W2, b2 = init_params()
    for i in range(iterations): 
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if (i % 10 == 0): 
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

if __name__ == "__main__": 
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, iterations=500, alpha=0.1)

    # Generate 20 random integer indices between 0 and the maximum index of your dataset
    max_index = X_train.shape[1] - 1  # Maximum valid index in X_train
    indices = np.random.randint(0, max_index + 1, size=20)

    for i in indices: 
        test_prediction(i, W1, b1, W2, b2)




