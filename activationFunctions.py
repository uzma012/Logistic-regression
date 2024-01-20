import numpy as np 
import math
def sigmoid(x):
    return 1/(1 + np.exp(-x))
def relu(x):
    return max(0.0, x)
def leaky_relu(x, alpha=0.1):
    return max(x, alpha * x)
def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))\

def softmax(x):
    # Subtracting the maximum value for numerical stability
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0)