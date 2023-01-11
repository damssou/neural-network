import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

x, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
y = y.reshape((y.shape[0], 1))
x_train = x.T
y_train = y.T


def matrix_size(x, y):

    dimensions = {
        'n0' : x.shape[0],
        'n1' : 32,
        'n2' : 32,
        'n3' : y.shape[0]
    }

    return dimensions


class neural_network:

    def __init__(self, dimensions):
        self.settings = {'c_max' : len(dimensions)}

        for c in range(1, len(dimensions)):
            self.settings['w' + str(c)] = np.random.randn(dimensions['n' + str(c)], dimensions['n' + str(c - 1)])
            self.settings['b' + str(c)] = np.random.randn(dimensions['n' + str(c)], 1)


    def forward_propagation(self, x):
        self.activations = {'a0' : x}
   
        for c in range(1, self.settings['c_max']):
            z = np.dot(self.settings['w' + str(c)], self.activations['a' + str(c - 1)]) + self.settings['b' + str(c)]
            self.activations['a' + str(c)] = 1 / (1 + np.exp(-z))


    def log_loss(self, y):
        self.a = self.activations['a' + str(self.settings['c_max'] - 1)]
        return (-1 / y.shape[1]) * (np.sum (y * np.log(self.a) + (1 - y) * np.log(1-self.a)))


    def coef(self, y):
        y_true = 0
    
        for i in range(y.shape[1]):
            if (self.a[0][i] >= 0.5 and y[0][i] == 1): y_true += 1
            elif (self.a[0][i] < 0.5 and y[0][i] == 0): y_true += 1

        return y_true / y.shape[1]


    def back_propagation(self, y, learning_rate):
        gradients = {'dz' + str(self.settings['c_max'] - 1) : self.activations['a' + str(self.settings['c_max'] - 1)] - y}

        for c in range(self.settings['c_max'] - 2, 0, -1):
            gradients['dz' + str(c)] = (np.dot((self.settings['w' + str(c + 1)].T), gradients['dz' + str(c + 1)])) * (self.activations['a' + str(c)] * (1 - self.activations['a' + str(c)]))
        
        for c in range(self.settings['c_max'] - 1, 0, -1):
            gradients['dw' + str(c)] = (1 / y.shape[1]) * (np.dot(gradients['dz' + str(c)], (self.activations['a' + str(c - 1)].T)))
            gradients['db' + str(c)] = (1 / y.shape[1]) * (np.sum(gradients['dz' + str(c)]))

        for c in range(1, self.settings['c_max']):
            self.settings['w' + str(c)] = self.settings['w' + str(c)] - learning_rate * gradients['dw' + str(c)]
            self.settings['b' + str(c)] = self.settings['b' + str(c)] - learning_rate * gradients['db' + str(c)]



def main(x, y):
 
    dimensions = matrix_size(x_train, y_train)
    IA = neural_network(dimensions)
    error = []
    score = []

    for i in range(1500):
        IA.forward_propagation(x)
        error.append(IA.log_loss(y))
        score.append(IA.coef(y))
        IA.back_propagation(y, learning_rate=0.1)
    
    plt.plot(error)
    plt.show()
    plt.plot(score)
    plt.show()
    

main(x_train, y_train)




