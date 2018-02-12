import numpy as np
import matplotlib.pyplot as plt

class LinearClassifier(object):
    def __init__(self):
        self.weights = [np.random.randn(), np.random.randn()]
        self.bais = np.random.randn()

    def predict(self, row):
        activation = self.bais
        for i in range(len(row) - 1):
            activation += self.weights[i] * row[i]
        return 1 if activation >= 0 else 0

    def train(self, data, learning_rate, n_epoch):
        """Estimates Perceptron weights using gradient descent"""
        for epoch in range(n_epoch):
            total_error = 0
            for row in data:
                prediction = self.predict(row)
                error = row[-1] - prediction
                total_error += error ** 2 #abs(error)
                self.bais += learning_rate * error

                for i in range(len(self.weights)):
                    self.weights[i] += learning_rate * error * row[i]
            print('>epoch %d, learning_rate : %.3f, errors during training : %.3f'
                  % (epoch, learning_rate, total_error))

    def plot(self):
        """Draws the classification line"""
        x = np.array(range(-7, 7))
        y = (-self.weights[0] / self.weights[1]) * x + self.bais
        plt.plot(x, y)

        """
        x2 = [self.weights[0], self.weights[1], -self.weights[1], self.weights[0]]
        x3 = [self.weights[0], self.weights[1], self.weights[1], -self.weights[0]]

        x2x3 = np.array([x2, x3])
        X, Y, U, V = zip(*x2x3)
        ax = plt.gca()
        ax.quiver(X, Y, U, V, scale=1, color='green')
        """



