import numpy
import matplotlib.pyplot as pyplot
from Dataset import Dataset


class FishClassifier(object):
    def __init__(self):
        self.weights = [0.5, 1]
        self.bais = 1.2

    def predict(self, row):
        activation = self.bais
        for i in range(len(row) - 1):
            activation += self.weights[i] * row[i]
        return 1 if activation >= 0 else 0

    def train(self, data, learningRate, n_session):
        # Estimate Perceptron weights using stochastic gradient descent
        for session in range(n_session):
            total_error = 0
            for row in data:
                predection = self.predict(row)
                error = row[-1] - predection
                total_error += abs(error)
                self.bais += learningRate * error

                for i in range(len(self.weights)):
                    self.weights[i] += learningRate * error * row[i]
            print('>session %d, learningrate=%.3f, errors : %.3f' % (session, learningRate, total_error))


    def draw(self):
        x2 = [self.weights[0], self.weights[1], -self.weights[1], self.weights[0]]
        x3 = [self.weights[0], self.weights[1], self.weights[1], -self.weights[0]]

        x2x3 = numpy.array([x2, x3])
        X, Y, U, V = zip(*x2x3)
        ax = pyplot.gca()
        ax.quiver(X, Y, U, V, scale=1, color='green')
        """
        x = numpy.array(range(-7, 7))
        y = (-self.weights[0] / self.weights[1]) * x + self.bais
        pyplot.plot(x, y)"""






if __name__ == '__main__':
    #generating and labeling data
    generator = Dataset()
    
    salmonDataset = generator.get2DGaussian(1000, [-2, -2])
    salmonDataset = generator.label(salmonDataset, 0)

    fishDataset = generator.get2DGaussian(1000, [2, 2])
    fishDataset = generator.label(fishDataset, 1)

    training_data = salmonDataset + fishDataset
    #######################################################


    classifier = FishClassifier()

    for row in training_data:
        predection  = classifier.predict(row)
        print("Expected : %d  predected : %d " % (row[-1], predection))

    classifier.draw()
    pyplot.show()

    learning_rate = 0.1
    sessions = 5
    classifier.train(training_data, learning_rate, sessions)




    print(classifier.weights)
    print(classifier.bais)
    classifier.draw()
    pyplot.show()
