import numpy
import matplotlib.pyplot as plt
from random import randrange, sample

covariance = [[1.2, 0], [0, 1.2]]  # Spherical covariance

class DataHandler(object):

    def get2DGaussian(self, size, mean, covariance=covariance):
        x, y = numpy.random.multivariate_normal(mean, covariance, size).T
        plt.plot(x, y, 'x')
        return list(zip(x, y))

    def getGaussian(self):
        mu, sigma = -2, 1 # mean and standard deviation

        dataset = numpy.random.normal(mu, sigma, 1000)

        def npdf(mu, sigma): # Normal probabilty density function
            return 1 / (sigma * numpy.sqrt(2 * numpy.pi)) * \
                   numpy.exp(-(bins - mu) ** 2 / (2 * sigma ** 2))


        count, bins, ignored = plt.hist(dataset, 30, normed=True)
        plt.plot(bins, npdf(mu, sigma), linewidth = 2, color = 'r')

        plt.show()

    def label(self, data, lable):
        for i in range(len(data)):
            data[i] += (lable,)
        return data

    def cross_validation_split(self, dataset, n_folds):
        """Splits a dataset into k folds"""
        splited_data = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            splited_data.append(fold)
        return splited_data

    def shuffle(self, dataset):
        """shuffles the data and eliminates redundancies"""
        return sample(dataset, len(dataset))

    def evaluate_model(self, dataset, model, n_folds, *args):
        """Evaluates a model using a k-folds cross validation split"""
        folds = self.cross_validation_split(dataset, n_folds)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, []) #groups the lists within train_set into one
            test_set = fold

            model.train(train_set, *args)
            predicted = [model.predict(row) for row in test_set]
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy(actual, predicted)
            print('\t>test fold %d, samples : %d, accuracy : %.2f%% \n'
                  % (folds.index(fold), len(fold), accuracy))
            scores.append(accuracy)
        return scores

    def accuracy(self, actual, predicted):
        """Calculates accuracy percentage"""
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0
