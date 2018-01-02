import numpy
import matplotlib.pyplot as pyplot

covariance = [[1.2, 0], [0, 1.2]]  # Spherical covariance

class Dataset(object):

    def get2DGaussian(self, size, mean, covariance=covariance):
        x, y = numpy.random.multivariate_normal(mean, covariance, size).T
        pyplot.plot(x, y, 'x')
        return list(zip(x, y))

    def getGaussian(self):
        mu, sigma = -2, 1 # mean and standard deviation

        dataset = numpy.random.normal(mu, sigma, 1000)

        def npdf(mu, sigma): # Normal probabilty density function
            return 1 / (sigma * numpy.sqrt(2 * numpy.pi)) * \
                   numpy.exp(-(bins - mu) ** 2 / (2 * sigma ** 2))


        count, bins, ignored = pyplot.hist(dataset, 30, normed=True)
        pyplot.plot(bins, npdf(mu, sigma), linewidth = 2, color = 'r')

        pyplot.show()

    def label(self, data, lable):
        for i in range(len(data)):
            data[i] += (lable,)
        return data

