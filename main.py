import matplotlib.pyplot as plt
from DataHandler import DataHandler
from LinearClassifier import LinearClassifier

if __name__ == '__main__':
    #generating a normal distributed data (1000 samples per class)
    data_handler = DataHandler()
    class0Dataset = data_handler.get2DGaussian(1000, [-2, -2])
    class1Dataset = data_handler.get2DGaussian(1000, [2, 2])

    #labling the data
    class0Dataset = data_handler.label(class0Dataset, 0)
    class1Dataset = data_handler.label(class1Dataset, 1)

    #shuffling the data
    dataset = data_handler.shuffle(class0Dataset + class1Dataset)
    ###############################################################

    classifier = LinearClassifier()

    print("initial weights : ", classifier.weights)
    print("initial bais : ", classifier.bais)
    actual = [row[-1] for row in dataset]
    pridected = [classifier.predict(row) for row in dataset]
    print("Accuracy before training : %.2f%%\n" %
          data_handler.accuracy(actual, pridected))
    classifier.plot()
    plt.show()


    learning_rate = 0.01
    n_folds = 5
    n_epoch = 2
    scores = data_handler.evaluate_model(dataset, classifier, n_folds, learning_rate, n_epoch)
    print('Scores: %s' % scores)
    print('Average Accuracy: %.2f%%' % (sum(scores) / float(len(scores))))

    print("final weights : ", classifier.weights)
    print("final bais : ", classifier.bais)

    # plot results
    x, y, label = zip(*class0Dataset)
    X, Y, label = zip(*class1Dataset)
    plt.plot(x, y, 'x')
    plt.plot(X, Y, 'x')
    classifier.plot()
    plt.show()