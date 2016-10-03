#!/usr/local/bin/python3

# Note, arff won't work with python 3.5, you need 3.4

import arff
import numpy as np
from os.path import isfile
import pickle
from sklearn.naive_bayes import MultinomialNB
import time

train35F = "best35/train-best35.arff"
dev35F  = "best35/dev-best35.arff"

train446F = "best446/train-best446.arff"
dev446F = "best446/dev-best446.arff"

def secondAttempt(trainDataFile, devDataFile):

    def readDataFile(dataFile):
        """
        Checks if pickle already exists, and if so loads it.
        Otherwise reads in the arff data and processes it accordingly.
        It then dumps the instances and labels tuple to a pickle file.
        """
        pickleFileName = dataFile.split(".")[:-1]
        pickleFileName.append(".p")
        pickleFileName = "".join(pickleFileName)

        # If instances and labels data exist already, unpickle it.
        if isfile(pickleFileName):
            print("Unpickling from {}".format(pickleFileName))
            with open(pickleFileName, 'rb') as f:
                instances, labels = pickle.load(f)
        # Otherwise read in the data, process it and pickle it for later.
        else:
            print("Reading in data file {}".format(dataFile))
            dataset = arff.load(open(dataFile, 'r'))
            data = dataset['data']

            print("Processing data file {}".format(dataFile))
            instances, labels = processData(data)

            print("Pickling instances/labels data to {}".format(pickleFileName))
            with open(pickleFileName, 'wb') as f:
                pickle.dump((instances, labels), f)

        return (instances, labels)


    def processData(data):
        """
        Returns instances and labels in separate numpy arrays.
        """
        instances, labels = [], []
        for i in data:
            instance = i[1:-1]
            label = i[-1]
            instances.append(instance)
            labels.append(label)

        instances = np.array(instances)
        labels = np.array(labels)

        return (instances, labels)


    def predictPrint(clf, instance, diagnostic=False):
        """
        Returns the predicted class for an instance according to the model.
        Has an option to print which features are present in the instance.
        """
        if diagnostic:
            print("Features present in instance: ")
            for i in range(len(instance)):
                if instance[i] == 1:
                    print(attributes[i], end=' ')
            print()

        return clf.predict([instance])


    """
    Creating and training the model.
    """
    startTime = time.time()

    instances, labels = readDataFile(trainDataFile)

    print("Training the model")
    clf = MultinomialNB()
    clf.fit(instances, labels)

    timeForTrain = time.time() - startTime

    #print(type(instances[0]))
    #print(predictPrint(clf, instances[0]))




    """
    Testing and evaluating the model
    """
    startTime = time.time()

    instances, labels = readDataFile(devDataFile)

    print("Testing the model")
    numCorrect = 0
    numWrong  = 0
    lenInstances = len(instances)
    for i in range(lenInstances):
        # Status update of how it's going.
        if i % 1000 == 0:
            print(str(i).zfill(len(str(lenInstances))) + "/" + str(lenInstances))
        instance = instances[i]
        label = labels[i]
        res = predictPrint(clf, instance)
        # print("-- Predicted label: {} || Correct label: {} --". format(res, label))
        if res == label:
            numCorrect += 1
        else:
            numWrong += 1

    timeForTest = time.time() - startTime

    print("Number of correct classifications: {}".format(numCorrect))
    print("Number of wrong classifications: {}".format(numWrong))
    print("Percentage of correct classifications: {0:.2f}%".format(numCorrect*100/numWrong))
    print("Time taken to train the model: {0:.2f} sec".format(timeForTrain))
    print("Time taken to test the model: {0:.2f} sec".format(timeForTest))






def firstAttempt():
    print("Reading in arff file {}".format(train35F))
    dataset = arff.load(open(train35F, 'r'))
    # Dataset has these keys: 'data', 'relation', 'attributes', 'description'
    classes = np.array(dataset["attributes"][-1][1])
    classes = classes.reshape((5, 1))
    print(classes)
    for i in classes:
        print(i)

    attributes = [i[1] for i in dataset["attributes"][1:-1]]

    print(classes)
    print("Converting to numpy array")
    data = np.array(dataset['data'])
    clf = MultinomialNB()

    #data = np.random.randint(2, size=(5, 100))

    from collections import Counter

    # print(Counter([len(i) for i in data]))
    # The above line verifies that all instance lines have the same length.

    print(data.shape, classes.shape)
    clf.fit(data[:-1], data)

    print("Printing data")
    print(data)

    def predictPrint(clf, instance):
        """
        for i in range(1, len(instance - 2)):
            if instance[i] == 1:
                print(attributes[i], end=' ')
        print()
        """
        print(clf.predict([instance]))

    predictPrint(clf, data[0])

if __name__ == "__main__":
    secondAttempt(train446F, dev446F)