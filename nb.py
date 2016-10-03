#!/usr/local/bin/python3

# Note, arff won't work with python 3.5, you need 3.4

import arff
import numpy as np
from os.path import isfile
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
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
                instances, labels, classes = pickle.load(f)
        # Otherwise read in the data, process it and pickle it for later.
        else:
            print("Reading in data file {}".format(dataFile))
            dataset = arff.load(open(dataFile, 'r'))
            data = dataset['data']
            classes = dataset["attributes"][-1][-1]

            print("Processing data file {}".format(dataFile))
            instances, labels = processData(data)

            print("Pickling instances/labels data to {}".format(pickleFileName))
            with open(pickleFileName, 'wb') as f:
                pickle.dump((instances, labels, classes), f)

        return (instances, labels, classes)


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

    instances, labels, classes = readDataFile(trainDataFile)

    print("Training the model")
    clf = MultinomialNB()
    clf.fit(instances, labels)

    timeForTrain = time.time() - startTime
    numTrainInstances = len(instances)


    """
    Testing and evaluating the model
    """
    startTime = time.time()

    # Throw away the classes, we've already read them in.
    instances, labels, _ = readDataFile(devDataFile)

    print("Testing the model")
    numCorrect = 0
    numWrong  = 0
    lenInstances = len(instances)
    predicted = []
    for i in range(lenInstances):
        # Status update of how it's going.
        if i % 1000 == 0:
            print(str(i).zfill(len(str(lenInstances))) + "/" + str(lenInstances))
        instance = instances[i]
        label = labels[i]
        res = predictPrint(clf, instance)
        predicted.append(res)
        # print("-- Predicted label: {} || Correct label: {} --". format(res, label))
        if res == label:
            numCorrect += 1
        else:
            numWrong += 1

    predicted = np.array(predicted)

    timeForTest = time.time() - startTime
    numDevInstances = len(instances)


    """
    Printing various evaluation metrics.
    """
    report = classification_report(labels, predicted, target_names=classes)
    print(report)

    print("Number of training instances: {}".format(numTrainInstances))
    print("Number of dev instances: {}".format(numDevInstances))
    print()

    print("Number of correct classifications: {}".format(numCorrect))
    print("Number of wrong classifications: {}".format(numWrong))
    print("Percentage of correct classifications: {0:.2f}%".format(numCorrect*100/(numCorrect+numWrong)))
    print()

    print("Time taken to train the model: {0:.2f} sec".format(timeForTrain))
    print("Time taken to test the model: {0:.2f} sec".format(timeForTest))
    print()

if __name__ == "__main__":
    secondAttempt(train35F, dev35F)