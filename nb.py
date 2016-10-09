#!/usr/local/bin/python3

"""
Author:     Daniel Porteous
Student #:  696965
Login:      porteousd
"""

# Note, arff won't work with python 3.5, you need 3.4

import arff
import numpy as np
from os.path import isfile
import pickle
from random import shuffle
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier
import time
import csv

train35F = "best35/train-best35.arff"
dev35F  = "best35/dev-best35.arff"
test35F = "best35/test-best35.arff"

train446F = "best446/train-best446.arff"
dev446F = "best446/dev-best446.arff"
test446F = "best446/test-best446.arff"

outputFileName = "predicted.csv"


def readArffFile(dataFile):
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
            ids, instances, labels, features, classes = pickle.load(f)
    # Otherwise read in the data, process it and pickle it for later.
    else:
        print("Reading in data file {}".format(dataFile))
        dataset = arff.load(open(dataFile, 'r'))

        data = dataset['data']
        # shuffle(data) <- This doesn't affect the results. Why?
        classes = dataset["attributes"][-1][-1]
        features = [i[-1] for i in dataset["attributes"][1:-1]]

        print("Processing data file {}".format(dataFile))
        ids, instances, labels = processData(data)

        print("Pickling data to {}".format(pickleFileName))
        with open(pickleFileName, 'wb') as f:
            pickle.dump((ids, instances, labels, features, classes), f)

    return (ids, instances, labels, features, classes)

def processData(data):
    """
    Returns instances and labels in separate numpy arrays.
    """
    ids, instances, labels = [], [], []
    for i in data:
        idField = int(i[0])
        instance = i[1:-1]
        label = i[-1]
        ids.append(idField)
        instances.append(instance)
        labels.append(label)

    ids = np.array(ids)
    instances = np.array(instances)
    labels = np.array(labels)

    return (ids, instances, labels)

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

def writeOutput(ids, predicted, fname):
    # ids and predicted should be the same length.
    idsWithPredicted = [list(a) for a in zip(ids, [i[0] for i in predicted])]
    with open(fname, "w") as f:
        a = csv.writer(f, delimiter=',')
        a.writerow(["Id", "Category"])
        a.writerows(idsWithPredicted)

# Prints a classification report in which you're allowed to specify your own beta for f-score.
# Prints different averages than classsification_report prints for some reason.
def parameterizableReport(correct, predicted, target_names, beta=1.0):
    results = precision_recall_fscore_support(correct, predicted, labels=target_names, beta=beta) # ,average="micro"
    print(results)
    transposed = list(map(list, zip(*results)))

    print("             precision    recall  f{0:.2f}-score    support".format(beta))
    print()
    i = 0
    out = []
    for line in transposed:
        compiled = []
        compiled.append(" " * (11-len(target_names[i])))
        compiled.append(str(target_names[i]))
        compiled.append("       {0:.2f}      {1:.2f}         {2:.2f}      {3:5d}".format(*line))
        #compiled.append(str(line[3]))
        out.append("".join(compiled))
        i += 1

    averagePrecision = np.average(results[0])
    averageRecall = np.average(results[1])
    averageFScore = np.average(results[2]) # TODO could change averaging method.
    totalSupport = sum(results[3])

    args = [averagePrecision, averageRecall, averageFScore, totalSupport]
    out.append("\navg / total       {0:.2f}      {1:.2f}         {2:.2f}      {3:5d}".format(*args))
    return "\n".join(out)



def trainAndEvaluate(trainDataFile, devDataFile, classifier):

    """
    Creating and training the model.
    """
    startTime = time.time()

    ids, instances, labels, features, classes = readArffFile(trainDataFile)

    print("Training the model")

    if classifier == "svc" or classifier == "svm":
        clf = LinearSVC()
    elif classifier == "nb":
        clf = MultinomialNB()
    elif classifier.lower() == "nbboost" or classifier.lower() == "nbboosted":
        clf = MultinomialNB()
        clf = AdaBoostClassifier(clf)
    elif classifier.lower() == "1r":
        print("Sorry, 1R / LinearRegression isn't working right now")
        exit()
        clf = LinearRegression(copy_X=False,fit_intercept=True, normalize=False)
    else:
        print("Invalid classifier choice.")
        return

    clf.fit(instances, labels)

    timeForTrain = time.time() - startTime
    numTrainInstances = len(instances)

    """
    Testing and evaluating the model
    """
    startTime = time.time()

    # Throw away the features and classes, we've already read them in.
    ids, instances, labels, _, _ = readArffFile(devDataFile)

    print("Testing the model")
    numCorrect = 0
    numWrong  = 0
    lenInstances = len(instances)
    predicted = []
    for i in range(lenInstances):
        # Status update of how it's going.
        if i % 1000 == 0:
            print("\r" + str(i).zfill(len(str(lenInstances))) + "/" + str(lenInstances) + " ", end="")
        instance = instances[i]
        label = labels[i]
        res = predictPrint(clf, instance)
        predicted.append(res)
        # print("-- Predicted label: {} || Correct label: {} --". format(res, label))
        if res == label:
            numCorrect += 1
        else:
            numWrong += 1
    print()

    predicted = np.array(predicted)
    writeOutput(ids, predicted, outputFileName)
    timeForTest = time.time() - startTime
    numDevInstances = len(instances)


    """
    Printing various evaluation metrics.
    """
    # report = classification_report(labels, predicted, target_names=classes)
    report = parameterizableReport(labels, predicted, beta=0.5, target_names=classes)
    print(report)
    print()
    #print(classification_report(labels, predicted, target_names=classes))

    """
    print("Number of training instances: {}".format(numTrainInstances))
    print("Number of dev instances: {}".format(numDevInstances))
    print()

    print("Number of correct classifications: {}".format(numCorrect))
    print("Number of wrong classifications: {}".format(numWrong))
    print("Percentage of correct classifications: {0:.2f}%".format(numCorrect*100/(numCorrect+numWrong)))
    print()
    """

    print("Time taken to train the model: {0:.2f} sec".format(timeForTrain))
    print("Time taken to test the model: {0:.2f} sec".format(timeForTest))
    print()


from sys import argv, exit

if __name__ == "__main__":
    if len(argv) == 3:

        dataSet = argv[1]
        classifier = argv[2]

        if dataSet == "35":
            trainFile = train35F
            devFile = dev35F
        elif dataSet == "446":
            trainFile = train446F
            devFile = dev446F
        elif dataSet == "35test":
            trainFile = train35F
            devFile = test35F            
        elif dataSet == "446test":
            trainFile = train446F
            devFile = test446F
        else:
            print("Invalid data set choice.")
            exit() 

        trainAndEvaluate(trainFile, devFile, classifier)
    else:
        print("Usage: {} [data_set] [classifier]".format(argv[0]))


"""
devTestIdsF = "test.ids"

    def readTestIdsFile(dataFile):

        tweetsDataFile = 

        with open(dataFile, "r") as f:
            contents = f.read().splitlines()
        print(contents)
    

    # This means we got the test.ids file, for which we don't have labels and have
    # to convert into our nice format.
    if devDataFile == devTestIdsF:
        ids, instances = readTestIdsFile(devDataFile)
        

        # This means that the extra arg to test against test.ids was passed.
        if (len(argv)) == 4:
            devFile = devTestIdsF
"""