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