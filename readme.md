# Knowledge Technologies Assignment 2

`ml.py` is the main script. The layout is a bit messy, but considering that it's
not marked quite a bit of work went into making it easily extensible for both
additional classifiers as well as additional evaluation methods. There is also
functionality which pickles the processed data pulled from the arff files, which
cuts down the time of each run enormously (from the order of 5 minutes for the
446 training dataset down to a few seconds).

Just try to run ml.py and it'll print usage information. Use it like this:

`./ml.py 446 nb macro`

The averaging type (e.g. "macro") is optional.

The supported datasets are 35 and 446. These represent the development datasets.
If you want to use the test sets instead, the syntax is 35test and 446test.
If you run the scripts with the test sets, it will crash after generating the
predictions (since there are no labels with which to evaluate), but everything
up to that point, including the predictions, will work fine.

The classifiers are nb, svm and nbBoosted, from an earlier idea where I would
experiment with boosting the algorithm with AdaBoost and changing the kernel
for SVM.

`plot_confusion_matrix.py` is just some script from the sci-kit documentation
which I modified, all it does is print a graphical confusion matrix.
Source here: https://goo.gl/XwMr6N

Predictions have been included for the four primary datasets considered. If
marking requires only one set, use predictedNB446test, as it has the best
results. These are of course for the test data, not the dev data.
