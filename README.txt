Files in this directory:

tweets.tgz: A collection of tweets from Twitter, with the following format:
tweet-id \t user-id \t tweet-text \t location-code
Each tweet corresponds to one where the user was semi-automatically identified as being in one of Boston (B), Houston (H), San Diego (SD), Seattle (Se), or Washington (W). These were partitioned as follows:
	train-tweets.txt: These will form the training data for Project 2

	dev-tweets.txt: These will form the development data for Project 2, which you should use to evaluate your classifier(s)

	test-tweets.txt: These will form the test data for Project 2; they are unlabelled (all labelled with "?")

best35.tgz: As described in the project specifications, mutual information was used to find the 10 terms in the collection with the best (apparent) predictive capacity for each of the 5 classes, of these 5x10=50 terms, 15 were duplicated between classes, leaving 35 terms in total.
For each tweet in the three datasets above, case was folded, non-alphabetic characters were removed, and the frequency for each of these 35 terms was recorded as an attribute value.
These files are in ARFF format: there is a header which details the 37 attributes (tweet id, these 35 terms, and the class (location)).
	train-best35.arff

	dev-best35.arff

	test-best35.arff

best446.tgz: As above, but rather than mutual information being used to find the 10 best terms for each of the 5 classes, we instead found the 100 best terms. The resulting unique set was 446 terms in total.
There are some oddball terms here, because the tweet set (and pre-processing methods) used to calculate these mutual information values isn't precisely the same as the method used to generate the frequencies for the ARFF files. (Oops!) You can try to account for this, but our estimates show the net effect on the utility of the terms will be small.
	train-best446.arff

	dev-best446.arff

	test-best446.arff

most100.tgz: As above, but rather than using a feature selection method like mutual information, we instead used the 100 most common terms in the collection (acccording to document frequency).
	train-most100.arff

	dev-most100.arff

	test-most100.arff
