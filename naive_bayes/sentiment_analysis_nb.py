import nltk, json, re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import numpy as np

nltk.download('punkt')

 Sentiment analysis naive bayes trainer
def calcSentiment_train(trainFile):
    global mnb
    global vectorized
    list_of_docs = []
    train_y = []

    # open and read trainfile
    with open(trainFile, 'r') as f:
        jsonlist = {}
        jsonlist = f.readlines()

    # load each line of jsonlist into dictionary, then split the review into an array and the sentiment into another
    for i in range(len(jsonlist)):
        dictionary = json.loads(jsonlist[i])
        review_corpus = dictionary["review"]
        list_of_docs.append(review_corpus)
        sentiment = dictionary["sentiment"]
        if sentiment:
            train_y.append(1)
        else:
            train_y.append(0)
    # vectorize the review array data
    print("Shape", np.shape(train_y))
    train_x = vectorized.fit_transform(list_of_docs)

    # train the NB model
    mnb.fit(train_x, train_y)


# load review, transform it with the vectorization table, then predict for the output
def calcSentiment_test(review):
    global vectorized
    test_x = vectorized.transform([review])
    return mnb.predict(test_x)
