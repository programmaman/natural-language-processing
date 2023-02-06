import gensim
import gensim.models.keyedvectors as word2vec
import re
import nltk
import numpy as np
import json
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')

def classifySubreddit_train(trainFile):
    global model
    global logistic_regression_model

    list_of_subreddits = []
    comment_list = []
    sentence_vector_list = []

    # open and read trainFile
    with open(trainFile, 'r', encoding='utf8') as f:
        jsonlist = {}
        jsonlist = f.readlines()

    # load each line of jsonlist into dictionary
    for i in range(len(jsonlist)):
        # sentence vector container
        sentence = np.zeros(300)

        # load ith row of the json list
        dictionary = json.loads(jsonlist[i])

        # extract the comment body and subreddit classification
        subreddit_comment = dictionary["body"]
        subreddit_classification = dictionary["subreddit"]
        list_of_subreddits.append(subreddit_classification)
        tokenized_comment = tokenize(subreddit_comment)

        # create sentence vector
        for word in tokenized_comment:
            if word not in stopwords_list:
                vector = getVector(word)
                sentence = sentence + vector
        # add vector to list
        sentence_vector_list.append(sentence)

    # fit logistic regression to data
    logistic_regression_model.fit(sentence_vector_list, list_of_subreddits)


def classifySubreddit_test(comment):
    global model
    global logistic_regression_model

    sentence_vector_list = []

    sentence = np.zeros(300)

    tokenized_text = tokenize(comment)

    for word in tokenized_text:
        if word not in stopwords_list:
            vector = getVector(word)
            sentence = sentence + vector

    sentence_vector_list.append(sentence)

    string = logistic_regression_model.predict(sentence_vector_list)
    return string[0]
