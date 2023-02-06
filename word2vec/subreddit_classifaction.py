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


# global variable to store model
model = None

# global variable to store logistic regression classifier
logistic_regression_model = LogisticRegression(random_state=0)

# stopwords list
stopwords_list = set(stopwords.words('english'))


def tokenize(string):
    string = string.lower()
    # string = re.sub(r'(\.(?!\S))', ' </s> <s> ', string)
    # string = re.sub(r'(\?(?!\S))', ' </s> <s> ', string)
    string = re.sub(r'[\.\,;:—"”()]', ' ', string)  # remove all formatting characters except whitespace
    string = re.sub(r'\s+', ' ', string)
    string = re.sub(r'\b[a-zA-Z]\s', ' ', string)
    return re.split("\\s+", string)


# returns a vector value if embedding exists and returns an empty embedding value if not
def getVector(w):
    global model

    if w in model:
        return model[w]

    else:
        return np.zeros(300)


def setModel(Model):
    global model
    model = Model


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
