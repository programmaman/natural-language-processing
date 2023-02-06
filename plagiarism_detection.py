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


def findPlagiarism(sentences, target):
    global model
    similarity_list = []

    # tokenize target
    tokenized_target = tokenize(target)
    filtered_target = []

    # remove "useless" words
    for word in tokenized_target:
        if word not in stopwords_list:
            filtered_target.append(word)

    # target vector
    transformed_target = [w for w in filtered_target if w in model.key_to_index]

    for i in range(len(sentences)):
        # tokenize sentences
        tokenized_sentence = tokenize(sentences[i])
        filtered_sentence = []

        # remove "useless" words
        for word in tokenized_sentence:
            if word not in stopwords_list:
                filtered_sentence.append(word)

        # sentence vector
        transformed_sentence = [w for w in filtered_sentence if w in model.key_to_index]

        # compare vector similarity and store it in list
        similarity_list.append(model.n_similarity(transformed_sentence, transformed_target))

    # get most similar vector entry
    max_similarity = max(similarity_list)

    # return answer
    return similarity_list.index(max_similarity)
