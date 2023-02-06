import nltk, json, re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import numpy as np

nltk.download('punkt')


# custom tokenizer: lowercase all text, replace end of sentence periods '.' and question marks '?' with <s> </s>,
# : remove symbols, remove extra whitespace and remove numbers
def tokenize(string):
    string = string.lower()
    string = re.sub(r'(\.(?!\S))', ' </s> <s> ', string)
    string = re.sub(r'(\?(?!\S))', ' </s> <s> ', string)
    string = re.sub(r'[\.\,;:—""“”]', ' ', string)  # remove all formatting characters except whitespace
    string = re.sub(r'\s+', ' ', string)
    string = re.sub(r'\b[a-zA-Z]\s', ' ', string)
    return re.split("\\s+", string)


# global variables to store ngram models
unigram_model, bigram_model, trigram_model = {}, {}, {}

# global variable to hold Naive Bayes model
mnb = MultinomialNB(alpha=1.0, fit_prior=True)

# countvectorizer: create bi-gram and tri-gram vectorization and remove stop words
vectorized = CountVectorizer(input='content', ngram_range=(2, 3), tokenizer=tokenize, stop_words='english')


# ngram trainer
def calcNGrams_train(trainFile):
    # open file
    with open(trainFile, 'rb') as f:
        corpus = f.read()
    f.close()

    # transform corpus in utf-8 encoding (this might be a mistake)
    corpus = corpus.decode('utf-8')

    # tokenize corpus using custom tokenizer
    tokens = tokenize(corpus)

    # unique words vocabulary (unused in n-gram model)
    vocabulary = set(tokens)

    # create unigram, bigram and trigram models
    train(tokens)


# create unigram, bigram and trigram models
def train(tokens):
    global unigram_model, bigram_model, trigram_model

    # use a single token, double token and triple token window to create the respective ngram models
    for i in range(len(tokens) - 2):
        unigram = (tokens[i])
        bigram = (tokens[i], tokens[i + 1])
        trigram = (tokens[i], tokens[i + 1], tokens[i + 2])

        # record the n-gram occurence frequency in the corpus,k\

        # unigram
        if unigram in unigram_model.keys():
            unigram_model[unigram] += 1
        else:
            unigram_model[unigram] = 1

        # bigram
        if bigram in bigram_model.keys():
            bigram_model[bigram] += 1
        else:
            bigram_model[bigram] = 1

        # trigram
        if trigram in trigram_model.keys():
            trigram_model[trigram] += 1
        else:
            trigram_model[trigram] = 1


# finds statistically meaningless text
def find_non_human_sentence(sentence_list, bigram_counts, trigram_counts):
    count_array = []
    for sentence in sentence_list:
        tokenized_input = tokenize(sentence)
        for i in range(len(tokenized_input) - 2):
            trigram = (tokenized_input[i], tokenized_input[i + 1], tokenized_input[i + 2])
            bigram = (tokenized_input[i], tokenized_input[i + 1])
            count = + bigram_counts.get(bigram, 0) + trigram_counts.get(trigram, 0)
        count_array.append(count)

    min_human_sentiment = min(count_array)

    for idx in range(0, len(count_array)):
        if min_human_sentiment == count_array[idx]:
            return idx


# test function using trained ngrams
def calcNGrams_test(sentences):
    index = find_non_human_sentence(sentences, bigram_model, trigram_model)
    return index
