import re

# TODO: This implentation needs to be reinterpreted into OOP/
# TODO: /structure to allow for more powerful meaning inferences

# Extract Hyper and Hypo Relationships
def problem1(noun_list, corpus):

    # Lower the case of all text, remove all formatting and remove unnecessary "a", "an", "the"
    corpus = regularize_text(corpus)

    # Add symbols for identifying meaningful segments
    corpus = symbolize_text(corpus)

    # Tokenize string into words
    tokenized_corpus = corpus.split()

    # Static hypernym to store parent hypernym when finding the meaning of lists of hyponyms
    static_hypernym = ''

    # Store index to capture list of hyponyms
    list_beginning_index = 0

    # Empty set to hold tuples
    answer = set()

    # Iterate through tokens until a symbol, "=", ">" or "+" is found
    for iterator in range(len(tokenized_corpus)):

        # if '=' is found, this is a "hyponym is hypernym" relationship.
        if tokenized_corpus[iterator] == '=':
            test_hyponym = tokenized_corpus[iterator - 1]
            test_hypernym = tokenized_corpus[iterator + 1]
            hyponym = ''
            hypernym = ''
            # check if token is in the NP list and replace it with NP list token if so
            for i in range(len(noun_list)):
                if test_hyponym in noun_list[i]:
                    hyponym = noun_list[i]
                if test_hypernym in noun_list[i]:
                    hypernym = noun_list[i]
            if len(hyponym) != 0 and len(hypernym) != 0:
                hypo_tuple = (hypernym, hyponym)
                answer.add(hypo_tuple)

        # if '>' is found, this is a "hypernym such as hyponym" relationship.
        if tokenized_corpus[iterator] == '>':
            list_beginning_index = iterator
            test_hypernym = tokenized_corpus[iterator - 1]
            test_hyponym = tokenized_corpus[iterator + 1]
            hyponym = ''
            hypernym = ''

            # check if token is in the NP list and replace it with NP list token if so
            for i in range(len(noun_list)):
                if test_hyponym in noun_list[i]:
                    hyponym = noun_list[i]
                if test_hypernym in noun_list[i]:
                    hypernym = noun_list[i]
                    static_hypernym = test_hypernym
            if len(hyponym) != 0 and len(hypernym) != 0:
                hypo_tuple = (hypernym, hyponym)
                answer.add(hypo_tuple)

        # if '+' is found, this is an additional "hypernym such as hyponym" relationship. Present these as tuples.
        if tokenized_corpus[iterator] == '+':
            list_ending_index = iterator
            test_hyponym = tokenized_corpus[iterator + 1]
            hyponym = ''

            # check if token is in the NP list and replace it with NP list token if so
            for i in range(len(noun_list)):
                if test_hyponym in noun_list[i]:
                    hyponym = noun_list[i]
            if len(hyponym) != 0 and len(static_hypernym) != 0:
                hypo_tuple = (static_hypernym, hyponym)
                answer.add(hypo_tuple)
            while list_beginning_index < list_ending_index:
                for i in range(len(noun_list)):
                    if tokenized_corpus[list_beginning_index] in noun_list[i]:
                        hyponym = noun_list[i]
                        hypo_tuple = (static_hypernym, hyponym)
                        answer.add(hypo_tuple)
                        break
                list_beginning_index = list_beginning_index + 1
    return answer


# Lower case of all text, remove all formatting except whitespace and remove standalone "a", "an", "the"
def regularize_text(string):
    string = string.lower()
    string = re.sub(r'(?!\ )\W', '', string)  # remove all formatting characters except whitespace
    # remove 'a', 'an', and 'the' from corpus
    string = re.sub(r'\bthe\b', '', string)
    string = re.sub(r'\ban\b', '', string)
    string = re.sub(r'\ba\b', '', string)
    return string


# Convert "is" relationships to an "=" sign for identifying in tokens, convert "including" and "such "as" into ">",
# and convert "and" and "or" into "+"
def symbolize_text(string):
    string = re.sub(r'(?:(is a kind of )|(?:(is a type of ))|(?:is ))', '= ', string)
    string = re.sub(r'(?:(was a kind of )|(?:(was a type of ))|(?:was ))', '= ', string)
    string = re.sub(r'(?:(are a kind of )|(?:(are a type of ))|(?:are ))', '= ', string)
    string = re.sub(r'including', '>', string)
    string = re.sub(r'such as', '>', string)
    string = re.sub(r'(and |or  )', '+ ', string)
    return string
