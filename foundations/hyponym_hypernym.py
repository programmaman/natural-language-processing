# Levenshtein Edit Distance
def problem2(string1, string2):
    # Create Table
    row = len(string1) + 1
    column = len(string2) + 1
    table = []

    for i in range(row):
        col = []
        for j in range(column):
            col.append(0)
        table.append(col)

    # Put string Length in Rows
    for i in range(1, row):
        table[i][0] = i
    # Put string length in column
    for j in range(1, column):
        table[0][j] = j

    for i in range(1, row):
        for j in range(1, column):
            if string1[i - 1] == string2[j - 1]:
                substitution_cost = 0
            else:
                substitution_cost = 2
            # Fill table with edit distances
            table[i][j] = min(table[i - 1][j] + 1,
                              table[i][j - 1] + 1,
                              table[i - 1][j - 1] + substitution_cost)

    return table[row - 1][column - 1]


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
