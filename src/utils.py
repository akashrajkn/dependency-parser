import numpy as np


def convert_sentence_to_adjancency_matrix(sentence):
    '''
    Input: sentence in json
    Output: adjancency matrix (gold standard)
    '''

    sentence_len = len(sentence['words'])

    # Initialize a matrix of size N x N
    adjancency_matrix = np.zeros((sentence_len, sentence_len))

    for word in sentence['words']:
        word_id = int(word['id'])
        head = int(word['head'])

        # Ignore the root(0)-(-1) connection
        if head == -1:
            continue

        adjancency_matrix[head][word_id] = 1

    return adjancency_matrix

def convert_adjancecy_matrix_to_sentece(matrix):
    pass
