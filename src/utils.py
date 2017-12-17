import json

import torch

import numpy as np


def convert_sentence_to_adjacency_matrix(sentence):
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


def adjacency_matrix_to_tensor(matrix):
    output = [0] * matrix.shape[0]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if matrix[i][j] == 1:
                output[j] = i
    output1 = torch.LongTensor(output)
    return(output1)


def get_labels():
    path = '../data/labels'

    with open(path) as f:
        content = f.readlines()

    labels = {}
    count = 1

    for line in content:
        temp = line.split(':')

        if len(temp) == 2:
            # labels.append(temp[0])
            labels[count] = temp[0]
            labels[temp[0]] = count
            count += 1

    with open(path + '.json', 'w+') as f:
        f.write(json.dumps(labels, indent=4))
