import numpy as np


def convert_sentence_to_adjacency_matrix(sentence):
    '''
    Input: sentence in json
    Output: adjacency matrix (gold standard)
    '''

    sentence_len = len(sentence['words'])

    # Initialize a matrix of size N x N
    adjacency_matrix = np.zeros((sentence_len, sentence_len))

    for word in sentence['words']:
        word_id = int(word['id'])
        head = int(word['head'])

        # Ignore the root(0)-(-1) connection
        if head == -1:
            continue

        adjacency_matrix[head][word_id] = 1

    return adjacency_matrix

def adjacency_matrix_to_tensor(matrix):
    output = [0] * matrix.shape[0]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if matrix[i][j] == 1:
                output[j] = i
    output1 = torch.LongTensor(output)
    return(output1)

test_matrix = np.array(([1, 0, 1, 1, 0],
                       [0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]))
print(test_matrix)
print(adjacency_matrix_to_tensor(test_matrix))

#something like this will convert directly
def convert_sentence_to_tensor(sentence):
    sentence_len = len(sentence['words'])
    output = [0] * sentence_len
    for word in sentence['words']:
        output[int(word['id'])] = int(word['head'])
    output1 = torch.LongTensor(output)
    return output1
