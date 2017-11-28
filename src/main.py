import torch
import json
import os
import MST
import numpy as np
import torch.autograd as autograd
from gensim.models import Word2Vec
from torch.autograd import Variable


current_path = os.path.dirname(__file__)


class MLP(torch.nn.Module):
    def __init__(self, input_dimension, hidden_dimension, output_dimension):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dimension, hidden_dimension)
        self.linear2 = torch.nn.Linear(hidden_dimension, output_dimension)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        h = self.linear1(torch.sigmoid(x)).clamp(min=0)
        y_pred = self.linear2(h)
        return y_pred


def dependency_parser(filepath=None):
    # corpus_words in Format: [['I', 'like', 'custard'],...]
    # corpus_pos in Format: [['NN', 'VB', 'PRN'],...]

    if filepath is None:
        filepath = current_path + '../data/en-ud-train.json'

    data = json.load(open(filepath, 'r'))

    corpus_words = []
    corpus_pos = []

    for sentence in data[1]:
        words = []
        pos_s = []
        for word in sentence['words']:
            words.append(word['form'])
            pos_s.append(word['xpostag'])
        corpus_words.append(words)
        corpus_pos.append(pos_s)

    len_word_embed = 4
    len_pos_embed = 2
    len_data_vec = len_word_embed + len_pos_embed
    len_feature_vec = 20

    word_embeddings = Word2Vec(corpus_words, size=len_word_embed, window=5, min_count=1, workers=8)
    pos_embeddings = Word2Vec(corpus_pos, size=len_pos_embed, window=5, min_count=1, workers=8)

    BiLSTM = torch.nn.LSTM(len_data_vec, len_data_vec, bidirectional=True) # the input and output sizes need not be the same

    MLP_head = MLP(len_data_vec*2, 120, len_feature_vec)
    MLP_dep = MLP(len_data_vec*2, 120, len_feature_vec)

    U_1 = Variable(torch.randn(len_feature_vec, len_feature_vec))
    u_2 = Variable(torch.randn(1, len_feature_vec))

    for sentence in data:
        # LSTM takes in data in the shape
        # (len_sequence, batch_size, len_data_vec)
        # I think len_sequence is the length of the sentence
        # and I think batch_size is the amount of sentences
        # why it needs to be specified beforehand, I do not understand
        # perhaps I am wrong
        # perhaps it is always 1 because we loop over sequence in data?
        y = torch.FloatTensor(len(sentence['words']), 1, len_data_vec)
        x = Variable(y)
        for i, word in enumerate(sentence['words']):
            word_embedding = word_embeddings[word['form']]
            pos_embedding = pos_embeddings[word['xpostag']]
            x[i, 0, :] = torch.cat((torch.from_numpy(word_embedding), torch.from_numpy(pos_embedding)), 0)

        # initialisation of the hidden state and hidden cells of the LSTM (what does that mean?)
        # do we need to initialise this for every sentence? or only once at the initialisation of the LSTM?
        # the first 2 stands for bi-directional, is 1 for regular LSTM
        hidden = (autograd.Variable(torch.randn(2, 1, len_data_vec)), autograd.Variable(torch.randn(2 ,1 ,len_data_vec)))
        r, _ = BiLSTM(x, hidden)

        # print(r.size()) # r should have the size (len_sequence, batch_size(=1?), len_data_vec*2 (2 because bidirectional)) (29, 1, 12)

        # FROM <row> TO <column>
        # TO DO: make this a neat matrix multiplication instead of a double loop. consider concatenating a 1 and have U_1 and u_2 in one bigger matrix.
        s = Variable(torch.FloatTensor(len(sentence['words']), len(sentence['words'])))
        for i, vec_h in enumerate(r[:]):
            h_head = MLP_head(vec_h)
            for j, vec_d in enumerate(r[:]):
                h_dep = MLP_dep(vec_d)
                # s[i] = h_head.T * U_1 * h_dep[i] + h_head.T * u_2
                s[i,j] = torch.mm(h_head, torch.mm(U_1, torch.t(h_dep))) + torch.mm(h_head, torch.t(u_2))

        # FIXME: break statement is temporary
        break

# test_matrix1 = np.random.rand(6, 6)
# print(MST.edmonds(test_matrix1, 0))

if __name__ == '__main__':
    dependency_parser()
