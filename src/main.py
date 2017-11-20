import torch
import json
import os
from gensim.models import Word2Vec
from torch.autograd import Variable

current_path = os.path.dirname(__file__)

class MLP(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

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
#         raise Exception('File not provided')
        filepath = current_path + '/../data/en-ud-train.json'
    
    data = json.load(open(filepath, 'r'))

    corpus_words = []
    corpus_pos = []

    for sentence in data:
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

    BiLSTM = torch.nn.LSTM(len_data_vec, len_data_vec, 2, bidirectional=True) # the input and output sizes need not be the same
    
    MLP_head = MLP(len_data_vec*2, 120, len_feature_vec)
    MLP_dep = MLP(len_data_vec*2, 120, len_feature_vec)

    U_1 = Variable(torch.FloatTensor(len_feature_vec, len_feature_vec))
    u_2 = Variable(torch.FloatTensor(len_feature_vec))

    for sentence in data:
        y = torch.FloatTensor(1, len(sentence['words']), len_data_vec).zero_()
        x = Variable(y)
        for i, word in enumerate(sentence['words']):
            word_embedding = word_embeddings[word['form']]
            pos_embedding = pos_embeddings[word['xpostag']]
            x[0, i] = torch.cat((torch.from_numpy(word_embedding), torch.from_numpy(pos_embedding)), 0)
        r, _ = BiLSTM(x)
        
        print(r.size()) # TO DO; debug this. r is size (1, 29, 20)?
        
        # FROM <row> TO <column>
        s = torch.FloatTensor(len(sentence['words']), len(sentence['words'])).zero_()
        for i, vec_h in enumerate(r[0,:]):
            h_head = MLP_head(vec_h)
            for j, vec_d in enumerate(r[0,:]):
                h_dep = MLP_dep(vec_d)
                print(h_head.size())
                print(U_1.size())
                print(u_2.size())
                print(h_dep.size())
                s[i,j] = torch.mm(h_head.t(), torch.mm(U_1, h_dep)) + torch.mm(h_head.t(), u_2)
        print(s)
        break

#         s[i] = h_head.T * U_1 * h_dep[i] + h_head.T * u_2


if __name__ == '__main__':
    dependency_parser()