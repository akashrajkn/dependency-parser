import json
import os
import MST
import numpy as np
#import torch
import torch.cuda as torch  #<-- would that work?
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from gensim.models import Word2Vec
from torch.autograd import Variable

def pretrain_word_embeddings(data, len_word_embed, len_pos_embed):
    corpus_words = []
    corpus_pos = []
    all_words = []
    all_pos = []
    # corpus_words in the following shape:
    #  [ [some sentence]
    #    [some other sentence] ]
    #
    # all_words in the shape:
    # [ some sentence some other sentence]

    for sentence in data:
        words = []
        pos_s = []
        for word in sentence['words']:
            words.append(word['form'])
            pos_s.append(word['xpostag'])
        corpus_words.append(words)
        corpus_pos.append(pos_s)
        all_words.extend(words)
        all_pos.extend(pos_s)

    # THIS NOW WORKS FOR ONLY ONE SENTENCE, NO?
    w2i = {word: idx for idx, word in enumerate(all_words)}
    p2i = {pos: idx for idx, pos in enumerate(all_pos)}

    # pre-train word and pos embeddings. These will be starting points for our learnable embeddings
    word_embeddings_gensim = Word2Vec(corpus_words, size=len_word_embed, window=5, min_count=1, workers=8)
    pos_embeddings_gensim = Word2Vec(corpus_pos, size=len_pos_embed, window=5, min_count=1, workers=8)

    # initialise the embeddings. The tensors are still empty
    pretrained_word_embeddings = torch.FloatTensor(max(w2i.values())+1, len_word_embed)
    pretrained_pos_embeddings = torch.FloatTensor(max(p2i.values())+1, len_pos_embed)

    # fill the tensors with the pre-trained embeddings
    # THE EMBEDDING HAS SIZE OF THE AMOUNT OF WORDS, NOT THE AMOUNT OF UNIQUE WORDS: WASTE OF SPACE?
    for word in w2i.keys():
        idx = w2i[word]
        pretrained_word_embeddings[idx, :] = torch.from_numpy(word_embeddings_gensim[word])
    for pos in p2i.keys():
        idx = p2i[pos]
        pretrained_pos_embeddings[idx, :] = torch.from_numpy(pos_embeddings_gensim[pos])

    return w2i, p2i, pretrained_word_embeddings, pretrained_pos_embeddings

def train():
    # corpus_words in Format: [['I', 'like', 'custard'],...]
    # corpus_pos in Format: [['NN', 'VB', 'PRN'],...]
    current_path = os.path.dirname(__file__)
    filepath = current_path + '../data/toy_data.json'
    if filepath is None:
        filepath = current_path + '../data/en-ud-train.json'
    data = json.load(open(filepath, 'r'))
    # dummy_data = np.loadtxt('../data/dummy.txt', dtype=str)
    # data = dummy_data # <-- temporary of course
    len_word_embed = 4
    len_pos_embed = 2

    w2i, p2i, pwe, ppe = pretrain_word_embeddings(data, len_word_embed, len_pos_embed)

    network = Network(w2i, p2i, pwe, ppe, len_word_embed, len_pos_embed)
    network_params = list(network.parameters())
    softmax = nn.Softmax()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network_params, lr=0.001)

    for epoch in range(100): # an epoch is a loop over the entire dataset
        for i in range(len(data)):
            network.zero_grad() # PyTorch remembers gradients. We can forget them now, because we are starting a new sentence
            adj_mat = network(data[i])
            # so the softmax goes over rows but we need it over columns. supper annoying. hence the transposition
            pred = torch.t(softmax(torch.t(adj_mat)))

            # THIS IS A FAKE GOLD TREE. GET THE REAL ONES
            gold_tree = torch.LongTensor(range(len(adj_mat)))
            # Gold tree will be like
            # [[0 0 1 0]
            #  [1 1 0 0]
            #  [0 0 0 0]
            #  [0 0 0 1]]
            # transform this to a longtensor of sentence length with correct 'classifications'
            # the above should become
            # [1, 1, 0, 3]

            target = torch.LongTensor(gold_tree.size())
            target = Variable(gold_tree, requires_grad=False)

            #input should be (batch_size, n_label) and target should be (batch_size) with values in [0, n_label-1].
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step() # update per epoch or per sentence, or some batch-size inbetween?

            # check here if the parameters are indeed being updated
            # worringly, they are not
            network_params = list(network.parameters())
            print(network_params)

class Network(nn.Module):
    def __init__(self, w2i, p2i, pretrained_word_embeddings, pretrained_pos_embeddings, len_word_embed, len_pos_embed, len_feature_vec=20, len_hidden_dimension=120):
        super(Network, self).__init__()
        self.len_word_embed = len_word_embed
        self.len_pos_embed = len_pos_embed
        self.len_data_vec = len_word_embed + len_pos_embed
        self.len_feature_vec = len_feature_vec
        self.hidden_dimension = len_hidden_dimension

        self.w2i = w2i
        self.p2i = p2i

        # trainable parameters
        self.word_embeddings = torch.nn.Embedding(len(pretrained_word_embeddings), len_word_embed)
        self.word_embeddings.weight = torch.nn.Parameter(pretrained_word_embeddings)
        self.pos_embeddings = torch.nn.Embedding(len(pretrained_pos_embeddings), len_pos_embed)
        self.pos_embeddings.weight = torch.nn.Parameter(pretrained_pos_embeddings)

        self.BiLSTM = torch.nn.LSTM(self.len_data_vec, self.len_data_vec, bidirectional=True)

        self.MLP_head_layer1 = torch.nn.Linear(self.len_data_vec*2, len_hidden_dimension)
        self.MLP_head_layer2 = torch.nn.Linear(len_hidden_dimension, len_feature_vec)
        self.MLP_dep_layer1 = torch.nn.Linear(self.len_data_vec*2, len_hidden_dimension)
        self.MLP_dep_layer2 = torch.nn.Linear(len_hidden_dimension, len_feature_vec)

        self.U_1 = nn.Parameter(torch.randn(len_feature_vec, len_feature_vec))
        self.u_2 = nn.Parameter(torch.randn(1, len_feature_vec))

    def MLP_head(self, r):
        hidden = self.MLP_head_layer1(torch.sigmoid(r)).clamp(min=0)
        h = self.MLP_head_layer2(hidden)
        return h

    def MLP_dep(self, r):
        hidden = self.MLP_dep_layer1(torch.sigmoid(r)).clamp(min=0)
        h = self.MLP_dep_layer2(hidden)
        return h

    def forward(self, sentence):
        x = Variable(torch.FloatTensor(len(sentence['words']), 1, self.len_data_vec))
        for i, word in enumerate(sentence['words']):
            y = Variable(torch.LongTensor([self.w2i[word['form']]]))
            word_embedding = self.word_embeddings(y) # row vect
            z = Variable(torch.LongTensor([self.p2i[word['xpostag']]]))
            pos_embedding = self.pos_embeddings(z) # row vect
            x[i, 0, :] = torch.cat((torch.t(word_embedding), torch.t(pos_embedding)), 0)

        hidden = (autograd.Variable(torch.randn(2, 1, self.len_data_vec)), autograd.Variable(torch.randn(2 ,1 ,self.len_data_vec)))
        r, _ = self.BiLSTM(x, hidden)

        adj_matrix = Variable(torch.FloatTensor(len(sentence['words']), len(sentence['words'])))
        for i, vec_h in enumerate(r[:]):
            h_head = self.MLP_head(vec_h)
            for j, vec_d in enumerate(r[:]):
                h_dep = self.MLP_dep(vec_d)
                adj_matrix[i,j] = torch.mm(h_head, torch.mm(self.U_1, torch.t(h_dep))) + torch.mm(h_head, torch.t(self.u_2))
        return adj_matrix

if __name__ == '__main__':
    train()
