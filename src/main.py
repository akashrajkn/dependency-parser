import json
import os
import copy
import MST
import time
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from time import gmtime, strftime
from gensim.models import Word2Vec
from torch.autograd import Variable
from utils import convert_sentence_to_adjacency_matrix, adjacency_matrix_to_tensor

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

    w2i = {word: idx for idx, word in enumerate(all_words)}
    p2i = {pos: idx for idx, pos in enumerate(all_pos)}

    # pre-train word and pos embeddings. These will be starting points for our learnable embeddings
    word_embeddings_gensim = Word2Vec(corpus_words, size=len_word_embed, window=5, min_count=1, workers=8)
    pos_embeddings_gensim = Word2Vec(corpus_pos, size=len_pos_embed, window=5, min_count=1, workers=8)

    # initialise the embeddings. The tensors are still empty
    pretrained_word_embeddings = torch.FloatTensor(max(w2i.values())+1, len_word_embed)
    pretrained_pos_embeddings = torch.FloatTensor(max(p2i.values())+1, len_pos_embed)

    # fill the tensors with the pre-trained embeddings
    for word in w2i.keys():
        idx = w2i[word]
        pretrained_word_embeddings[idx, :] = torch.from_numpy(word_embeddings_gensim[word])
    for pos in p2i.keys():
        idx = p2i[pos]
        pretrained_pos_embeddings[idx, :] = torch.from_numpy(pos_embeddings_gensim[pos])
    return w2i, p2i, pretrained_word_embeddings, pretrained_pos_embeddings

def train(show=True):
    # corpus_words in Format: [['I', 'like', 'custard'],...]
    # corpus_pos in Format: [['NN', 'VB', 'PRN'],...]
    current_path = os.path.dirname(__file__)
    #filepath = current_path + '../data/en-ud-train-short.json'
    filepath = current_path + '../data/toy_data.json'
    if filepath is None:
        filepath = current_path + '../data/en-ud-train.json'
    data = json.load(open(filepath, 'r'))
    len_word_embed = 100
    len_pos_embed = 20

    w2i, p2i, pwe, ppe = pretrain_word_embeddings(data, len_word_embed, len_pos_embed)

    network = Network(w2i, p2i, pwe, ppe, len_word_embed, len_pos_embed)
    if torch.cuda.is_available():
        network.cuda()
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    wd = 1e-6
    betas=(0.9, 0.9)
    optimizer = optim.Adam(network.parameters(), lr=lr, betas=betas, weight_decay=wd)

    start = time.time()

    losses = []

    current_date_and_time = strftime("%Y-%m-%d-%H:%M:%S", gmtime()) # current time is 1hr off?
    new_dir = "weights/weights_trained_from_"+current_date_and_time
    os.mkdir(new_dir)

    with open( new_dir+"/log_file.txt", "w") as output_file:
        output_file.write("started writing at "+current_date_and_time+'\n')
        output_file.write("lr = "+str(lr)+'\n')
        output_file.write("weight decay = "+str(wd)+'\n')

    print("size of the dataset:", len(data))
    for epoch in range(200): # an epoch is a loop over the entire dataset
        for i in range(len(data)):
            network.zero_grad() # PyTorch remembers gradients. We can forget them now, because we are starting a new sentence

            # prepare target
            gold_mat = convert_sentence_to_adjacency_matrix(data[i])
            gold_tree = adjacency_matrix_to_tensor(gold_mat)
            target = Variable(gold_tree, requires_grad=False)

            # prepare input
            sequence = torch.LongTensor(len(data[i]['words']), 2)
            for i, word in enumerate(data[i]['words']):
                sequence[i,0] = w2i[word['form']]
                sequence[i,1] = p2i[word['xpostag']]

            sequence_var = Variable(sequence)

            # prepare GPU
            if torch.cuda.is_available():
                target = target.cuda()
                sequence_var = sequence_var.cuda()

            # forward, backward, update
            adj_mat = network(sequence_var)
            pred = torch.t(adj_mat) # nn.CrossEntropyLoss() wants the classes in the second dimension
            loss = criterion(pred, target)
            losses.append(loss.data.cpu().numpy()[0])
            loss.backward()
            optimizer.step()

        # print an update
        current_date_and_time = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
        saved_trainer = copy.deepcopy(network)
        with open( new_dir+"/latest_weights.pkl", "wb") as output_file:
            pickle.dump(saved_trainer, output_file)
        with open( new_dir+"/log_file.txt", "a") as output_file:
            output_file.write("backed up made at"+current_date_and_time)
            output_file.write("loss after epoch "+str(epoch)+": "+str(losses[-1])+'\n')
        if show:
            print('-------')
            print("latest parameter backup at ", current_date_and_time)
            print("epoch {} loss {:.4f}".format(epoch, losses[-1]))

    end = time.time()
    if show:
        print("execution took ", end - start, " seconds")
        plt.plot(losses)
        plt.show()


class Network(nn.Module):
    def __init__(self, w2i, p2i, pretrained_word_embeddings, pretrained_pos_embeddings, len_word_embed, len_pos_embed, len_feature_vec=20, len_hidden_dimension=500, lstm_hidden_size=400):
        super(Network, self).__init__()
        self.len_word_embed = len_word_embed
        self.len_pos_embed = len_pos_embed
        self.len_data_vec = len_word_embed + len_pos_embed
        self.len_feature_vec = len_feature_vec
        self.hidden_dimension = len_hidden_dimension
        self.lstm_hidden_size = lstm_hidden_size

        self.w2i = w2i
        self.p2i = p2i

        # trainable parameters
        self.word_embeddings = torch.nn.Embedding(len(pretrained_word_embeddings), len_word_embed)
        self.word_embeddings.weight = torch.nn.Parameter(pretrained_word_embeddings)
        self.pos_embeddings = torch.nn.Embedding(len(pretrained_pos_embeddings), len_pos_embed)
        self.pos_embeddings.weight = torch.nn.Parameter(pretrained_pos_embeddings)

        self.BiLSTM = torch.nn.LSTM(input_size=self.len_data_vec, hidden_size=self.lstm_hidden_size, num_layers = 3, dropout=.33, bidirectional=True)

        self.MLP_head_layer1 = torch.nn.Linear(self.len_data_vec*2, len_hidden_dimension)
        self.MLP_head_layer2 = torch.nn.Linear(len_hidden_dimension, len_feature_vec)
        self.MLP_dep_layer1 = torch.nn.Linear(self.len_data_vec*2, len_hidden_dimension)
        self.MLP_dep_layer2 = torch.nn.Linear(len_hidden_dimension, len_feature_vec)

        self.U_1 = nn.Parameter(torch.randn(len_feature_vec, len_feature_vec))
        self.u_2 = nn.Parameter(torch.randn(1, len_feature_vec))

    def MLP_head(self, r):
        hidden = self.MLP_head_layer1(F.relu(r)).clamp(min=0)
        h = self.MLP_head_layer2(hidden)
        return h

    def MLP_dep(self, r):
        hidden = self.MLP_dep_layer1(F.relu(r)).clamp(min=0)
        h = self.MLP_dep_layer2(hidden)
        return h

    def forward(self, sequence):
        seq_len = len(sequence[0])
        word_sequence = sequence[:,0]
        pos_sequence = sequence[:,1]

        word_embeddings = self.word_embeddings(word_sequence)
        pos_embeddings = self.pos_embeddings(pos_sequence)

        x = torch.cat((word_embeddings, pos_embeddings), 1)
        x = x[:, None, :] # add an empty y-dimension, because that's how LSTM takes its input

        hidden_init_1 = torch.zeros(6, 1, self.lstm_hidden_size) # initialse random instead of zeros?
        hidden_init_2 = torch.zeros(6, 1, self.lstm_hidden_size)
        if torch.cuda.is_available:
            hidden_init_1 = hidden_init_1.cuda()
            hidden_init_2 = hidden_init_2.cuda()
        hidden = (autograd.Variable(hidden_init_1), autograd.Variable(hidden_init_2))

        r, _ = self.BiLSTM(x, hidden)

        adj_matrix = autograd.Variable(torch.FloatTensor(seq_len, seq_len))
        h_head = torch.squeeze(self.MLP_head(r))
        h_dep = torch.squeeze(self.MLP_dep(r))
        adj_matrix = h_head @ self.U_1 @ torch.t(h_dep) + h_head @ torch.t(self.u_2)

        return adj_matrix

if __name__ == '__main__':
    train()
