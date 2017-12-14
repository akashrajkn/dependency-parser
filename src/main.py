import json
import os
import copy
import torch
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

from time import gmtime, strftime

from torch.autograd import Variable

from gensim.models import Word2Vec
from utils import convert_sentence_to_adjacency_matrix, adjacency_matrix_to_tensor


def pretrain_word_embeddings(data, len_word_embed, len_pos_embed):
    '''
    corpus_words in the following shape:
    [ [some sentence]
      [some other sentence] ]

    all_words in the shape:
    [ some sentence some other sentence]
    '''

    corpus_words = []
    corpus_pos = []
    all_words = []
    all_pos = []

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


def save_parameters(network, path='/latest_weights.pkl'):
    with open(path, 'wb') as output_file:
        pickle.dump(network, output_file)


def train(show=True, save=False):
    '''
    corpus_words in Format: [['I', 'like', 'custard'],...]
    corpus_pos in Format: [['NN', 'VB', 'PRN'],...]
    '''
    current_path = os.path.dirname(os.path.realpath(__file__))

    filepath_dataset = current_path + '/../data/toy_data.json'
    #filepath_dataset = current_path + '/../data/en-ud-train-short.json'
    if filepath_dataset is None:
        filepath_dataset = current_path + '/../data/en-ud-train.json'

    data = json.load(open(filepath_dataset, 'r'))

    len_word_embed = 100
    len_pos_embed = 20
    lr = 0.001
    weight_decay = 1e-6
    betas = (0.9, 0.9)

    w2i, p2i, pwe, ppe = pretrain_word_embeddings(data, len_word_embed, len_pos_embed)

    network = Network(w2i, p2i, pwe, ppe, len_word_embed, len_pos_embed)

    if torch.cuda.is_available():
        network.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    start = time.time()
    # Save trained weights
    if save:
        current_date_and_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
        new_dir = current_path + '/../weights/trained_from_' + current_date_and_time
        log_file = new_dir + '/log_file.txt'
        os.mkdir(new_dir)

        with open(log_file, 'w') as output_file:
            output_file.write('started writing at ' + current_date_and_time + '\n' +
                              'dataset = ' + str(filepath_dataset) + '\n' +
                              'lr = ' + str(lr) + '\n' +
                              'weight decay = ' + str(weight_decay) + '\n' +
                              'betas = ' + str(betas) + '\n' +
                              'len_word_embed = ' + str(network.len_word_embed) + '\n' +
                              'len_pos_embed = ' + str(network.len_pos_embed) + '\n' +
                              'len_feature_vec = ' + str(network.len_feature_vec) + '\n' +
                              'lstm_hidden_size = ' + str(network.lstm_hidden_size) + '\n' +
                              'mlp_arc_hidden_size = ' + str(network.mlp_arc_hidden_size) + '\n')

    n_data = len(data)
    losses_per_data = []
    losses_per_epoch = []
    if show:
        print('size of the dataset: ', n_data)
    # an epoch is a loop over the entire dataset
    for epoch in range(100):
        for i in range(len(data)):
            network.zero_grad()  # PyTorch remembers gradients. We can forget them now, because we are starting a new sentence

            # prepare target
            gold_mat = convert_sentence_to_adjacency_matrix(data[i])
            gold_tree = adjacency_matrix_to_tensor(gold_mat)
            target = Variable(gold_tree, requires_grad=False)

            # prepare input
            sequence = torch.LongTensor(len(data[i]['words']), 2)
            for i, word in enumerate(data[i]['words']):
                sequence[i,0] = w2i[word['form']]
                sequence[i,1] = p2i[word['xpostag']]
                #sequence[i,2] = target[0][i]

            sequence_var = Variable(sequence)

            # prepare GPU
            if torch.cuda.is_available():
                target = target.cuda()
                sequence_var = sequence_var.cuda()

            adj_mat = network(sequence_var)
            pred = torch.t(adj_mat)  # nn.CrossEntropyLoss() wants the classes in the second dimension
            arc_loss = criterion(pred, target)
            losses_per_data.append(arc_loss.data.cpu().numpy()[0])
            arc_loss.backward()

            # adj_mat, labels_pred = network(sequence_var)
            # label_loss = label_criterion(labels_pred, labels_target)
            # label_loss.backward()

            optimizer.step()

            # clip to prevent exploding gradient problem?
            # l = [ x.grad.data for x in network.parameters() if x.grad is not None]
            # print( np.mean(l) )
            # norm = torch.nn.utils.clip_grad_norm(network.parameters(), max_norm=0)
            # params = list(filter(lambda p: p.grad is not None, network.parameters()))
            # print(norm)
        losses_per_epoch.append(np.mean(losses_per_data[-n_data:]))

        # print an update
        current_date_and_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())

        if save:
            with open(log_file, 'a') as output_file:
                output_file.write(  'latest backup at ' + current_date_and_time + '\n' +
                                    'loss after epoch ' + str(epoch) + ': ' + str(losses_per_data[-1]) + '\n')
            torch.save(network.state_dict(), new_dir + '/latest_weights')
            plt.ylim(0,5)
            plt.subplot(211)
            plt.title('loss per datapoint(top) and epoch(bottom)')
            plt.plot(losses_per_data)
            plt.subplot(212)
            plt.plot(losses_per_epoch)
            plt.savefig(new_dir + '/convergence.png')

        if show:
            print('-'*10)
            print('latest backup at ', current_date_and_time)
            print('epoch {} loss {:.4f}'.format(epoch, losses_per_epoch[-1]))

    end = time.time()

    if show:
        print('execution took ', end - start, ' seconds')
        plt.show()
    return

class Network(nn.Module):
    '''
    Define the Biaffine LSTM network
    '''
    def __init__(self, w2i, p2i, pretrained_word_embeddings, pretrained_pos_embeddings,
                 len_word_embed, len_pos_embed, len_feature_vec=20, lstm_hidden_size=400,
                 mlp_arc_hidden_size=500, mlp_label_hidden_size=200, n_label=20):

        super(Network, self).__init__()
        self.len_word_embed = len_word_embed
        self.len_pos_embed = len_pos_embed
        self.len_data_vec = len_word_embed + len_pos_embed
        self.len_feature_vec = len_feature_vec
        self.lstm_hidden_size = lstm_hidden_size
        self.mlp_arc_hidden_size = mlp_arc_hidden_size
        self.mlp_label_hidden_size = mlp_label_hidden_size
        self.n_label = n_label
        self.w2i = w2i
        self.p2i = p2i

        # trainable parameters
        self.word_embeddings = torch.nn.Embedding(len(pretrained_word_embeddings), len_word_embed)
        self.word_embeddings.weight = torch.nn.Parameter(pretrained_word_embeddings)
        self.pos_embeddings = torch.nn.Embedding(len(pretrained_pos_embeddings), len_pos_embed)
        self.pos_embeddings.weight = torch.nn.Parameter(pretrained_pos_embeddings)

        self.BiLSTM = torch.nn.LSTM(input_size=self.len_data_vec, hidden_size=self.lstm_hidden_size,
                                    num_layers = 3, dropout=.33, bidirectional=True)

        self.MLP_arc_head_layer1 = torch.nn.Linear(self.lstm_hidden_size * 2, mlp_arc_hidden_size)
        self.MLP_arc_head_layer2 = torch.nn.Linear(mlp_arc_hidden_size, len_feature_vec)
        self.MLP_arc_dep_layer1 = torch.nn.Linear(self.lstm_hidden_size * 2, mlp_arc_hidden_size)
        self.MLP_arc_dep_layer2 = torch.nn.Linear(mlp_arc_hidden_size, len_feature_vec)

        self.MLP_label_head_layer1 = torch.nn.Linear(self.lstm_hidden_size * 2, mlp_label_hidden_size)
        self.MLP_label_head_layer2 = torch.nn.Linear(mlp_label_hidden_size, len_feature_vec)
        self.MLP_label_dep_layer1 = torch.nn.Linear(self.lstm_hidden_size * 2, mlp_label_hidden_size)
        self.MLP_label_dep_layer2 = torch.nn.Linear(mlp_label_hidden_size, len_feature_vec)

        self.U_1 = nn.Parameter(torch.randn(len_feature_vec, len_feature_vec))
        self.u_2 = nn.Parameter(torch.randn(1, len_feature_vec))

    def MLP_arc_head(self, r):
        hidden = F.relu(self.MLP_arc_head_layer1(r))
        h = self.MLP_arc_head_layer2(hidden)
        return h

    def MLP_arc_dep(self, r):
        hidden = F.relu(self.MLP_arc_dep_layer1(r))
        h = self.MLP_arc_dep_layer2(hidden)
        return h

    def MLP_label_head(self, r):
        hidden = F.relu(self.MLP_label_head_layer1(r))
        h = self.MLP_label_head_layer2(hidden)
        return h

    def MLP_label_dep(self, r):
        hidden = F.relu(self.MLP_label_dep_layer1(r))
        h = self.MLP_label_dep_layer2(hidden)
        return h

    def forward(self, sequence):
        seq_len = len(sequence[0])
        word_sequence = sequence[:,0]
        pos_sequence = sequence[:,1]
        #gold_tree = sequence[:,2]

        word_embeddings = self.word_embeddings(word_sequence)
        pos_embeddings = self.pos_embeddings(pos_sequence)

        x = torch.cat((word_embeddings, pos_embeddings), 1)
        x = x[:, None, :]  # add an empty y-dimension, because that's how LSTM takes its input

        hidden_init_1 = torch.zeros(6, 1, self.lstm_hidden_size)  # initialse random instead of zeros?
        hidden_init_2 = torch.zeros(6, 1, self.lstm_hidden_size)

        if torch.cuda.is_available:
            hidden_init_1 = hidden_init_1.cuda()
            hidden_init_2 = hidden_init_2.cuda()

        hidden = (autograd.Variable(hidden_init_1), autograd.Variable(hidden_init_2))

        r, _ = self.BiLSTM(x, hidden)

        adj_matrix = autograd.Variable(torch.FloatTensor(seq_len, seq_len))
        h_arc_head = torch.squeeze(self.MLP_arc_head(r))
        h_arc_dep = torch.squeeze(self.MLP_arc_dep(r))
        adj_matrix = h_arc_head @ self.U_1 @ torch.t(h_arc_dep) + h_arc_head @ torch.t(self.u_2)

        #r[gold_tree] =

        return adj_matrix #, pred_labels

if __name__ == '__main__':
    train()
