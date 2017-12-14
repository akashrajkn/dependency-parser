#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 17:39:46 2017

@author: jackharding
"""
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
from MST_FINAL import get_edmonds, edges_to_single

current_path = os.path.dirname(__file__)
filepath = current_path + '/../data/toy_data.json'
#if filepath is None:
#filepath = current_path + '/../data/en-ud-train.json'
training_data = json.load(open(filepath, 'r'))

len_word_embed = 100
len_pos_embed = 20
lr = 0.001
weight_decay = 1e-6
betas = (0.9, 0.9)

w2i, p2i, pwe, ppe = pretrain_word_embeddings(training_data, len_word_embed, len_pos_embed)
network = Network(w2i, p2i, pwe, ppe, len_word_embed, len_pos_embed)
softmax = nn.Softmax()
#takes in two tensors representing dependency trees
#and returns their unlabelled attachement score
def UAS_score(pred_tensor, gold_tensor):
    length = len(pred_tensor)
    count = 0
    for i in range(length):
        if pred_tensor[i] == gold_tensor[i]:
            count += 1
    return (count/length)*100

#test_tensor = torch.LongTensor([0, 0, 1, 2, 3, 4])
#gold_tensor = torch.LongTensor([0, 2, 1, 4, 5, 4])
#print(UAS_score(test_tensor, gold_tensor))
  
def edges_to_tensor(edges):
    #this function is used to convert the output of the mst into a tensor
    #INPUT: a list of edges (ordered pairs) of an mst
    #OUTPUT: a torch tensor
    output = [0] * (len(edges) + 1)
    for edge in edges:
        output[edge[1]] = edge[0]
    output1 = torch.LongTensor(output)
    return output1

def test(sentence):
    gold_mat = convert_sentence_to_adjacency_matrix(sentence)
    gold_tree = adjacency_matrix_to_tensor(gold_mat)
    target = Variable(gold_tree, requires_grad=False)
    
    sequence = torch.LongTensor(len(sentence['words']), 2)
    for i, word in enumerate(sentence['words']):
        sequence[i,0] = w2i[word['form']]
        sequence[i,1] = p2i[word['xpostag']]
    sequence_var = Variable(sequence)
        
    adj_mat = network(sequence_var)
    adj_mat = softmax(torch.t(adj_mat))
    adj_mat = adj_mat.data.numpy()
    pred_edges = get_edmonds(adj_mat, 0)
    pred_single = edges_to_single(pred_edges)
    pred_tensor = torch.LongTensor(pred_single)
    score_for_sentence = UAS_score(pred_tensor, gold_tree)
    return score_for_sentence

sentence = training_data[0]
#print(test(sentence))
