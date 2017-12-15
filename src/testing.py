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

from main import Network


#takes in two tensors representing dependency trees
#and returns their unlabelled attachement score
def UAS_score(pred_tensor, gold_tensor):
    '''
    Computes UAS score for the predicted tensor
    '''
    length = len(pred_tensor)
    count = 0
    for i in range(length):
        if pred_tensor[i] == gold_tensor[i]:
            count += 1
    return count, length

def LAS_score(pred_tensor, gold_tensor, pred_labels, gold_labels):
    '''
    Computes LAS score for the predicted tensor
    '''
    # if torch.cuda.is_available():
    #     gold_labels = gold_labels.cpu()
    #
    # gold_labels = gold_labels.data.numpy()

    length = len(pred_tensor)
    count = 0
    for i in range(length):
        if pred_tensor[i] == gold_tensor[i]:
            # print("----")
            # print(pred_labels[i].data[0])
            # print(gold_labels[i])
            # print("----")
            if pred_labels[i].data[0] == gold_labels[i]:
                count += 1
    return count, length

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

def score_sentence(network, sentence, gold_labels):
    gold_mat = convert_sentence_to_adjacency_matrix(sentence)
    gold_tree = adjacency_matrix_to_tensor(gold_mat)
    target = Variable(gold_tree, requires_grad=False)

    if torch.cuda.is_available():
        sequence = torch.cuda.LongTensor(len(sentence['words']), 2)
    else:
        sequence = torch.LongTensor(len(sentence['words']), 2)

    for i, word in enumerate(sentence['words']):
        sequence[i,0] = network.w2i[word['form']]
        sequence[i,1] = network.p2i[word['xpostag']]
    sequence_var = Variable(sequence)

    adj_mat, _ = network(sequence_var)

    adj_mat = F.softmax(torch.t(adj_mat))

    if torch.cuda.is_available():
        adj_mat = adj_mat.cpu()

    adj_mat = adj_mat.data.numpy()
    pred_edges = get_edmonds(adj_mat, 0)
    pred_single = edges_to_single(pred_edges)

    # For labels
    if torch.cuda.is_available():
        sequence_with_label = torch.cuda.LongTensor(len(sentence['words']), 3)
    else:
        sequence_with_label = torch.LongTensor(len(sentence['words']), 3)

    for i, word in enumerate(sentence['words']):
        sequence_with_label[i,0] = network.w2i[word['form']]
        sequence_with_label[i,1] = network.p2i[word['xpostag']]
        sequence_with_label[i,2] = pred_single[i]
    sequence_var = Variable(sequence_with_label)

    _, predicted_labels = network(sequence_var)
    pred_labels = [0]*len(predicted_labels)
    for i in range(len(predicted_labels)):

        # print(len(predicted_labels[i]))

        pred_labels[i] = torch.max(predicted_labels[i])


    pred_tensor = torch.LongTensor(pred_single)
    uas_score = UAS_score(pred_tensor, gold_tree)
    las_score = LAS_score(pred_tensor, gold_tree, pred_labels, gold_labels)

    return uas_score, las_score

# sentence = training_data[0]
#print(test_sentence(sentence))


def test(model, l2i, filepath):
    '''
    Input:
        - filepath: path of test dataset in json format
    '''

    # test_data = json.load(open(filepath, 'r'))
    if filepath is None:
        test_data = json.load(open('../data/toy_data.json', 'r'))

    uas_scores = 0
    las_scores = 0
    arc_count = 0

    for sentence in test_data:
        gold_labels = []
        for j, word in enumerate(sentence['words']):
            gold_labels.append(l2i[word['deprel']])

        uas, las = score_sentence(model, sentence, gold_labels)
        uas_scores += uas[0]
        las_scores += las[0]
        arc_count += uas[1]

    return (uas_scores/arc_count)*100, (las_scores/arc_count)*100


if __name__ == '__main__':

    network_model = torch.load('../weights/latest_weights')

    l2i = json.load(open('../data/labels.json', 'r'))

    score = test(network_model, l2i, None)

    print(score)
