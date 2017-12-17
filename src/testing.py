import json
import os
import torch

import numpy as np

import torch.autograd as autograd
import torch.nn.functional as F

from torch.autograd import Variable

from utils import convert_sentence_to_adjacency_matrix, adjacency_matrix_to_tensor
from MST_FINAL import get_edmonds, edges_to_single

from main import Network


def UAS_score(pred_tensor, gold_tensor):
    '''
    Takes in two tensors representing dependency trees
    and returns their unlabelled attachement score
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
    length = len(pred_tensor)
    count = 0
    for i in range(length):
        if pred_tensor[i] == gold_tensor[i]:
            if pred_labels[i].data[0] == gold_labels[i]:
                count += 1
    return count, length


def edges_to_tensor(edges):
    '''
    this function is used to convert the output of the mst into a tensor
    INPUT: a list of edges (ordered pairs) of an mst
    OUTPUT: a torch tensor
    '''
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
    pred_edges = get_edmonds(adj_mat.T, 0)
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

    adj_mat, predicted_labels = network(sequence_var)

    pred_labels = [0]*len(predicted_labels)
    for i in range(len(predicted_labels)):
        _, pred_labels[i] = predicted_labels[i].max(0)

    pred_tensor = torch.LongTensor(pred_single)
    uas_score = UAS_score(pred_tensor, gold_tree)
    las_score = LAS_score(pred_tensor, gold_tree, pred_labels, gold_labels)

    return uas_score, las_score


def test(model, l2i, filepath):
    '''
    Input:
        - filepath: path of test dataset in json format
    '''
    if filepath is None:
        print('Provide a datset')
        retrun

    test_data = json.load(open(filepath, 'r'))

    uas_scores = 0
    las_scores = 0
    arc_count = 0

    for sentence in test_data:
        gold_labels = []
        try:
            for j, word in enumerate(sentence['words']):
                gold_labels.append(l2i[word['deprel']])

            uas, las = score_sentence(model, sentence, gold_labels)
            uas_scores += uas[0]
            las_scores += las[0]
            arc_count += uas[1]
        except Exception as e:
            print(e)

    return (uas_scores/arc_count)*100, (las_scores/arc_count)*100


if __name__ == '__main__':
    '''
    To change the test set, use one of the directories:
    1. english_full
    2. english_short
    3. hindi_full
    4. hindi_short

    Change the path of the labels accordingly
    '''
    # Change directory path here to test for other datasets
    dataset = 'english_short'

    try:
        print('************ DATASET: {} ************'.format(dataset))
        network_model = torch.load('../data/' + dataset + '/latest_weights')
        l2i = json.load(open('../data/' + dataset + '/labels.json', 'r'))

        print('Weights loaded, Begin predicting on the test set')
        score = test(network_model, l2i, '../data/' + dataset + '/test_unk.json')
        print('SCORE: ', score)
    except:
        print('------\nExtract latest_weights zip file: /data/' + dataset + '/latest_weights.zip\n------')
