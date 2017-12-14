#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 17:39:46 2017

@author: jackharding
"""
import torch

#takes in two tensors representing dependency trees
#and returns their unlabelled attachement score
def UAS_score(pred_tensor, gold_tensor):
    count = 0
    for i in range(len(pred_tensor)):
        if pred_tensor[i] == gold_tensor[i]:
            count += 1
    return count

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

#test_edges = [(0, 2), (3, 5), (2, 3), (3, 4), (2, 1)]
#print(edges_to_tensor(test_edges))
