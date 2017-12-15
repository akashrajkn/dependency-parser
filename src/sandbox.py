import torch
import json
import copy
import os
import MST
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from gensim.models import Word2Vec
from torch.autograd import Variable

d = {}
d['hey'] = 1
d['yo'] = 2
if 'hey' not in d:
    d['hey'] = 3

print(d)

#
# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#         self.U = nn.Parameter(torch.randn(3, 5))
#         self.MLP_layer_1 = torch.nn.Linear(3, 4)
#         self.MLP_layer_2 = torch.nn.Linear(4,3)
#
#     def MLP(self, input):
#         hidden = self.MLP_layer_1(torch.sigmoid(input)).clamp(min=0)
#         output = self.MLP_layer_2(hidden)
#         return output
#
#     def forward(self, inp):
#         step1 = torch.t(3 + torch.mm(self.U, inp))
#         output = self.MLP(step1)
#         return output
#
# def mse_loss(input, target):
#     return torch.sum((input - target) ** 2)
#
# network = Network()
# print(torch.cuda.is_available())
# network.cuda()
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(network.parameters(), lr=0.001)
#
# inputs = Variable(torch.randn(5,1))
# inputs = inputs.cuda()
# targets = Variable(torch.randn(3,1), requires_grad = False)
# targets = targets.cuda()
#
# for i in range(10):
#     network.zero_grad()
#     outputs = network(inputs)
#     loss = mse_loss(outputs, torch.t(targets))
#     # print(loss)
#     loss.backward()
#     optimizer.step()
#     print(loss.data)
