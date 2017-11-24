#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 18:02:24 2017

@author: jackharding
"""

import numpy as np
from collections import defaultdict

test_matrix1 = np.random.rand(6, 6)

#print(test_matrix1)

def get_nodes(adjacency_matrix):
    nodes_list = []
    for node in range(adjacency_matrix.shape[0]):
        nodes_list.append(node)
    return nodes_list

#print(get_nodes(test_matrix1))
    
def preprocess(matrix, root):
    nodes_list = get_nodes(matrix)
    for node in nodes_list:
        matrix[node][node] = 0.0
        matrix[node][root] = 0.0
    return matrix

test_matrix = preprocess(test_matrix1, 0)
#print(test_matrix)
    
def entering_weights(adjacency_matrix, node):
    return adjacency_matrix[:, node]
    
#print(entering_weights(test_matrix, 3))

def edges(graph):
    edges = []
    for i in range(graph.shape[0]):
        for j in range(graph.shape[0]):
            if graph[i][j] != 0:
                edges.append((i, j))
    return(edges)
    
#print(edges(test_matrix))


def biggest_edge(graph):
    nodes_list = get_nodes(graph)
    biggest_edge_dict = {}
    for node in nodes_list:
        biggest_edge_dict[node] = max(entering_weights(graph, node))
    return biggest_edge_dict

#print(biggest_edge(test_matrix))

def biggest_edge_index(graph):
    nodes_list = get_nodes(graph)
    biggest_edge_dict = biggest_edge(graph)
    biggest_edge_index_dict = {}
    for node1 in nodes_list:
        for node2 in nodes_list:
            if graph[node1][node2] == biggest_edge_dict[node2]:
                biggest_edge_index_dict[node2] = node1
    return biggest_edge_index_dict

#print(biggest_edge_index(test_matrix))

#creates the graph with the largest edge to each node
def try_biggest(graph, root):
    
    output = np.zeros(graph.shape)
    nodes_list = get_nodes(graph)
    biggest_edge_index_dict = biggest_edge_index(graph)
    for node1 in nodes_list:
        for node2 in nodes_list:
            if biggest_edge_index_dict[node1] == node2:
                output[node2][node1] = graph[node2][node1]
    return output

#print(try_biggest(test_matrix, 0))

#returns True if there is a cycle in the graph,False otherwise
#adapted from code I found on Stackexchange
def cyclic(graph):
    nodes_list = get_nodes(graph)
    
    path = set()
    visited = set()

    def visit(node):
        if node in visited:
            return False
        visited.add(node)
        path.add(node)
        neighbours = []
        for node1 in nodes_list:
            if graph[node][node1] != 0.0:
                neighbours.append(node1)
        for neighbour in neighbours:
            if neighbour in path or visit(neighbour):
                return True
        path.remove(node)
        return False

    return any(visit(node) for node in nodes_list)

#print(cyclic(try_biggest(test_matrix)))

def is_mst(graph_attempt):
    return not cyclic(graph_attempt)

#print(is_mst(try_biggest(test_matrix)))


#returns a list of all cycles in the graph (e.g. [1, 2, 3, 1] is a cycle)
#if there are no cycles, returns the empty list
def find_cycles(graph):
    nodes_list = get_nodes(graph)
    cycles_list = []
    
    path = []
    visited = []

    def visit(node):
        if node in visited:
            return
        visited.append(node)
        path.append(node)
        neighbours = []
        for node1 in nodes_list:
            if graph[node][node1] != 0.0:
                neighbours.append(node1)
        for neighbour in neighbours:
            if neighbour in path or visit(neighbour):
                cycles_list.append(path.copy())
                cycles_list[-1].append(neighbour)
        path.remove(node)
        return
    
    for node in nodes_list:
        visit(node)
    
    return cycles_list

#print(find_cycles(try_biggest(test_matrix, 0)))

def sum_path(graph, pair_list):
    sum_path = 0.0
    for (node1, node2) in pair_list:
        sum_path += graph[node1][node2]
    return sum_path

def edmonds(graph, root):
    print("ORIGINAL GRAPH")
    print(graph)
    conversion_dictionaries_list = []
    new_graph = preprocess(graph, root)
    graph_attempt = try_biggest(graph, root)
    
    while not is_mst(new_graph):
        conversion_dict = defaultdict()
        nodes_list = get_nodes(new_graph)
        graph_attempt = try_biggest(new_graph, root)
        
#        print("GRAPH ATTEMPT")
#        print(graph_attempt)
        
        if is_mst(graph_attempt):
            new_graph = graph_attempt
            break
        
        cycle1 = find_cycles(graph_attempt)[0]
#        print("CYCLE", cycle1)
        
        new_graph_nodes = []
        contracted_node = "contracted node"    
        for node in nodes_list:
            if node not in set(cycle1):
                new_graph_nodes.append(node)

#        print("NEW GRAPH NODES", new_graph_nodes)
        
        updated_indices = {}
        for i in range(len(new_graph_nodes)):
            updated_indices[new_graph_nodes[i]] = i
        
        updated_indices[contracted_node] = len(new_graph_nodes)

        new_graph = np.zeros((len(new_graph_nodes) + 1, len(new_graph_nodes) + 1))
        
#            edges not in cycle
        for node1 in new_graph_nodes:
            for node2 in new_graph_nodes:
                new_graph[updated_indices[node1]][updated_indices[node2]] = graph[node1][node2]
        
#            outward edges
        for node2 in new_graph_nodes:
            max_from_cycle = [0.0, ("dummy", "dummy")]
            for node1 in set(cycle1):
                if graph[node1][node2] > max_from_cycle[0]:
                    max_from_cycle[0] = graph[node1][node2]
                    max_from_cycle[1] = (node1, node2)
            new_graph[updated_indices[contracted_node]][updated_indices[node2]] = max_from_cycle[0]
            if max_from_cycle[1] != ("dummy", "dummy"):
                conversion_dict[(updated_indices[contracted_node], updated_indices[node2])] = [max_from_cycle[1]]
          
#            inward edges
        edges_in_cycle = list(zip(cycle1, cycle1[1:]))
#        print("EDGES IN CYCLE", edges_in_cycle)
        
        for node in new_graph_nodes:
            for i in range(len(edges_in_cycle)):
                trial_list = edges_in_cycle.copy()
                if i == len(edges_in_cycle) - 1:
                    trial_list[i] = (node, trial_list[0][0])
                else:
                    trial_list[i] = (node, trial_list[i+1][0])

                max_path_score = 0.0
                max_path = []
                for path in trial_list:
                    if sum_path(graph, trial_list) > max_path_score:
                        max_path_score = sum_path(graph, trial_list)
                        max_path = trial_list
            
            conversion_dict[(updated_indices[node], updated_indices[contracted_node])] = max_path
            new_graph[updated_indices[node]][updated_indices[contracted_node]] = max_path_score            
        
        for node1 in new_graph_nodes:
            for node2 in new_graph_nodes:
                conversion_dict[(updated_indices[node1],updated_indices[node2])] = [(node1, node2)]

        conversion_dictionaries_list.append(conversion_dict)
#        print("NEW GRAPH")
#        print(new_graph)
    
#    print("CONVERSION", conversion_dictionaries_list)

    list_of_nodes = []
    for node1 in range(new_graph.shape[0]):
        for node2 in range(new_graph.shape[0]):
            if new_graph[node1][node2] > 0.0:
                list_of_nodes.append((node1, node2))

    while len(conversion_dictionaries_list) > 0:        
#        print("LIST OF NODES", list_of_nodes)
        new_list = []
        for pair in list_of_nodes:
#            print(conversion_dictionaries_list[-1][pair])
            for node_pair in conversion_dictionaries_list[-1][pair]:
                new_list.append(node_pair)
            
        list_of_nodes = new_list
        del conversion_dictionaries_list[-1]
    
    print("LIST OF NODES", list_of_nodes)
    
    output_mst = np.zeros(graph.shape)
    for (node1, node2) in list_of_nodes:
        output_mst[node1][node2] = graph[node1][node2]
    
    return output_mst
     

print(edmonds(test_matrix1, 0))