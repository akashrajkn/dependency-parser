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

#takes in a graph and root and returns an mst (a list of arcs) rooted at the root
def edmonds(graph, root):    
    conversion_dictionaries_list = []
    graph1 = preprocess(graph, root)
    
    while True:
        
        graph_attempt = try_biggest(graph1, root)
#        print("GRAPH ATTEMPT")
#        print(graph_attempt)
        
        if is_mst(graph_attempt):
            graph1 = graph_attempt
            break
    
        conversion_dict = defaultdict()
        nodes_list = get_nodes(graph1)
        
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
                new_graph[updated_indices[node1]][updated_indices[node2]] = graph1[node1][node2]
        
#            outward edges
        for node2 in new_graph_nodes:
            max_from_cycle = [0.0, ("dummy", "dummy")]
            for node1 in set(cycle1):
                if graph1[node1][node2] > max_from_cycle[0]:
                    max_from_cycle[0] = graph1[node1][node2]
                    max_from_cycle[1] = (node1, node2)
            new_graph[updated_indices[contracted_node]][updated_indices[node2]] = max_from_cycle[0]
            if max_from_cycle[1] != ("dummy", "dummy"):
                conversion_dict[(updated_indices[contracted_node], updated_indices[node2])] = [max_from_cycle[1]]
          
#            inward edges
        nodes_in_cycle = list(set(cycle1))
        
        test_graph = np.zeros((len(nodes_in_cycle), len(nodes_in_cycle)))
        for node1 in range(len(nodes_in_cycle)):
            for node2 in range(len(nodes_in_cycle)):
                test_graph[node1][node2] = graph1[nodes_in_cycle[node1]][nodes_in_cycle[node2]]
        
#        print("TEST GRAPH")
#        print(test_graph)
        
        
        for node1 in new_graph_nodes:
            max_path_score = 0.0
            max_path = []
            for i in range(len(nodes_in_cycle)):
                path_from_node = edmonds(test_graph, i)
#                print("FIRST PATH", path_from_node)
                for j in range(len(path_from_node)):
                    path_from_node[j] = (nodes_in_cycle[path_from_node[j][0]], nodes_in_cycle[path_from_node[j][1]])
                path_from_node.append((node1, nodes_in_cycle[i]))
#                print("PATH_FROM_NODE", path_from_node)
                current_path_score = sum_path(graph1, path_from_node)
                if current_path_score > max_path_score:
                    max_path_score = current_path_score
                    max_path = path_from_node
            
#            print("MAX PATH", max_path)
            conversion_dict[(updated_indices[node1], updated_indices[contracted_node])] = max_path
            new_graph[updated_indices[node1]][updated_indices[contracted_node]] = max_path_score            

        for node1 in new_graph_nodes:
            for node2 in new_graph_nodes:
                conversion_dict[(updated_indices[node1],updated_indices[node2])] = [(node1, node2)]

        conversion_dictionaries_list.append(conversion_dict)
        
        graph1 = new_graph
#        print("NEW GRAPH")
#        print(graph1)
#    print("CONVERSION", conversion_dictionaries_list)

#backtracking to find edges used in original graph
    list_of_nodes = edges(graph1)
    if len(conversion_dictionaries_list) == 0:
#        print("HELLO")
        return list_of_nodes
   
    while len(conversion_dictionaries_list) > 0:        
#        print("LIST OF NODES", list_of_nodes)
        new_list = []
        for pair in list_of_nodes:
#            print(conversion_dictionaries_list[-1][pair])
            for node_pair in conversion_dictionaries_list[-1][pair]:
                new_list.append(node_pair)           
        list_of_nodes = new_list
        del conversion_dictionaries_list[-1]   
#    print("THIS IS THE RESULT")
    return list_of_nodes
#_________________________________________
#END OF EDMONDS ALGORITHM
#_________________________________________

#test_matrix2 = np.array([[0., 0., 0.05517177, 0., 0., 0.],
#                [0., 0., 0.39692219, 0.84662711, 0.00740889, 0.59934641],
#                [0., 0.40309773, 0., 0.3227141, 0.15214198, 0.35110729],
#                [0., 0.2147357, 0.53986018, 0., 0.14796666, 0.44470598],
#                [0., 0.39123883,  0.15957777, 0.68773443, 0., 0.00811983],
#                [0., 0.29981688, 0.76250822, 0.9826993, 0.8283224, 0.]])

#
#print(test_matrix2)
#print("____________________________________")
#print(edmonds(test_matrix2, 0))

#print(test_matrix1) 
#print(edmonds(test_matrix1, 0))         

#candidate_mst = edmonds(test_matrix1, 0)
#print(candidate_mst)

#test case from NLP HW2
#hw2_matrix = np.array([[0,0,15,0,0],[0,0,5,5,15],[0,20,0,5,30],[0,10,20,0,5],[0,5,10,15,0]])
#print(hw2_matrix)
#print(edmonds(hw2_matrix, 0))

#tests whether an mst is a dependency tree (i.e. if it has branching from the root)
#INPUT: List of arcs (ordered pairs), Index of ROOT
#OUTPUT: Boolean value
def is_dependency_graph(mst, root):
    counter = 0
    for arc in mst:
        if arc[0] == root:
            counter += 1
    if counter > 1:
        return False
    return True

#print(is_dependency_graph(candidate_mst, 0))

#returns the list of nodes accessible in one step from the root
def nodes_from_root(graph, root):
    root_nodes_list = []
    nodes_list = get_nodes(graph)
    for node in nodes_list:
        if graph[root][node] != 0.0:
            root_nodes_list.append(node)
    return root_nodes_list

#print(nodes_from_root(test_matrix1, 0))

#removes arcs from the root unless they are to a specific input node
def reduced_graph(graph, root, node):
    new_graph = np.copy(graph)
    nodes_list = get_nodes(graph)
    for node1 in nodes_list:
        if node1 != node:
            new_graph[root][node1] = 0.0          
    return new_graph

#given an adjacency matrix, returns its score
def score_of_graph(graph):
    edge_list = edges(graph)
    return sum_path(graph, edge_list)

#IMPORTANT - FINAL FUNCTION - GUARANTEED TO FIND VALID DEPENDENCY TREE
#this is the final edmonds function, returns an mst that we can actually use
#as a dependence tree
#INPUT: Adjacency matrix, Index of ROOT
#OUTPUT: Dependency tree (given as a list of arcs - ordered pairs)

def get_edmonds(graph1, root):

    print("ORIGINAL GRAPH")
    print(preprocess(graph1, root))
    
    graph = np.copy(graph1)
    mst = edmonds(graph, root)    
    root_nodes = nodes_from_root(graph1, root)

    if is_dependency_graph(mst, root):
        return mst 
    
    else:
        print("WOULDN'T HAVE BEEN A DEPENDENCY TREE")
        score = 0
        for node in root_nodes:
            new_graph = reduced_graph(graph, root, node)
#            print("NEW GRAPH")
#            print(new_graph)
            new_attempt = edmonds(new_graph, root)
#            print("EDGES OF MST")
#            print(edges(new_attempt))
#            print("SCORE", score_of_graph(new_attempt))
            current_score = sum_path(graph, new_attempt)
            if current_score > score:
                score = current_score
                best_attempt = new_attempt
                
    return best_attempt

#END OF FUNCTION

#mst_final = get_edmonds(test_matrix1, 0)
#print(mst_final)

#THIS IS THE END OF THE MST ALGORITHM

#_____________________________________________________________________________________
#THE CODE BELOW THIS POINT TESTS WHETHER A GIVEN DEPENDENCY TREE IS PROJECTIVE OR NOT
#_____________________________________________________________________________________

#takes in a list of arcs and a node and returns the list of nodes the node can reach
def path_initial_steps(path, node):
    initial_steps = []
    for arc in path:
        if arc[0] == node:
            initial_steps.append(arc)
    return initial_steps

test_path = [(0, 19), (1, 7), (2, 5), (2, 12), (2, 17), (3, 9), (4, 11), (5, 3), (5, 8), (5, 13), (6, 10), (6, 16), (9, 15), (10, 4), (11, 14), (12, 18), (14, 2), (16, 1), (19, 6)]
#print(path_initial_steps(test_path, 18))

#takes in a list of arcs and a node and returns the list of nodes the node can reach
def path_end_points(path, node):
    nodes_reachable = []
    initial_steps = path_initial_steps(path, node)

    while len(initial_steps) > 0:
#        print("INITIAL STEPS", initial_steps)
        new_steps = []
        for arc in initial_steps:
            nodes_reachable.append(arc[1])
            new_steps.append(arc[1])
        
        new_arcs = []
        for node in new_steps:
            arcs_per_node = path_initial_steps(path, node)
            if len(arcs_per_node) > 0:
                for arc2 in arcs_per_node:
                    new_arcs.append(arc2)
        
        initial_steps = new_arcs
    
    return nodes_reachable          
            
#print(path_end_points(test_path, 0))

#tests whether a given arc is projective
def is_projective_arc(path, arc):
    head = arc[0]
    dependent = arc[1]
    nodes_reachable = path_end_points(path, head)
#    print("CAN REACH NODES", nodes_reachable)
    if head < dependent:
        relevant_nodes = [x for x in range(head + 1, dependent)]
    else:
        relevant_nodes = [x for x in range(dependent + 1, head)]
    
    for node in relevant_nodes:
        if node not in nodes_reachable:
#            print("CANNOT REACH NODE", node)
            return False

    return True

#print(is_projective_arc(test_path, (1, 7)))
   
#tests whether a tree (construed as a list of arcs) is projective
def is_projective_tree(tree):
    for arc in tree:
        if not is_projective_arc(tree, arc):
#            print("ARC IS NOT PROJECTIVE", arc)
            return False
    return True

#print(is_projective_tree(test_path))
#print(is_projective_tree(mst_final))
