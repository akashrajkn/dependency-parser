#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 18:02:24 2017

@author: jackharding
"""

import numpy as np
from collections import defaultdict
import itertools


test_matrix1 = np.random.rand(15, 15)


def get_nodes(adjacency_matrix):
    '''
    returns a list of the nodes in the matrix
    '''
    nodes_list = []
    for node in range(adjacency_matrix.shape[0]):
        nodes_list.append(node)
    return nodes_list


def preprocess(matrix1, root):
    '''
    removes edges going to the root, and from elements to themselves
    '''
    nodes_list = get_nodes(matrix1)
    matrix = np.copy(matrix1)
    for node in nodes_list:
        matrix[node][node] = 0.0
        matrix[node][root] = 0.0
    return matrix


def entering_weights(adjacency_matrix, node):
    '''
    returns a list of the weights to each node
    '''
    return adjacency_matrix[:, node]


def edges(graph):
    '''
    returns a list of the non-empty edges in the graph
    '''
    edges = []
    for i in range(graph.shape[0]):
        for j in range(graph.shape[0]):
            if graph[i][j] != 0.0:
                edges.append((i, j))
    return(edges)


def biggest_edge(graph):
    '''
    returns a dictionary with nodes as keys and the biggest weight to each node as values
    '''
    nodes_list = get_nodes(graph)
    biggest_edge_dict = {}
    for node in nodes_list:
        biggest_edge_dict[node] = max(entering_weights(graph, node))
    return biggest_edge_dict


def biggest_edge_index(graph):
    '''
    returns a dict with nodes as keys and the index of the biggest incoming weight for each node as values
    '''
    nodes_list = get_nodes(graph)
    biggest_edge_dict = biggest_edge(graph)
    biggest_edge_index_dict = {}
    for node1 in nodes_list:
        for node2 in nodes_list:
            if graph[node1][node2] == biggest_edge_dict[node2]:
                biggest_edge_index_dict[node2] = node1
    return biggest_edge_index_dict


def try_biggest(graph, root):
    '''
    creates the graph with the largest edge to each node
    '''
    output = np.zeros(graph.shape)
    nodes_list = get_nodes(graph)
    biggest_edge_index_dict = biggest_edge_index(graph)
    for node1 in nodes_list:
        for node2 in nodes_list:
            if biggest_edge_index_dict[node1] == node2:
                output[node2][node1] = graph[node2][node1]
    return output


def try_biggest_alternative(graph, root):
    output = np.zeros(graph.shape)
    nodes_list = get_nodes(graph)
    biggest_edge_index_dict = biggest_edge_index(graph)
    for node1 in nodes_list:
        for node2 in nodes_list:
            if biggest_edge_index_dict[node1] == node2:
                output[node2][node1] = graph[node2][node1]
    return output


def cyclic(graph):
    '''
    returns True if there is a cycle in the graph,False otherwise
    adapted from code I found on Stackexchange
    '''
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


def is_mst(graph_attempt):
    return not cyclic(graph_attempt)


def find_cycles(graph):
    '''
    returns a list of all cycles in the graph (e.g. [1, 2, 3, 1] is a cycle)
    if there are no cycles, returns the empty list
    '''
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


def sum_path(graph, pair_list):
    '''
    given a list of edges, sums the weights along each edge
    '''
    sum_path = 0.0
    for (node1, node2) in pair_list:
        sum_path += graph[node1][node2]
    return sum_path


def path_through_set(set1):
    '''
    takes in a set and returns all the possible paths through each element of the set
    '''
    path = list(itertools.permutations(set1, len(set1)))
    return path


def path_into_pair(path):
    return list(zip(path, path[1:]))


def edmonds(graph, root):
    '''
    takes in a graph and root and returns an mst rooted at the root
    '''
    conversion_dictionaries_list = []
    graph1 = preprocess(graph, root)

    while True:
        graph_attempt = try_biggest(graph1, root)

        if is_mst(graph_attempt):
            graph1 = graph_attempt
            break

        conversion_dict = defaultdict()
        nodes_list = get_nodes(graph1)
        cycle1 = find_cycles(graph_attempt)[0]

        new_graph_nodes = []
        contracted_node = "contracted node"
        for node in nodes_list:
            if node not in set(cycle1):
                new_graph_nodes.append(node)

        updated_indices = {}
        for i in range(len(new_graph_nodes)):
            updated_indices[new_graph_nodes[i]] = i

        updated_indices[contracted_node] = len(new_graph_nodes)

        new_graph = np.zeros((len(new_graph_nodes) + 1, len(new_graph_nodes) + 1))

        for node1 in new_graph_nodes:
            for node2 in new_graph_nodes:
                new_graph[updated_indices[node1]][updated_indices[node2]] = graph1[node1][node2]

        #    outward edges
        for node2 in new_graph_nodes:
            max_from_cycle = [0.0, ("dummy", "dummy")]
            for node1 in set(cycle1):
                if graph1[node1][node2] > max_from_cycle[0]:
                    max_from_cycle[0] = graph1[node1][node2]
                    max_from_cycle[1] = (node1, node2)
            new_graph[updated_indices[contracted_node]][updated_indices[node2]] = max_from_cycle[0]
            if max_from_cycle[1] != ("dummy", "dummy"):
                conversion_dict[(updated_indices[contracted_node], updated_indices[node2])] = [max_from_cycle[1]]

        #    inward edges
        nodes_in_cycle = list(set(cycle1))

        test_graph = np.zeros((len(nodes_in_cycle), len(nodes_in_cycle)))
        for node1 in range(len(nodes_in_cycle)):
            for node2 in range(len(nodes_in_cycle)):
                test_graph[node1][node2] = graph1[nodes_in_cycle[node1]][nodes_in_cycle[node2]]

        for node1 in new_graph_nodes:
            max_path_score = 0.0
            max_path = []
            for i in range(len(nodes_in_cycle)):
                path_from_node = edmonds(test_graph, i)

                for j in range(len(path_from_node)):
                    path_from_node[j] = (nodes_in_cycle[path_from_node[j][0]], nodes_in_cycle[path_from_node[j][1]])
                path_from_node.append((node1, nodes_in_cycle[i]))

                current_path_score = sum_path(graph1, path_from_node)
                if current_path_score > max_path_score:
                    max_path_score = current_path_score
                    max_path = path_from_node

            conversion_dict[(updated_indices[node1], updated_indices[contracted_node])] = max_path
            new_graph[updated_indices[node1]][updated_indices[contracted_node]] = max_path_score

        for node1 in new_graph_nodes:
            for node2 in new_graph_nodes:
                conversion_dict[(updated_indices[node1],updated_indices[node2])] = [(node1, node2)]

        #adding backtracking pointers for each recursive layer
        conversion_dictionaries_list.append(conversion_dict)
        graph1 = new_graph

    # backtracking to find edges used in original graph given final contracted mst
    list_of_nodes = edges(graph1)
    if len(conversion_dictionaries_list) == 0:
        return list_of_nodes

    while len(conversion_dictionaries_list) > 0:
        new_list = []
        for pair in list_of_nodes:
            for node_pair in conversion_dictionaries_list[-1][pair]:
                new_list.append(node_pair)
        list_of_nodes = new_list
        del conversion_dictionaries_list[-1]
    return list_of_nodes


def is_dependency_graph(mst, root):
    '''
    tests if there are multiple arcs from the root
    '''
    counter = 0
    for arc in mst:
        if arc[0] == root:
            counter += 1
    if counter > 1:
        return False
    return True


def nodes_from_root(graph, root):
    '''
    gives a list of the nodes reachable in one step from the root
    '''
    root_nodes_list = []
    nodes_list = get_nodes(graph)
    for node in nodes_list:
        if graph[root][node] != 0.0:
            root_nodes_list.append(node)
    return root_nodes_list


def reduced_graph(graph, root, node):
    '''
    makes it so there's only one (specified) node reachable in one step from the root
    '''
    new_graph = np.copy(graph)
    nodes_list = get_nodes(graph)
    for node1 in nodes_list:
        if node1 != node:
            new_graph[root][node1] = 0.0
    return new_graph


def score_of_graph(graph):
    '''
    sums all the edges in an adjacency matrix, useful for comparing candidate msts
    '''
    edge_list = edges(graph)
    return sum_path(graph, edge_list)


def get_edmonds(graph1, root):
    '''
    this is THE FINAL edmonds function
    '''
    graph = np.copy(graph1)
    mst = edmonds(graph, root)
    root_nodes = nodes_from_root(graph1, root)

    if is_dependency_graph(mst, root):
        return mst
    else:
        score = 0
        for node in root_nodes:
            new_graph = reduced_graph(graph, root, node)

            new_attempt = edmonds(new_graph, root)
            current_score = sum_path(graph, new_attempt)
            if current_score > score:
                score = current_score
                best_attempt = new_attempt
    return best_attempt

mst_final = get_edmonds(test_matrix1, 0)


def edges_to_single(list_of_edges):
    '''
    converts the get_edmonds output (a list of edges) to the format used in the paper
    '''
    output = [0] * (len(list_of_edges) + 1)
    for edge in list_of_edges:
        output[edge[1]] = edge[0]
    return output


my_result = edges_to_single(mst_final)


def mst(scores):
    '''
    Chu-Liu-Edmonds' algorithm for finding minimum spanning arborescence in graphs.
    Calculates the arborescence with node 0 as root.
    Source: https://github.com/chantera/biaffineparser/blob/master/utils.py

    WARNING: mind the comment below. This mst function expects scores[i][j] to be the score from j to i,
    not from i to j (as you would probably expect!). If you use a graph where you have the convention
    that a head points to its dependent, you will need to transpose it before calling this function.
    That is, call `mst(scores.T)` instead of `mst(scores)`.

    :param scores: `scores[i][j]` is the weight of edge from node `j` to node `i`
    :returns an array containing the head node (node with edge pointing to current node) for each node,
             with head[0] fixed as 0
    '''
    length = scores.shape[0]
    scores = scores * (1 - np.eye(length))
    heads = np.argmax(scores, axis=1)
    heads[0] = 0
    tokens = np.arange(1, length)
    roots = np.where(heads[tokens] == 0)[0] + 1

    # deal with roots
    if len(roots) < 1:
        root_scores = scores[tokens, 0]
        head_scores = scores[tokens, heads[tokens]]
        new_root = tokens[np.argmax(root_scores / head_scores)]
        heads[new_root] = 0

    elif len(roots) > 1:
        root_scores = scores[roots, 0]
        scores[roots, 0] = 0
        new_heads = np.argmax(scores[roots][:, tokens], axis=1) + 1
        new_root = roots[np.argmin(scores[roots, new_heads] / root_scores)]
        heads[roots] = new_heads
        heads[new_root] = 0

    # construct edges and vertices
    edges = defaultdict(set)
    vertices = set((0,))
    for dep, head in enumerate(heads[tokens]):
        vertices.add(dep + 1)
        edges[head].add(dep + 1)

    # identify cycles & contract
    for cycle in _find_cycle(vertices, edges):
        dependents = set()
        to_visit = set(cycle)
        while len(to_visit) > 0:
            node = to_visit.pop()
            if node not in dependents:
                dependents.add(node)
                to_visit.update(edges[node])
        cycle = np.array(list(cycle))
        old_heads = heads[cycle]
        old_scores = scores[cycle, old_heads]
        non_heads = np.array(list(dependents))
        scores[np.repeat(cycle, len(non_heads)),
               np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
        new_heads = np.argmax(scores[cycle][:, tokens], axis=1) + 1
        new_scores = scores[cycle, new_heads] / old_scores
        change = np.argmax(new_scores)
        changed_cycle = cycle[change]
        old_head = old_heads[change]
        new_head = new_heads[change]
        heads[changed_cycle] = new_head
        edges[new_head].add(changed_cycle)
        edges[old_head].remove(changed_cycle)

    return heads


def _find_cycle(vertices, edges):
    '''
    Finds cycles in given graph, where the graph is provided as (vertices, edges).
    https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm  # NOQA
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/etc/tarjan.py  # NOQA
    '''
    _index = [0]
    _stack = []
    _indices = {}
    _lowlinks = {}
    _onstack = defaultdict(lambda: False)
    _SCCs = []

    def _strongconnect(v):
        _indices[v] = _index[0]
        _lowlinks[v] = _index[0]
        _index[0] += 1
        _stack.append(v)
        _onstack[v] = True

        for w in edges[v]:
            if w not in _indices:
                _strongconnect(w)
                _lowlinks[v] = min(_lowlinks[v], _lowlinks[w])
            elif _onstack[w]:
                _lowlinks[v] = min(_lowlinks[v], _indices[w])

        if _lowlinks[v] == _indices[v]:
            SCC = set()
            while True:
                w = _stack.pop()
                _onstack[w] = False
                SCC.add(w)
                if not (w != v):
                    break
            _SCCs.append(SCC)

    for v in vertices:
        if v not in _indices:
            _strongconnect(v)

    return [SCC for SCC in _SCCs if len(SCC) > 1]


def answer_to_edges(answer):
    edges = []
    for i in range(len(answer)):
        edges.append((answer[i], i))
    return edges


def which_is_better(size, time):
    my_score = 0
    their_score = 0
    for i in range(time):
        new_test = np.random.rand(size, size)

        my_answer = edges_to_single(get_edmonds(new_test, 0))
        my_converted = answer_to_edges(my_answer)
        my_score += sum_path(new_test, my_converted)

        their_answer = mst(new_test.transpose())
        their_converted = answer_to_edges(their_answer)
        their_score += sum_path(new_test, their_converted)

    return "MY AVERAGE:", my_score/time, "NOTEBOOK AVERAGE", their_score/time
