def path_initial_steps(path, node):
    '''
    takes in a list of arcs and a node and returns the list of nodes the node can reach
    '''
    initial_steps = []
    for arc in path:
        if arc[0] == node:
            initial_steps.append(arc)
    return initial_steps


def path_end_points(path, node):
    '''
    takes in a list of arcs and a node and returns the list of nodes the node can reach
    '''

    nodes_reachable = []
    initial_steps = path_initial_steps(path, node)
    while len(initial_steps) > 0:
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


def is_projective_arc(path, arc):
    head = arc[0]
    dependent = arc[1]
    nodes_reachable = path_end_points(path, head)

    if head < dependent:
        relevant_nodes = [x for x in range(head + 1, dependent)]
    else:
        relevant_nodes = [x for x in range(dependent + 1, head)]
    for node in relevant_nodes:
        if node not in nodes_reachable:
            return False
    return True


def is_projective_tree(tree):
    for arc in tree:
        if not is_projective_arc(tree, arc):
            return False
    return True
