#takes in a list of arcs and a node and returns the list of nodes the node can reach
def path_initial_steps(path, node):
    initial_steps = []
    for arc in path:
        if arc[0] == node:
            initial_steps.append(arc)
    return initial_steps

#test_path = [(0, 19), (1, 7), (2, 5), (2, 12), (2, 17), (3, 9), (4, 11), (5, 3), (5, 8), (5, 13), (6, 10), (6, 16), (9, 15), (10, 4), (11, 14), (12, 18), (14, 2), (16, 1), (19, 6)]
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
    
def is_projective_tree(tree):
    for arc in tree:
        if not is_projective_arc(tree, arc):
#            print("ARC IS NOT PROJECTIVE", arc)
            return False
    return True

#print(is_projective_tree(test_path))
#print(is_projective_tree(mst_final))
