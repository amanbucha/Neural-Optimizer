import operator

def simplify_ops_fx(graph):
    for node in list(graph.nodes):
        if node.op == "call_function":
            if node.target == operator.add and any(isinstance(arg, (int, float)) and arg == 0 for arg in node.args):
                other = node.args[0] if node.args[1] == 0 else node.args[1]
                node.replace_all_uses_with(other)
                graph.erase_node(node)
            elif node.target == operator.mul and any(isinstance(arg, (int, float)) and arg == 1 for arg in node.args):
                other = node.args[0] if node.args[1] == 1 else node.args[1]
                node.replace_all_uses_with(other)
                graph.erase_node(node)
    return graph
