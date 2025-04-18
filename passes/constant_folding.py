import operator
import torch.fx as fx
import torch

def constant_fold_fx(graph):
    env = {}
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target in [operator.add, operator.mul, operator.sub, operator.truediv]:
            args = [env.get(arg.name, None) if isinstance(arg, fx.Node) else arg for arg in node.args]
            if all(isinstance(arg, (int, float, torch.Tensor)) for arg in args):
                with graph.inserting_before(node):
                    value = node.target(*args)
                    new_node = graph.create_node("get_attr", target="const_" + node.name)
                    env[node.name] = value
                    node.replace_all_uses_with(new_node)
                    graph.erase_node(node)
    return graph