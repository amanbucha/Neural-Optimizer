import torch
import torch.fx as fx

def simplify_floats(val, epsilon=1e-9):
    """Convert float to int if very close."""
    if isinstance(val, float):
        if abs(val - round(val)) < epsilon:
            return int(round(val))
    return val


def simplify_graph_floats_fx(graph: fx.GraphModule, epsilon=1e-9):
    for node in list(graph.nodes):
        # Look for ops with constant float args
        if node.op == 'call_function':
            new_args = []
            changed = False
            for arg in node.args:
                if isinstance(arg, (float, int)):
                    simplified = simplify_floats(arg, epsilon)
                    if simplified != arg:
                        changed = True
                    new_args.append(simplified)
                else:
                    new_args.append(arg)

            if changed:
                with graph.inserting_after(node):
                    new_node = graph.call_function(node.target, tuple(new_args))
                    node.replace_all_uses_with(new_node)
                    graph.erase_node(node)

        # Replace constant tensor values
        elif node.op == 'call_function' and node.target == torch.tensor:
            val = node.args[0]
            if isinstance(val, float):
                simplified = simplify_floats(val, epsilon)
                if simplified != val:
                    with graph.inserting_after(node):
                        new_node = graph.call_function(torch.tensor, (simplified,))
                        node.replace_all_uses_with(new_node)
                        graph.erase_node(node)

    graph.lint()
    return graph
