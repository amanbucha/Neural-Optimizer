import torch.nn as nn
from torch.fx import GraphModule
from torch.fx.graph import Graph

def fuse_linear_chain_fx(graph: Graph, module: GraphModule):
    nodes = list(graph.nodes)
    for i in range(len(nodes) - 1):
        n1, n2 = nodes[i], nodes[i + 1]

        if n1.op == 'call_module' and n2.op == 'call_module':
            mod1 = module.get_submodule(n1.target)
            mod2 = module.get_submodule(n2.target)

            if isinstance(mod1, nn.Linear) and isinstance(mod2, nn.Linear):
                # Fuse if n1 feeds directly into n2
                if n2.args[0] != n1:
                    continue

                # Compute fused weights and bias
                W1, b1 = mod1.weight, mod1.bias
                W2, b2 = mod2.weight, mod2.bias

                W_fused = W2 @ W1
                b_fused = W2 @ b1 + b2 if b1 is not None else b2

                fused_linear = nn.Linear(W1.size(1), W2.size(0))
                fused_linear.weight.data.copy_(W_fused)
                fused_linear.bias.data.copy_(b_fused)

                fused_name = "fused_linear"
                module.add_module(fused_name, fused_linear)

                with graph.inserting_after(n2):
                    new_node = graph.call_module(fused_name, args=n1.args)

                n2.replace_all_uses_with(new_node)
                graph.erase_node(n2)
                graph.erase_node(n1)
                return  # restart due to modified graph