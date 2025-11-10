import torch
import torch.fx as fx

qmin, qmax = -128, 127

def quantize_tensor(t, num_bits=8):
    min_val, max_val = t.min(), t.max()
    if min_val == max_val:
        return t, 1.0, 0  # nothing to quantize
    scale = (max_val - min_val) / float(qmax - qmin)
    zero_point = qmin - round(min_val / scale)
    q = torch.clamp(torch.round(t / scale) + zero_point, qmin, qmax).to(torch.int8)
    return q, scale, zero_point

def dequantize_tensor(q, scale, zero_point):
    return (q.float() - zero_point) * scale

def quantize_graph_fx(graph: fx.GraphModule):
    modules = dict(graph.named_modules())
    for node in list(graph.graph.nodes):
        if node.op == 'get_attr':
            attr_name = node.target
            if hasattr(graph, attr_name):
                t = getattr(graph, attr_name)
                if isinstance(t, torch.Tensor) and t.dtype == torch.float32:
                    q, scale, zp = quantize_tensor(t)
                    setattr(graph, attr_name, q)
                    graph.register_buffer(attr_name + "_scale", torch.tensor(scale))
                    graph.register_buffer(attr_name + "_zero_point", torch.tensor(zp))

        elif node.op == 'call_function' and node.target == torch.tensor:
            val = node.args[0]
            if isinstance(val, torch.Tensor) and val.dtype == torch.float32:
                q, scale, zp = quantize_tensor(val)
                with graph.graph.inserting_after(node):
                    dq_node = graph.graph.call_function(
                        dequantize_tensor, args=(q, scale, zp)
                    )
                    node.replace_all_uses_with(dq_node)
                    graph.graph.erase_node(node)

    graph.recompile()
    return graph
