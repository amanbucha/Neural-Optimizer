import torch.nn as nn
import torch

def fuse_conv_bn_fx(graph, model):
    modules = dict(model.named_modules())

    for node in list(graph.nodes):
        if node.op == "call_module" and isinstance(modules[node.target], nn.Conv2d):
            users = list(node.users.keys())
            if len(users) != 1:
                continue
            next_node = users[0]
            if next_node.op != "call_module":
                continue
            target = node.target
            if target not in modules or not isinstance(modules[target], nn.Conv2d):
                continue
            conv = modules[target]
            bn = modules[next_node.target]
            if not isinstance(bn, nn.BatchNorm2d):
                continue

            fused = fuse_conv_bn_weights(conv, bn)
            new_name = node.target + "fused_bn"
            model.add_module(new_name, fused)

            with graph.inserting_after(next_node):
                new_node = graph.call_module(new_name, args=node.args)
                next_node.replace_all_uses_with(new_node)
                graph.erase_node(next_node)
                graph.erase_node(node)

    return graph

def fuse_conv_bn_weights(conv, bn):
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True
    )

    w_conv = conv.weight.clone()
    if conv.bias is None:
        b_conv = torch.zeros(w_conv.size(0), device=w_conv.device)
    else:
        b_conv = conv.bias.clone()

    w_bn = bn.weight.clone()
    b_bn = bn.bias.clone()
    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps

    std = torch.sqrt(running_var + eps)
    w_scale = w_bn / std
    b_scale = b_bn - w_bn * running_mean / std

    fused_conv.weight.data = w_conv * w_scale.reshape([-1, 1, 1, 1])
    fused_conv.bias.data = b_conv * w_scale + b_scale
    return fused_conv