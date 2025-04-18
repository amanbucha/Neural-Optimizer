import torch.nn as nn
import torch
import torch.nn.functional as F

def fuse_conv_layers_simple(conv1, conv2):
    if conv1.stride != (1, 1) or conv2.stride != (1, 1):
        raise ValueError("Both convolutions must have stride 1")

    if conv1.padding != (0, 0) or conv2.padding != (0, 0):
        raise ValueError("Both convolutions must have padding 0")

    if conv1.dilation != (1, 1) or conv2.dilation != (1, 1):
        raise ValueError("Both convolutions must have dilation 1")

    if conv1.out_channels != conv2.in_channels:
        raise ValueError("Output channels of conv1 must match input channels of conv2")

    fused_kernel_h = conv1.kernel_size[0] + conv2.kernel_size[0] - 1
    fused_kernel_w = conv1.kernel_size[1] + conv2.kernel_size[1] - 1

    fused_conv = nn.Conv2d(
        in_channels=conv1.in_channels,
        out_channels=conv2.out_channels,
        kernel_size=(fused_kernel_h, fused_kernel_w),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        bias=True if (conv1.bias is not None or conv2.bias is not None) else False
    )

    for oc2 in range(conv2.out_channels):
        for ic1 in range(conv1.in_channels):
            fused_kernel = torch.zeros((fused_kernel_h, fused_kernel_w), device=conv1.weight.device)

            for mid in range(conv1.out_channels):
                kernel1 = conv1.weight[mid, ic1].detach()
                kernel2 = conv2.weight[oc2, mid].detach()

                k1_expanded = kernel1.unsqueeze(0).unsqueeze(0)
                k2_expanded = kernel2.unsqueeze(0).unsqueeze(0)

                result = F.conv2d(
                    k1_expanded,
                    k2_expanded.flip(-1, -2),
                    padding=(conv2.kernel_size[0] - 1, conv2.kernel_size[1] - 1)
                ).squeeze()

                fused_kernel += result

            fused_conv.weight.data[oc2, ic1] = fused_kernel

    if fused_conv.bias is not None:
        fused_bias = torch.zeros(conv2.out_channels, device=conv1.weight.device)

        if conv1.bias is not None:
            for oc2 in range(conv2.out_channels):
                for mid in range(conv1.out_channels):
                    fused_bias[oc2] += torch.sum(conv2.weight[oc2, mid]) * conv1.bias[mid]

        if conv2.bias is not None:
            fused_bias += conv2.bias

        fused_conv.bias.data = fused_bias

    return fused_conv


def fuse_conv_chain_fx(graph: torch.fx.graph, module: torch.fx.GraphModule):
    nodes = list(graph.nodes)
    for i in range(len(nodes) - 1):
        n1, n2 = nodes[i], nodes[i + 1]

        if n1.op == 'call_module' and n2.op == 'call_module':
            mod1 = module.get_submodule(n1.target)
            mod2 = module.get_submodule(n2.target)

            if isinstance(mod1, nn.Conv2d) and isinstance(mod2, nn.Conv2d):
                if n2.args[0] != n1:
                    continue

                fused_conv = fuse_conv_layers_simple(mod1, mod2)
                fused_name = "fused_conv"
                module.add_module(fused_name, fused_conv)

                with graph.inserting_after(n2):
                    new_node = graph.call_module(fused_name, args=n1.args)

                n2.replace_all_uses_with(new_node)
                graph.erase_node(n2)
                graph.erase_node(n1)
                return