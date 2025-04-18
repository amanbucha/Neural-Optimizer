from torch.fx import symbolic_trace, GraphModule
from passes.constant_folding import constant_fold_fx
from passes.dead_code_elimination import eliminate_dead_code_fx
from passes.op_simplification import simplify_ops_fx
from passes.fuse_conv_bn import fuse_conv_bn_fx
from passes.fuse_conv_chains import fuse_conv_chain_fx
from passes.fuse_linear_chains import fuse_linear_chain_fx
from passes.simplify_floats import simplify_graph_floats_fx

def optimize_fx_model(model):
    traced = symbolic_trace(model)
    graph = traced.graph
    simplify_graph_floats_fx(graph)
    fuse_conv_chain_fx(graph, traced)
    fuse_conv_bn_fx(graph, traced)
    constant_fold_fx(graph)
    simplify_ops_fx(graph)
    eliminate_dead_code_fx(graph)
    fuse_linear_chain_fx(graph, traced)
    graph.lint()
    optimized = GraphModule(traced, graph)
    return optimized
