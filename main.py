from utils.tracer import CustomTracer
from torch.fx import GraphModule
from passes.optimize import optimize_fx_model
from utils.visualizer import visualize_fx
from examples.redundant_cnn import get_model

model, example_input = get_model()

tracer = CustomTracer()

graph = tracer.trace(model)
traced = GraphModule(model, graph)
visualize_fx(traced, title="Before_Optimization")

optimized = optimize_fx_model(traced)
visualize_fx(optimized, title="After_Optimization")