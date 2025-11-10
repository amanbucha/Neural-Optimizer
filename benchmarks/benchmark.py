import torch
import time
from torch.fx import GraphModule
from utils.tracer import CustomTracer
from examples.redundant_cnn import get_model
from passes.optimize import optimize_fx_model

RUNS = 300_000

def benchmark_model(model, data):
    start_time = time.time()
    for input_data in data:
        model(input_data)
    end_time = time.time()
    return end_time - start_time

def benchmark():
    model, example_input = get_model()
    tracer = CustomTracer()
    traced = GraphModule(model, tracer.trace(model))
    neural_compiler_model = optimize_fx_model(traced)
    torch_model = torch.jit.optimize_for_inference(torch.jit.script(model))
    data = generate_random_data(example_input)
    print(f"For {len(data):,} runs:")
    for name, mod in [
        ("Original Model", model),
        ("Neural Compiler Optimized Model", neural_compiler_model),
        ("TorchScript Optimized Model", torch_model)
    ]:
        duration = benchmark_model(mod, data)
        print(f"{name} took {duration:.4f} seconds.")

def generate_random_data(example_input):
    random_data = [torch.randn_like(example_input) for _ in range(RUNS)]
    return random_data

if __name__ == "__main__":
    benchmark()
