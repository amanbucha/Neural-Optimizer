import torch
from torch.fx import GraphModule
from utils.tracer import CustomTracer
from examples.redundant_cnn import get_model
from passes.optimize import optimize_fx_model
from utils.compare_models import compare_outputs
import pytest

def test_run():
    model, example_input = get_model()
    num_tests = 10
    tracer = CustomTracer()
    traced = GraphModule(model, tracer.trace(model))
    optimized = optimize_fx_model(traced)

    all_passed = True

    for i in range(num_tests):
        x = torch.randn_like(example_input)
        passed, error = compare_outputs(traced, optimized, x)
        if not passed:
            all_passed = False
            pytest.fail(f"Test {i + 1} failed.")
        else:
            print(f"Test {i + 1} passed.")

    if all_passed:
        print("All tests passed successfully.")
    else:
        pytest.fail("Some tests failed. Check heatmaps for detail.")

if __name__ == "__main__":
    test_run()