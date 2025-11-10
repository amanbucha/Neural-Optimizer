# Neural Optimizer

A lightweight framework to optimize PyTorch models by rewriting computation graphs using `torch.fx`.  

### Features
- Custom tracer for capturing models with redundant or simplify-able ops
- Graph-level optimization passes for improving runtime efficiency
- Designed for inference-time optimization (post-training; weights must be defined)
- Automated tests that compare the outputs of the original and optimized models to ensure numerical equivalence (with a small error tolerance)

### Optimization Passes
- Conv → Conv fusion
- Linear → Linear fusion
- Conv → BatchNorm fusion
- Float simplification
- Constant folding
- Dead code elimination
- Quantization

### Example

**Before Optimization**  
![Before Optimization.png](images/Before_Optimization.png)

**After Optimization**    
![After Optimization.png](images/After_Optimization.png)

**Verification with Tests**

Test both the models on random inputs to ensure accuracy

```
Test 1 passed.
Test 2 passed.
Test 3 passed.
Test 4 passed.
Test 5 passed.
Test 6 passed.
Test 7 passed.
Test 8 passed.
Test 9 passed.
Test 10 passed.
All tests passed successfully.
.
1 passed in 0.81s
```

### Benchmarks

We compare inference times for three versions of an example model:  
1. The original model
2. The Neural Optimizer–optimized model
3. The TorchScript–optimized model


```
(.venv) aman.bucha@LDD2QFY4HK Neural-Optimizer % python benchmarks/benchmark.py
For 300,000 runs:
Original Model took 24.3303 seconds.
Neural Optimizer Optimized Model took 8.4652 seconds.
TorchScript Optimized Model took 7.6490 seconds.
```

The Neural Optimizer achieves a **~3× speedup** over the baseline using only basic graph-level optimizations — already performing close to TorchScript’s optimized runtime.

> **Note:** All benchmarks were performed **on CPU** with **single-sample (non-batched) inference** for a fair comparison of graph-level execution speed.


### Limitations
- Optimizations are valid **only after model weights are initialized**
- Not intended for training-time graph rewriting (may break backprop or introduce inaccuracies during gradient computation)


### Setup (with virtual environment)

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Set PYTHONPATH to the current working directory
export PYTHONPATH=$(pwd)

# Install dependencies
pip install -r requirements.txt
```

### Running the Project
```bash
python main.py
```

### Running Tests
```bash
pytest -sq > tests/test.log
```

### Benchmarking
```bash
python benchmarks/benchmark.py
```

### Future Work
- Integrate TensorFlow backend  
- Add CUDA kernel-level optimizations for GPU benchmarking  
- Improve numerical stability checks for quantization passes

### License
MIT License © 2025 Aman Bucha



