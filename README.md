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


