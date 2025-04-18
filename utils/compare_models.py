import torch

def compare_outputs(model1, model2, input_tensor, verbose=False):
    with torch.no_grad():
        out1 = model1(input_tensor)
        out2 = model2(input_tensor)
    if torch.allclose(out1, out2, rtol=1e-4, atol=1e-5):
        return True, None
    if verbose:
        print("Output mismatch.")
    diff = (out1 - out2).abs().cpu().numpy()
    return False, diff