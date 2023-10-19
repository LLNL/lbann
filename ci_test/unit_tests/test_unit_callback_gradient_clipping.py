import lbann
import numpy as np
import test_util
from glob import glob
import functools
import os

def check_gradients(global_norm=True):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Clear any gradient outputs from previous runs.
            grad_files = glob(os.path.join(test_util._get_work_dir(__file__), 'gradients*.txt'))
            for gf in grad_files:
                os.remove(gf)

            # Run the model.
            f(*args, **kwargs)

            eps = np.finfo(np.float32).eps
            grad_files = glob(os.path.join(test_util._get_work_dir(__file__), 'gradients*.txt'))

            # Compute the weight gradient norms, check they are less than 1,
            # and update global gradient norm.
            norm = 0
            for gf in grad_files:
                weight_norm = np.square(np.loadtxt(gf)).sum()
                assert np.sqrt(weight_norm) <= 1 + eps
                norm += weight_norm
            
            # Check the global gradient norm is less than 1 if requested.
            if global_norm:
                assert np.sqrt(norm) <= 1 + eps
        
        return wrapper
    
    return decorator

def setup_tester(scale, global_norm):
    np.random.seed(20231018)
    x = np.random.normal(scale=scale, size=[8, 16]).astype(np.float32)
    ref = np.zeros_like(x)

    tester = test_util.ModelTester()
    x = tester.inputs(x)
    ref = tester.make_reference(ref)

    y = lbann.FullyConnected(x,
                             num_neurons=16,
                             has_bias=True)
    
    z = lbann.FullyConnected(y,
                             num_neurons=16,
                             has_bias=True)
    
    tester.set_loss(lbann.MeanSquaredError(z, ref), tolerance=10*scale**2)
    tester.extra_callbacks = [
        lbann.CallbackClipGradientNorm(global_norm=global_norm, value=1),
        lbann.CallbackDumpGradients(basename='gradients')
    ]
    return tester

# Case where no clipping is needed.
@check_gradients(global_norm=True)
@test_util.lbann_test(train=True)
def test_gradient_no_clipping():
    return setup_tester(scale=0.1, global_norm=True)

# Case with global clipping.
@check_gradients(global_norm=True)
@test_util.lbann_test(train=True)
def test_gradient_clipping():
    return setup_tester(scale=1, global_norm=True)

# Case with per-weight clipping only.
@check_gradients(global_norm=False)
@test_util.lbann_test(train=True)
def test_gradient_clipping_local():
    return setup_tester(scale=10, global_norm=False)
