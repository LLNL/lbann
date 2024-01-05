import lbann
import numpy as np
import os
import os.path
import sys

current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
module_name = os.path.splitext(os.path.basename(current_file))[0]

sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'common_python'))
import test_util


# ==============================================
# Functionality for Python metric
# ==============================================
# Note: The Python metric class imports this file as a module and calls
# the function below to return a value.
def evaluate(experiment_path, rank):
    if not experiment_path or not isinstance(experiment_path, str):
        return -1.0
    if experiment_path != 'trainer0/model0':
        return -2.0
    if not isinstance(rank, int) or rank < 0:
        return -3.0

    # Successful result
    return 1.0


# ==============================================
# Test
# ==============================================


@test_util.lbann_test()
def test_metric():
    # Prepare reference output
    np.random.seed(20240104)
    x = np.random.rand(2, 2).astype(np.float32)
    ref = x + 1

    tester = test_util.ModelTester()

    x = tester.inputs(x)
    reference = tester.make_reference(ref)

    # Test layer
    y = lbann.AddConstant(x, constant=1)

    tester.extra_metrics.append(
        lbann.PythonMetric(name='pymetric',
                           module=module_name,
                           module_dir=current_dir,
                           function='evaluate'))
    tester.extra_callbacks.append(
        lbann.CallbackCheckMetric(metric='pymetric',
                                  lower_bound=1.0,
                                  upper_bound=1.0,
                                  error_on_failure=True,
                                  execution_modes='test'))

    # Set test loss
    tester.set_loss(lbann.MeanSquaredError(y, reference))
    tester.set_check_gradients_tensor(lbann.Square(y))
    return tester
