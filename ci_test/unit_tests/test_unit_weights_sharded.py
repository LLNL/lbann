import lbann
import numpy as np
import test_util
import pytest


@pytest.mark.parametrize('shard_dim', [16, 3])
@test_util.lbann_test(check_gradients=True)
def test_one(shard_dim):
    # Prepare reference output
    np.random.seed(20231012)
    x = np.random.rand(shard_dim, shard_dim).astype(np.float32)
    w = np.random.rand(shard_dim, shard_dim).astype(np.float32)
    b = np.random.rand(1, shard_dim).astype(np.float32)
    ref = x @ w.T + b

    # Add sharded weights
    linearity_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ValueInitializer(values=np.nditer(w, order='F')),
        sharded=True,
    )
    bias_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ValueInitializer(values=np.nditer(b)),
        sharded=True,
    )

    tester = test_util.ModelTester()

    x = tester.inputs(x)
    reference = tester.make_reference(ref)

    # Test layer
    y = lbann.FullyConnected(x,
                             weights=(linearity_weights, bias_weights),
                             data_layout='data_parallel',
                             num_neurons=shard_dim,
                             has_bias=True,
                             transpose=False)

    # Set test loss
    tester.set_loss(lbann.MeanSquaredError(y, reference))
    tester.set_check_gradients_tensor(lbann.Square(y))
    return tester
