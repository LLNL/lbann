import functools
import operator

import numpy as np
import lbann
import lbann.modules

class FFTShift(lbann.modules.Module):
    """Shift zero-frequency component of discrete Fourier transform to
    center of spectrum.

    The input can have any dimension, although the first dimension is
    interpreted as the channel dimension (e.g. CHW images). fftshift
    is applied independently to each channel.

    See:

      https://numpy.org/doc/stable/reference/generated/numpy.fft.fftshift.html

    """

    def __init__(self):
        pass

    def forward(self, x, dims):
        """Apply fftshift.

        Args:
            x (lbann.Layer): Input tensor
            dims (tuple of int): Dimensions of x (dim 0 corresponds to
                channel)

        Returns:
            Layer: Output tensor

        """

        # Get gather indices by applying fftshift to tensor filled with indices
        # Note: Independent fftshift for each channel (dim 0)
        spatial_size = np.prod(dims[1:])
        spatial_inds = np.arange(spatial_size).reshape(dims[1:])
        spatial_inds = np.fft.fftshift(spatial_inds)
        channel_offsets = np.arange(0, dims[0]*spatial_size, spatial_size)
        channel_offsets = channel_offsets.reshape([-1] + [1]*spatial_inds.ndim)
        inds = np.expand_dims(spatial_inds, 0) + channel_offsets

        # Construct LBANN layer graph
        size = np.prod(dims)
        x = lbann.Reshape(x, dims=[size])
        inds = lbann.WeightsLayer(
            weights=lbann.Weights(
                lbann.ValueInitializer(values=inds.flatten()),
                optimizer=lbann.NoOptimizer(),
            ),
            dims=[size],
        )
        y = lbann.Gather(x, inds)
        return lbann.Reshape(y, dims=dims)

# Test fftshift by performing metric checking and gradient checking
if __name__ == "__main__":

    # Dummy input
    reader = lbann.reader_pb2.DataReader()
    def add_data_reader(role):
        _reader = reader.reader.add()
        _reader.name = 'synthetic'
        _reader.role = role
        _reader.num_samples = 1
        _reader.num_labels = 1
        _reader.synth_dimensions = '1'
        _reader.fraction_of_data_to_use = 1.0
    add_data_reader('train')
    add_data_reader('test')
    input_ = lbann.Input()

    # NumPy implementation
    dims = [2,3,4,7]
    np_x = np.random.uniform(size=dims).astype(np.float32)
    np_y = np.zeros_like(np_x)
    for i in range(dims[0]):
        np_y[i] = np.fft.fftshift(np_x[i])
    np_scales = np.random.uniform(size=np.prod(dims)).astype(np.float32)
    np_z = np.inner(np_y.flatten(), np_scales).item()
    tol = 8 * np_z * np.finfo(np.float32).eps

    # LBANN implementation
    lbann_x = lbann.WeightsLayer(
        weights=lbann.Weights(
            lbann.ValueInitializer(values=np_x.flatten()),
        ),
        dims=np_x.shape,
    )
    lbann_y = FFTShift()(lbann_x, dims)
    lbann_scales = lbann.WeightsLayer(
        weights=lbann.Weights(
            lbann.ValueInitializer(values=np_scales),
            optimizer=lbann.NoOptimizer(),
        ),
        dims=np_scales.shape,
    )
    lbann_z = lbann.MatMul(
        lbann.Reshape(lbann_y, dims=[1,-1]),
        lbann.Reshape(lbann_scales, dims=[-1,1])
    )

    # Construct LBANN model with metric checking and gradient checking
    metric = lbann.Metric(lbann_z, name='metric')
    callbacks = [
        lbann.CallbackCheckMetric(
            metric=metric.name,
            lower_bound=np_z-tol,
            upper_bound=np_z+tol,
            error_on_failure=True,
            execution_modes='test',
        ),
        lbann.CallbackCheckGradients(error_on_failure=True),
    ]
    model = lbann.Model(
        epochs=0,
        layers=lbann.traverse_layer_graph([input_, lbann_x]),
        objective_function=lbann_z,
        metrics=metric,
        callbacks=callbacks,
    )

    # Run LBANN
    lbann.run(
        trainer=lbann.Trainer(mini_batch_size=1),
        model=model,
        data_reader=reader,
        optimizer=lbann.SGD(),
        job_name='lbann_fftshift_test',
    )
