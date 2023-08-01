import numpy as np
import lbann
import lbann.modules

class RadialProfile(lbann.modules.Module):
    """Compute average pixel value w.r.t. distance from image center.

    We compute the distance between each image pixel and the image
    center. These distances are binned (with a bin size of 1), and the
    average pixel value in each bin is computed.

    A separate profile is computed for each image channel. The image
    can have any spatial dimension, but the first dimension is
    interpreted as the channel dimension (e.g. CHW format).

    """

    def __init__(self):
        pass

    def forward(self, image, dims, max_r):
        """Compute radial profile.

        Args:
            image (lbann.Layer): Image
            dims (tuple of int): Image dimensions (dim 0 corresponds
                to channel)
            max_r (int): Maximum radial distance. Pixels outside this
                distance are ignored.

        Returns:
            Layer: num_channels x max_r radial profile

        """

        # Bin spatial positions
        r, r_counts = self._find_radial_bins(dims[1:], max_r)

        # Reciprocal of bin counts
        # Note: If a count is 0, its reciprocal is 0.
        r_counts_recip = [0 if c==0 else 1/c for c in r_counts]

        # Get scatter indices and scaling factors
        # Note: Independent binning for each channel (dim 0)
        tile_dims = [dims[0]] + [1]*r.ndim
        inds_vals = np.tile(r, tile_dims)
        inds_vals += np.arange(0, dims[0]*max_r, max_r).reshape(tile_dims)
        inds_vals[:,r>=max_r] = -1
        inds_vals = inds_vals.flatten()
        scales_vals = r_counts_recip * dims[0]

        # Construct LBANN layer graph
        image = lbann.Reshape(image, dims=[np.prod(dims)])
        inds = lbann.WeightsLayer(
            weights=lbann.Weights(
                lbann.ValueInitializer(values=inds_vals),
                optimizer=lbann.NoOptimizer(),
            ),
            dims=[len(inds_vals)],
        )
        r_sums = lbann.Scatter(image, inds, dims=[dims[0]*max_r])
        scales = lbann.WeightsLayer(
            weights=lbann.Weights(
                lbann.ValueInitializer(values=scales_vals),
                optimizer=lbann.NoOptimizer(),
            ),
            dims=[len(scales_vals)],
        )
        r_means = lbann.Multiply(scales, r_sums)
        return lbann.Reshape(r_means, dims=[dims[0], max_r])

    def _find_radial_bins(self, dims, max_r):
        """Bin tensor positions based on distance from center.

        Args:
            dims (tuple of int): Tensor dimensions
            max_r (int): Maximum radial distance. Positions outside
                this distance are ignored.

        Returns:
            numpy.ndarray of int: Bin for each tensor position. Some
                bins may be greater than max_r. Its dimensions match
                dims.
            numpy.ndarray of int: Number of positions in each bin.
                It is 1D and with a length of max_r.

        """

        # Find bin for each position
        r2 = np.zeros([])
        for i, d in enumerate(dims):
            x = np.arange(d) - (d-1)/2
            r2 = np.expand_dims(r2, -1) + x**2
        r = np.sqrt(r2).astype(int)

        # Count number of positions in each bin
        # Note: Pad/truncate to max_r
        r_counts = np.bincount(r.flatten(), minlength=max_r)
        r_counts = r_counts[:max_r]

        return r, r_counts

# Test by computing radial profile for user-provided image
if __name__ == "__main__":

    # Imports
    import argparse
    import matplotlib.image

    # Command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'image', action='store', type=str,
        help='image file', metavar='FILE',
    )
    args = parser.parse_args()

    # Load image
    image = matplotlib.image.imread(args.image)
    if image.ndim == 2:
        image = np.expand_dims(image, 2)
    assert image.ndim == 3, f'failed to load 2D image from {args.image}'
    if image.shape[-1] == 1:
        image = np.tile(image, (1,1,3))
    elif image.shape[-1] == 4:
        image = image[:,:,:3]
    assert image.shape[-1] == 3, f'failed to load RGB image from {args.image}'
    image = np.transpose(image, (2,0,1))

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

    # Radial profile
    x = lbann.WeightsLayer(
        weights=lbann.Weights(
            lbann.ValueInitializer(values=image.flatten()),
        ),
        dims=image.shape,
    )
    max_r = image.shape[-1] // 2
    rprof = RadialProfile()(x, image.shape, max_r)
    rprof_slice = lbann.Slice(rprof, slice_points=[0,1,2,3])
    red = lbann.Identity(rprof_slice, name='red')
    green = lbann.Identity(rprof_slice, name='green')
    blue = lbann.Identity(rprof_slice, name='blue')

    # Construct model
    callbacks = [
        lbann.CallbackDumpOutputs(layers=['red', 'green', 'blue']),
    ]
    model = lbann.Model(
        epochs=0,
        layers=lbann.traverse_layer_graph([input_, rprof]),
        callbacks=callbacks,
    )

    # Run LBANN
    lbann.run(
        trainer=lbann.Trainer(mini_batch_size=1),
        model=model,
        data_reader=reader,
        optimizer=lbann.NoOptimizer(),
        job_name='lbann_radial_profile_test',
    )
