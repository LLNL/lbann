import lbann
import os
import os.path

# ----------------------------------
# Construct Graph
# ----------------------------------
def gen_layers(latent_dim, number_of_atoms):
    ''' Generates the model for the 3D Convolutional Auto Encoder.

                returns the Directed Acyclic Graph (DAG) that the lbann
        model will run on.
    '''

    input_ = lbann.Input( target_mode = "reconstruction")
    tensors = lbann.Identity(input_)

    tensors = lbann.Reshape(tensors, dims=[11, 32, 32, 32], name="Sample")
    # Input tensor shape is  (number_of_atoms)x32x32x32

    # Encoder

    x = lbann.Identity(tensors)
    for i in range(4):
        out_channels = latent_dim // (2 ** (3-i))

        x = lbann.Convolution(x,
                              num_dims = 3,
                              out_channels = out_channels,
                              groups = 1,
                              kernel_size = 4,
                              stride = 2,
                              dilation = 1,
                              padding = 1,
                              has_bias = True,
                              name="Conv_{0}".format(i))

        x = lbann.BatchNormalization(x, name="Batch_NORM_{0}".format(i+1))
        x = lbann.LeakyRelu(x, name="Conv_{0}_Activation".format(i+1))

    # Shape: (latent_dim)x2x2x2
    encoded = lbann.Convolution(x,
                                num_dims = 3,
                                out_channels = latent_dim,
                                groups = 1,
                                kernel_size = 2,
                                stride = 2,
                                dilation = 1,
                                padding  = 0,
                                has_bias = True,
                                name ="encoded")

    # Shape: (latent_dim)1x1x1

    # Decoder

    x = lbann.Deconvolution(encoded,
                            num_dims = 3,
                            out_channels = number_of_atoms * 16,
                            groups = 1,
                            kernel_size = 4,
                            padding = 0,
                            stride = 2,
                            dilation = 1,
                            has_bias = True,
                            name="Deconv_1")
    x = lbann.BatchNormalization(x, name="BN_D1")
    x = lbann.Tanh(x, name="Deconv_1_Activation")

    for i in range(3):
        out_channels = number_of_atoms * (2 ** (2-i))
        x = lbann.Deconvolution(x,
                                num_dims = 3,
                                out_channels = out_channels,
                                groups = 1,
                                kernel_size = 4,
                                padding = 1,
                                stride = 2,
                                dilation = 1,
                                has_bias = True,
                                name="Deconv_{0}".format(i+2))
        x = lbann.BatchNormalization(x, name="BN_D{0}".format(i+2))

        if (i != 2): #Save the last activation layer because we want to dump the outputs
            x = lbann.Tanh(x, name="Deconv_{0}_Activation".format(i+2))

    decoded = lbann.Tanh(x,
                 name = "decoded")

    img_loss = lbann.MeanSquaredError([decoded, tensors])

    metrics = [lbann.Metric(img_loss, name='recon_error')]
    # ----------------------------------
    # Set up DAG
    # ----------------------------------

    layers = lbann.traverse_layer_graph(input_) #Generate Model DAG
    return layers, img_loss, metrics
def make_data_reader():
    reader = lbann.reader_pb2.DataReader()
    _reader = reader.reader.add()
    _reader.name = 'python'
    _reader.role = 'train'
    _reader.shuffle = True
    _reader.fraction_of_data_to_use = 1.0
    _reader.python.module = 'dataset'
    _reader.python.module_dir = os.path.dirname(os.path.realpath(__file__))
    _reader.python.sample_function = 'get_train'
    _reader.python.num_samples_function = 'num_train_samples'
    _reader.python.sample_dims_function = 'sample_dims'

    return reader
