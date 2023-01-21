import lbann
import lbann.modules as lm
import numpy as np

class Upsample(lm.Module):
    '''Nearest-neighbor upsample a 2D tensor by a factor of 2 using the native gather layer.'''

    def __init__(self, in_size):
        self.in_size = in_size
        
        x = np.stack([np.arange(self.in_size)] * 2, axis=1).ravel()
        i, j = np.meshgrid(x, x, indexing='ij')
        inds = i * self.in_size + j

        self.inds = lbann.WeightsLayer(
            weights=lbann.Weights(
                lbann.ValueInitializer(values=inds.flatten()),
                optimizer=lbann.NoOptimizer(),
            ),
            dims=[len(inds.ravel())],
        )

    def forward(self, x):
        x = lbann.Reshape(x, dims=[-1, self.in_size * self.in_size])
        x = lbann.Gather(x, self.inds, axis=1)
        x = lbann.Reshape(x, dims=[-1, 2 * self.in_size, 2 * self.in_size])
        return x

class Downsample(lm.Module):
    '''Downsample a 2D tensor using average pooling.'''

    def forward(self, x):
        return lbann.Pooling(
            x,
            pool_mode='average',
            num_dims=2,
            has_vectors=False,
            pool_dims_i=2,
            pool_pads_i=0,
            pool_strides_i=2
        )

class ResBlock(lm.Module):
    def __init__(self, mid_channels, out_channels):
        self.conv1 = lm.Convolution2dModule(mid_channels, 3, padding=1)
        self.conv2 = lm.Convolution2dModule(out_channels, 3, padding=1)
        self.skip = lm.Convolution2dModule(out_channels, 1)

    def forward(self, x):
        o = self.conv1(lbann.LeakyRelu(x, negative_slope=0.2))
        o = self.conv2(lbann.LeakyRelu(o, negative_slope=0.2))
        o = lbann.Add(lbann.Scale(o, constant=0.1), self.skip(x))
        return o

class Generator(lm.Module):
    def __init__(self, name='gen_'):
        self.init_layer = lm.FullyConnectedModule(512*4*4, name=name+'fc')
        self.final_conv = lm.Convolution2dModule(3, 3, padding=1, name=name+'final_conv')

        self.layers = [
            ResBlock(512, 512), # 4 x 4
            ResBlock(256, 256), # 8 x 8
            ResBlock(128, 128), # 16 x 16
            ResBlock(64, 64), # 32 x 32
            ResBlock(32, 32), # 64 x 64
            ResBlock(16, 16), # 128 x 128
        ]

        self.up_layers = [
            Upsample(4),
            Upsample(8),
            Upsample(16),
            Upsample(32),
            Upsample(64),
            None
        ]

        for i, l in enumerate(self.layers):
            l.conv1.name = name + f'conv_1_{i}'
            l.conv2.name = name + f'conv_2_{i}'
            l.skip.name = name + f'skip_{i}'
    
    def forward(self, x):
        x = lbann.LeakyRelu(lbann.Reshape(self.init_layer(x), dims=[-1, 4, 4]), negative_slope=0.2)

        for l, ul in zip(self.layers, self.up_layers):
            x = l(x)
            if ul:
                x = ul(x)

        return lbann.Sigmoid(self.final_conv(lbann.LeakyRelu(x, negative_slope=0.2)), name='gen_out')

class Discriminator(lm.Module):
    def __init__(self, name='disc_'):
        self.init_conv = lm.Convolution2dModule(16, 3, padding=1, name=name+'init_conv')
        self.final_layer = lm.FullyConnectedModule(1, name=name+'fc')

        self.layers = [
            ResBlock(16, 32), # 128 x 128
            ResBlock(32, 64), # 64 x 64
            ResBlock(64, 128), # 32 x 32
            ResBlock(128, 256), # 16 x 16
            ResBlock(256, 512), # 8 x 8
            ResBlock(512, 512), # 4 x 4
        ]

        self.down_layers = [
            Downsample(),
            Downsample(),
            Downsample(),
            Downsample(),
            Downsample(),
            None
        ]

        for i, l in enumerate(self.layers):
            l.conv1.name = name + f'conv_1_{i}'
            l.conv2.name = name + f'conv_2_{i}'
            l.skip.name = name + f'skip_{i}'
    
    def forward(self, x):
        x = self.init_conv(lbann.AddConstant(lbann.Scale(x, constant=2), constant=-1))

        for l, dl in zip(self.layers, self.down_layers):
            x = l(x)
            if dl:
                x = dl(x)
        
        x = self.final_layer(lbann.Reshape(lbann.LeakyRelu(x, negative_slope=0.2), dims=[-1]))

        return x

def build_model(num_epochs):
    # Setup the networks.
    gen = Generator()
    disc = Discriminator()

    # Get real training samples and use generator to map latent noise vectors 
    # to fake samples.
    real = lbann.Reshape(lbann.Input(data_field='samples'), dims=[3, 128, 128])
    fake = gen(lbann.Gaussian(mean=0, stdev=1, neuron_dims=[64]))

    # Get discriminator outputs for real and fake samples.
    real_out = disc(real)
    fake_out = disc(fake)

    # Compute the loss functions for the generator and discriminator 
    # (non-saturating loss).
    real_loss = lbann.Softplus(lbann.Negative(real_out))
    fake_loss = lbann.Softplus(fake_out)

    # Setup the switches to alternate between generator and discriminator
    # losses.
    disc_loss = lbann.IdentityZero(lbann.Add(real_loss, fake_loss))
    gen_loss = lbann.IdentityZero(lbann.Softplus(lbann.Negative(fake_out)))

    loss = lbann.Add(disc_loss, gen_loss)

    layers = list(lbann.traverse_layer_graph(loss))

    # Get the generator and discriminator layers for the AlternateUpdates 
    # callback.
    disc_layers = [disc_loss.name]
    gen_layers = [gen_loss.name]
    for l in layers:
        if l.weights and 'disc_' in l.name:
            disc_layers.append(l.name)

        if l.weights and 'gen_' in l.name:
            gen_layers.append(l.name)

    model = lbann.Model(num_epochs,
                        layers=layers,
                        objective_function=loss,
                        callbacks=[
                            lbann.CallbackPrint(), 
                            lbann.CallbackTimer(),
                            lbann.CallbackAlternateUpdates(
                                layers_1=' '.join(disc_layers),
                                layers_2=' '.join(gen_layers),
                                iters_1=1,
                                iters_2=1
                            ),
                            lbann.CallbackDumpOutputs(
                                layers='gen_out',
                                execution_modes='train',
                                directory='dump_outs',
                                batch_interval=500,
                                format='npy'
                            )
                        ])
    
    return model