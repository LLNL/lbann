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

class Generator(lm.Module):
    def __init__(self, name='gen_'):
        self.layers = [
            lm.Convolution2dModule(64, 4, transpose=True), # 4x4
            lm.Convolution2dModule(32, 3, padding=1), # 8x8
            lm.Convolution2dModule(16, 3, padding=1), # 16x16
            lm.Convolution2dModule(1, 3, padding=1), # 32x32
        ]

        for i, l in enumerate(self.layers):
            l.name = name + f'conv_{i}'

        self.name = name

    def forward(self, x):
        x = lbann.Reshape(x, dims=[-1, 1, 1])

        in_size = 4
        for i, l in enumerate(self.layers):
            if i > 0:
                x = Upsample(in_size)(x)
                in_size *= 2
                x = lbann.LeakyRelu(x, negative_slope=0.2)
            
            x = l(x)

        x = lbann.Sigmoid(x, name=self.name + 'out')

        return x

class Discriminator(lm.Module):
    def __init__(self, name='disc_'):
        self.layers = [
            lm.Convolution2dModule(16, 3, padding=1), # 32x32
            lm.Convolution2dModule(32, 3, padding=1), # 16x16
            lm.Convolution2dModule(64, 3, padding=1), # 8x8
            lm.Convolution2dModule(1, 4), # 4x4
        ]

        for i, l in enumerate(self.layers):
            l.name = name + f'conv_{i}'
    
    def forward(self, x):
        for i, l in enumerate(self.layers):
            if i > 0:
                x = Downsample()(x)
                x = lbann.LeakyRelu(x, negative_slope=0.2)
            
            x = l(x)

        x = lbann.Reshape(x, dims=[-1])

        return x

def build_model(num_epochs=100):
    # Setup the networks.
    gen = Generator()
    disc = Discriminator()

    # Get real training samples and use generator to map latent noise vectors 
    # to fake samples.
    real = lbann.Reshape(lbann.Input(data_field='samples'), dims=[1, 32, 32])
    fake = gen(lbann.Gaussian(mean=0, stdev=1, neuron_dims=[16]))

    # Get discriminator outputs for real and fake samples.
    real_out = disc(real)
    fake_out = disc(fake)

    # Setup the binary switches to alternate between generator and
    # discriminator losses.
    disc_switch = lbann.BinarySwitch(num_neurons=[1], name='disc_switch')
    gen_switch = lbann.BinarySwitch(num_neurons=[1], name='gen_switch')

    # Compute the loss functions for the generator and discriminator (hinge loss).
    real_loss = lbann.Relu(lbann.ConstantSubtract(real_out, constant=1))
    fake_loss = lbann.Relu(lbann.AddConstant(fake_out, constant=1))

    disc_loss = lbann.Multiply(lbann.Add(real_loss, fake_loss), disc_switch) # 0 if disc_switch layer is frozen.
    gen_loss = lbann.Multiply(lbann.Negative(fake_out), gen_switch) # 0 if gen_switch layer is frozen.

    loss = lbann.Add(disc_loss, gen_loss)

    layers = list(lbann.traverse_layer_graph(loss))

    # Get the generator and discriminator layers for the AlternateUpdates 
    # callback.
    disc_layers = [disc_switch.name]
    gen_layers = [gen_switch.name]
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
                                batch_interval=1000,
                                format='npy'
                            )
                        ])
    
    return model