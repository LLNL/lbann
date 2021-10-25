import lbann
import lbann.modules

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: lbann.Pooling(num_dims = 2,
                                                            pool_strides_i = stride,
                                                            pool_pads_i = 1,
                                                            pool_mode = "avg"), #count_include_pad missing
    'max_pool_3x3': lambda C, stride, affine: lbann.Pooling(num_dims = 2,
                                                            pool_strides_i = stride, 
                                                            pool_pads_i = 1,
                                                            pool_mode = "max"),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),    
}


class ReLUConvBN(lbann.modules.Module):
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=False):
        # affine missing for BatchNorm
        super().__init__()
        self.relu = lbann.Relu
        self.conv = lbann.Convolution(num_dims = 2,
                                      num_output_channels = C_out,
                                      conv_dims_i = kernel_size,
                                      conv_strides_i = stride,
                                      conv_pads_i = padding,
                                      has_bias = False)
        self.batchnorm = lbann.BatchNormalization

    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        return self.batchnorm(x)

class DilConv(lbann.modules.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=False):
        # affine missing
        super().__init__()
        self.relu = lbann.Relu
        self.conv1 = lbann.Convolution(num_dims = 2,
                                      num_output_channels = C_in,
                                      conv_dims_i = kernel_size,
                                      conv_strides_i = stride,
                                      conv_pads_i = padding,
                                      conv_dilations_i = dilation,
                                      num_groups = C_in,
                                      has_bias = False)
        self.conv2 = lbann.Convolution(num_dims = 2,
                                       num_output_channels = C_out,
                                       conv_dims_i = kernel_size,
                                       conv_pads_i = 0,
                                       has_bias = False) 
        self.batchnorm = lbann.BatchNormalization

    def forward(self, x):
        x = self.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.batchnorm(x)

class SepConv(lbann.modules.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=False):
        super().__init__()
        self.relu = lbann.Relu
        self.conv1 = lbann.Convolution(num_dims = 2,
                                       num_output_channels = C_in,
                                       conv_dims_i = kernel_size,
                                       conv_strides_i = stride,
                                       conv_pads_i = padding,
                                       num_groups = C_in,
                                       has_bias = False)
        self.conv2 = lbann.Convolution(num_dims = 2,
                                       num_output_channels = C_in,
                                       conv_dims_i = 1,
                                       conv_pads_i = 0,
                                       has_bias = False)
        self.batchnorm = lbann.BatchNormalization # no of features from o/p channels of prev layer
        self.conv3 = lbann.Convolution(num_dims = 2,
                                       num_output_channels = C_in,
                                       conv_dims_i = kernel_size,
                                       conv_strides_i = 1,
                                       conv_pads_i = padding,
                                       num_groups = C_in,
                                       has_bias = False)
        self.conv4 = lbann.Convolution(num_dims = 2,
                                       num_output_channels = C_out,
                                       conv_dims_i = 1,
                                       conv_pads_i = 0,
                                       has_bias = False)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.batchnorm(x)

class Identity(lbann.modules.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Zero(lbann.modules.Module):

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.
        return x[:,:,::self.stride,::self.stride] * 0.

class FactorizedReduce(lbann.modules.Module):

    def __init__(self, C_in, C_out, affine=False):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = lbann.Relu
        self.conv1 = lbann.Convolution(num_dims = 2,
                                       num_output_channels = C_out // 2,
                                       conv_dims_i = 1,
                                       conv_strides_i = 2,
                                       conv_pads_i = 0,
                                       has_bias = False)
        self.batchnorm = lbann.BatchNormalization

    def forward(self, x):
        x = self.relu(x)
        out = lbann.Concatenation([self.conv1(x), self.conv2(x[:,:,1:,1:])], dim=0) #concatenate along first dim
        out = self.bn(out)
        return out

def main():
    R = ReLUConvBN(2, 4, 3, 1, 0)
    D = DilConv(2, 4, 3, 1, 0, 0)
    S = SepConv(2, 4, 3, 1, 0)
    I = Identity
    Z = Zero(1)
    F = FactorizedReduce(2, 4)

if __name__ == '__main__':
    main()   
