import lbann
import lbann.modules

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: Pool(mode = "avg"),
    'max_pool_3x3': lambda C, stride, affine: Pool(mode = "max"),
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
        self.conv = lbann.modules.Convolution2dModule(out_channels = C_out,
                                                      kernel_size = kernel_size,
                                                      stride = stride,
                                                      padding = padding,
                                                      bias = False)

    def forward(self, x):
        x = lbann.Relu(x)
        x = self.conv(x)
        x = lbann.BatchNormalization(x)
        return x

class DilConv(lbann.modules.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=False):
        # affine missing
        super().__init__()
        self.conv1 = lbann.modules.Convolution2dModule(out_channels = C_in,
                                                       kernel_size = kernel_size,
                                                       stride = stride,
                                                       padding = padding,
                                                       dilation = dilation,
                                                       groups = C_in,
                                                       bias = False)
        self.conv2 = lbann.modules.Convolution2dModule(out_channels = C_out,
                                                       kernel_size = kernel_size,
                                                       padding = 0,
                                                       bias = False) 

    def forward(self, x):
        x = lbann.Relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = lbann.BatchNormalization(x)
        return x

class SepConv(lbann.modules.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=False):
        super().__init__()
        self.relu = lbann.Relu
        self.conv1 = lbann.modules.Convolution2dModule(out_channels = C_in,
                                                       kernel_size = kernel_size,
                                                       stride = stride,
                                                       padding = padding,
                                                       groups = C_in,
                                                       bias = False)
        self.conv2 = lbann.modules.Convolution2dModule(out_channels = C_in,
                                                       kernel_size = 1,
                                                       padding = 0,
                                                       bias = False)
        self.batchnorm = lbann.BatchNormalization # no of features from o/p channels of prev layer
        self.conv3 = lbann.modules.Convolution2dModule(out_channels = C_in,
                                                       kernel_size = kernel_size,
                                                       stride = 1,
                                                       padding = padding,
                                                       groups = C_in,
                                                       bias = False)
        self.conv4 = lbann.modules.Convolution2dModule(out_channels = C_out,
                                                       kernel_size = 1,
                                                       padding = 0,
                                                       bias = False)

    def forward(self, x):
        x = lbann.Relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = lbann.BatchNormalization(x)
        x = lbann.Relu(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = lbann.BatchNormalization(x)
        return x

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
        self.conv1 = lbann.modules.Convolution2dModule(out_channels = C_out // 2,
                                                       kernel_size = 1,
                                                       stride = 2,
                                                       padding = 0,
                                                       bias = False)
        #NSGA repo defines conv2 with exact specs as conv1 which I dont think is necessary

    def forward(self, x):
        x = lbann.Relu(x)
        x = lbann.Concatenation([self.conv1(x), self.conv1(x[:,:,1:,1:])], dim=0) #concatenate along first dim
        x = lbann.BatchNormalization(x)
        return x

class Pool(lbann.modules.Module):

    def __init__(self, mode):
        self.mode = mode

    def forward(self, x):
        if self.mode == "avg":
            return lbann.Pooling(x, 
                                 num_dims = 2,
                                 pool_dims_i = 3,
                                 pool_strides_i = stride,
                                 pool_pads_i = 1,
                                 pool_mode = "avg") #count_include_pad missing
        elif self.mode == "max":
            return lbann.Pooling(x,
                                 num_dims = 2,
                                 pool_dims_i = 3,
                                 pool_strides_i = stride,
                                 pool_pads_i = 1,
                                 pool_mode = "max")
                                  
def main():
    R = ReLUConvBN(2, 4, 3, 1, 0)
    D = DilConv(2, 4, 3, 1, 0, 0)
    S = SepConv(2, 4, 3, 1, 0)
    I = Identity
    Z = Zero(1)
    F = FactorizedReduce(2, 4)
    P = Pool("avg")

if __name__ == '__main__':
    main()   
