import lbann
import lbann.modules as nn
from lbann.modules import Module
from lbann.util import str_list


class CNN3D(Module):
    """docstring for CNN3D"""
    def __init__(self):
        super(CNN3D, self).__init__()

    def forward(self, x, fused=False):
        fc1 = nn.FullyConnectedModule(size=10)
        
        conv_1_kernel = str_list([7, 7, 7])
        conv_1_res_1_kernel = str_list([7, 7, 7])
        conv_1_res_2_kernel = str_list([7, 7, 7])
        conv_2_kernel = str_list([7, 7, 7])
        conv_3_kernel = str_list([5, 5, 5])

        conv_1_stride = str_list([2, 2, 2])
        conv_1_res_1_stride = str_list([1, 1, 1])
        conv_1_res_2_stride = str_list([1, 1, 1])
        conv_2_stride = str_list([3, 3, 3])
        conv_3_stride = str_list([2, 2, 2])

        avg_pool3d_ksize = str_list([2, 2, 2])
        avg_pool3d_stride = str_list([2, 2, 2])

        zero_padding = str_list([0, 0, 0])
        x = lbann.Convolution(x,
                              num_dims=3,
                              num_output_channels=64,
                              num_groups=1,
                              conv_dims=conv_1_kernel,
                              conv_strides=conv_1_stride,
                              conv_pads=str_list([3, 3, 3]),
                              has_bias=True,
                              has_vectors=True,
                              name="Conv_1")

        x = lbann.Relu(x,
                       name="Relu_1")

        x_1 = lbann.BatchNormalization(x,
                                       name="BN_1")
        x = lbann.Convolution(x_1,
                              num_dims=3,
                              num_output_channels=64,
                              num_groups=1,
                              conv_dims=conv_1_res_1_kernel,
                              conv_strides=conv_1_res_1_stride,
                              conv_pads=str_list([3, 3, 3]),
                              has_bias=True,
                              has_vectors=True,
                              name="Conv_1_res_1")
        x = lbann.Relu(x,
                       name="Relu_res_1")
        x = lbann.BatchNormalization(x,
                                     name="BN_Res_1")

        x_2 = lbann.Sum(x, x_1, name="Conv_Layer_1_+Conv_Layer_Res_1")

        x = lbann.Convolution(x_2,
                              num_dims=3,
                              num_output_channels=64,
                              num_groups=1,
                              conv_dims=conv_1_res_2_kernel,
                              conv_strides=conv_1_res_2_stride,
                              conv_pads=str_list([3, 3, 3]),
                              has_bias=True,
                              has_vectors=True,
                              name="Conv_1_res_2")
        x = lbann.Relu(x,
                       name="Relu_res_2")

        x = lbann.BatchNormalization(x,
                                     name="BN_res_2")

        x_3 = lbann.Sum(x, x_1, name="Conv_Layer_1+Conv_Layer_3")

        x = lbann.Convolution(x_3,
                              num_dims=3,
                              num_output_channels=96,
                              num_groups=1,
                              conv_dims=conv_2_kernel,
                              conv_strides=conv_2_stride,
                              conv_pads=zero_padding,
                              has_bias=True,
                              has_vectors=True,
                              name="Conv_2")
        x = lbann.Relu(x,
                       name="Relu_2")
        x = lbann.BatchNormalization(x,
                                     name="BN_2")

        x = lbann.Pooling(x,
                          num_dims=3,
                          pool_dims=avg_pool3d_ksize,
                          pool_strides=avg_pool3d_stride,
                          pool_pads=zero_padding,
                          has_vectors=True,
                          pool_mode="average_no_pad",
                          name="avg_pooling_1")

        x = lbann.Convolution(x,
                              num_dims=3,
                              num_output_channels=128,
                              num_groups=1,
                              conv_dims=conv_3_kernel,
                              conv_strides=conv_3_stride,
                              conv_pads=str_list([1, 1, 1]),
                              has_bias=True,
                              has_vectors=True,
                              name="Conv_3")
        x = lbann.Relu(x,
                       name="Relu_3")
        x = lbann.BatchNormalization(x,
                                     name="BN_3")

        x = lbann.Pooling(x,
                          num_dims=3,
                          pool_dims=avg_pool3d_ksize,
                          pool_strides=avg_pool3d_stride,
                          pool_pads=str_list([1, 1, 1]),
                          has_vectors=True,
                          pool_mode="average_no_pad",
                          name="avg_pooling_2")

        if (fused):
            x = fc1(x)
        else:
            fc2 = nn.FullyConnectedModule(size=1)
            x = fc2(x)
        return x
