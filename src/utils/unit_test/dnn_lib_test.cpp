////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

// MUST include this
#include "Catch2BasicSupport.hpp"

#include <lbann/utils/dnn_enums.hpp>
#include <lbann/utils/dnn_lib/helpers.hpp>

#include <lbann/utils/dnn_lib/convolution.hpp>
#include <lbann/utils/dnn_lib/dropout.hpp>
#include <lbann/utils/dnn_lib/local_response_normalization.hpp>
#include <lbann/utils/dnn_lib/pooling.hpp>
#include <lbann/utils/dnn_lib/softmax.hpp>

using namespace lbann;

TEMPLATE_TEST_CASE("Tensor operations", "[dnn_lib]", float)
{
  int N = 128, c = 4, h = 128, w = 128;
  const dnn_lib::ScalingParamType<TestType> alpha = 1.;
  const dnn_lib::ScalingParamType<TestType> beta = 0.;

  dnn_lib::TensorDescriptor aDesc;
  aDesc.set(dnn_lib::get_data_type<TestType>(), {N, c, h, w});
  El::Matrix<TestType, El::Device::GPU> A(c * h * w, N);
  dnn_lib::TensorDescriptor cDesc;
  cDesc.set(dnn_lib::get_data_type<TestType>(), {N, c, h, w});
  El::Matrix<TestType, El::Device::GPU> C(c * h * w, N);

  SECTION("Add tensor")
  {
    REQUIRE_NOTHROW(dnn_lib::add_tensor(alpha, aDesc, A, beta, cDesc, C));
  }
}

TEMPLATE_TEST_CASE("Computing convolution layers", "[dnn_lib]", float)
{
  // Parameters describing convolution and tensor sizes
  int N = 128;
  int in_c = 1, in_h = 5, in_w = 5;
  int out_c = 1, out_h = 6, out_w = 6;
  int filt_k = 1, filt_c = 1, filt_h = 2, filt_w = 2;
  int pad_h = 1, pad_w = 1;
  int str_h = 1, str_w = 1;
  int dil_h = 1, dil_w = 1;

  // Scaling parameters
  const dnn_lib::ScalingParamType<TestType> alpha = 1.;
  const dnn_lib::ScalingParamType<TestType> beta = 0.;

  // Convolution Descriptor
  dnn_lib::ConvolutionDescriptor convDesc;
  convDesc.set({pad_h, pad_w},
               {str_h, str_w},
               {dil_h, dil_w},
               dnn_lib::get_data_type<TestType>());

  // Input/Output Tensors and descriptors
  dnn_lib::FilterDescriptor wDesc;
  wDesc.set(dnn_lib::get_data_type<TestType>(),
            dnn_lib::DNN_TENSOR_NCHW,
            {filt_k, filt_c, filt_h, filt_w});
  El::Matrix<TestType, El::Device::GPU> w(filt_k * filt_c * filt_h * filt_w, N);
  dnn_lib::FilterDescriptor dwDesc;
  dwDesc.set(dnn_lib::get_data_type<TestType>(),
             dnn_lib::DNN_TENSOR_NCHW,
             {filt_k, filt_c, filt_h, filt_w});
  El::Matrix<TestType, El::Device::GPU> dw(filt_k * filt_c * filt_h * filt_w,
                                           N);
  dnn_lib::TensorDescriptor dbDesc;
  dbDesc.set(dnn_lib::get_data_type<TestType>(), {1, out_c, 1, 1});
  El::Matrix<TestType, El::Device::GPU> db(out_c, 1);
  dnn_lib::TensorDescriptor xDesc;
  xDesc.set(dnn_lib::get_data_type<TestType>(), {N, in_c, in_h, in_w});
  El::Matrix<TestType, El::Device::GPU> x(in_c * in_h * in_w, N);
  dnn_lib::TensorDescriptor dxDesc;
  dxDesc.set(dnn_lib::get_data_type<TestType>(), {N, in_c, in_h, in_w});
  El::Matrix<TestType, El::Device::GPU> dx(in_c * in_h * in_w, N);
  dnn_lib::TensorDescriptor yDesc;
  yDesc.set(dnn_lib::get_data_type<TestType>(), {N, out_c, out_h, out_w});
  El::Matrix<TestType, El::Device::GPU> y(out_c * out_h * out_w, N);
  dnn_lib::TensorDescriptor dyDesc;
  dyDesc.set(dnn_lib::get_data_type<TestType>(), {N, out_c, out_h, out_w});
  El::Matrix<TestType, El::Device::GPU> dy(out_c * out_h * out_w, N);

  // Workspace
  size_t workspace_size = (1 << 30) / sizeof(TestType);
  El::Matrix<TestType, El::Device::GPU> workSpace(workspace_size, 1);

  // Convolution Algorithm
  SECTION("convolution forward")
  {
    dnn_lib::fwd_conv_alg_config alg_config =
      dnn_lib::get_fwd_algorithm(true,
                                 true,
                                 xDesc,
                                 x.LockedBuffer(),
                                 wDesc,
                                 w.LockedBuffer(),
                                 convDesc,
                                 yDesc,
                                 y.Buffer(),
                                 workspace_size,
                                 workSpace.Buffer());
    convDesc.set_math_mode(alg_config.second);
    REQUIRE_NOTHROW(dnn_lib::convolution_forward(alpha,
                                                 xDesc,
                                                 x,
                                                 wDesc,
                                                 w,
                                                 convDesc,
                                                 alg_config.first,
                                                 workSpace,
                                                 beta,
                                                 yDesc,
                                                 y));
  }

  SECTION("convolution backward data")
  {
    dnn_lib::bwd_data_conv_alg_config alg_config =
      dnn_lib::get_bwd_data_algorithm(true,
                                      true,
                                      wDesc,
                                      w.LockedBuffer(),
                                      dyDesc,
                                      dy.LockedBuffer(),
                                      convDesc,
                                      dxDesc,
                                      dx.Buffer(),
                                      workspace_size,
                                      workSpace.Buffer());
    convDesc.set_math_mode(alg_config.second);
    REQUIRE_NOTHROW(dnn_lib::convolution_backward_data(alpha,
                                                       wDesc,
                                                       w,
                                                       dyDesc,
                                                       dy,
                                                       convDesc,
                                                       alg_config.first,
                                                       workSpace,
                                                       beta,
                                                       dxDesc,
                                                       dx));
  }

  SECTION("convolution backward bias")
  {
    REQUIRE_NOTHROW(
      dnn_lib::convolution_backward_bias(alpha, dyDesc, dy, beta, dbDesc, db));
  }

  SECTION("convolution backward filter")
  {
    dnn_lib::bwd_filter_conv_alg_config alg_config =
      dnn_lib::get_bwd_filter_algorithm(true,
                                        true,
                                        xDesc,
                                        x.LockedBuffer(),
                                        dyDesc,
                                        dy.LockedBuffer(),
                                        convDesc,
                                        dwDesc,
                                        dw.Buffer(),
                                        workspace_size,
                                        workSpace.Buffer());
    convDesc.set_math_mode(alg_config.second);
    REQUIRE_NOTHROW(dnn_lib::convolution_backward_filter(alpha,
                                                         xDesc,
                                                         x,
                                                         dyDesc,
                                                         dy,
                                                         convDesc,
                                                         alg_config.first,
                                                         workSpace,
                                                         beta,
                                                         dwDesc,
                                                         dw));
  }
}

TEMPLATE_TEST_CASE("Computing dropout layers", "[dnn_lib]", float)
{
  // Parameters describing dropout and tensor sizes
  int N = 128, c = 1, h = 128, w = 128;
  float dropout = 0.25;
  int seed = 1337;

  // Dropout descriptor
  size_t states_size = dnn_lib::get_dropout_states_size() / sizeof(TestType);
  El::Matrix<TestType, El::Device::GPU> states(states_size, 1);
  dnn_lib::DropoutDescriptor dropoutDesc;
  dropoutDesc.set(dropout,
                  states.Buffer(),
                  states_size * sizeof(TestType),
                  seed);

  // Input/Output tensors and descriptors
  dnn_lib::TensorDescriptor xDesc;
  xDesc.set(dnn_lib::get_data_type<TestType>(), {N, c, h, w});
  El::Matrix<TestType, El::Device::GPU> x(c * h * w, N);
  dnn_lib::TensorDescriptor dxDesc;
  dxDesc.set(dnn_lib::get_data_type<TestType>(), {N, c, h, w});
  El::Matrix<TestType, El::Device::GPU> dx(c * h * w, N);
  dnn_lib::TensorDescriptor yDesc;
  yDesc.set(dnn_lib::get_data_type<TestType>(), {N, c, h, w});
  El::Matrix<TestType, El::Device::GPU> y(c * h * w, N);
  dnn_lib::TensorDescriptor dyDesc;
  dyDesc.set(dnn_lib::get_data_type<TestType>(), {N, c, h, w});
  El::Matrix<TestType, El::Device::GPU> dy(c * h * w, N);

  // Workspace
  size_t workspace_size =
    dnn_lib::get_dropout_reserve_space_size(xDesc) / sizeof(TestType);
  El::Matrix<TestType, El::Device::GPU> workSpace(workspace_size, 1);

  SECTION("dropout forward")
  {
    REQUIRE_NOTHROW(
      dnn_lib::dropout_forward(dropoutDesc, xDesc, x, yDesc, y, workSpace));
  }
  SECTION("dropout backward")
  {
    REQUIRE_NOTHROW(dnn_lib::dropout_backward(dropoutDesc,
                                              dyDesc,
                                              dy,
                                              dxDesc,
                                              dx,
                                              workSpace));
  }
}

TEMPLATE_TEST_CASE("Computing LRN layers", "[dnn_lib]", float)
{
  // Parameters describing LRN and tensor sizes
  int N = 128, c = 3, h = 128, w = 128;
  int lrnN = 4;
  double lrnAlpha = 0.0001, lrnBeta = 0.75, lrnK = 2.0;

  // Scaling parameters
  const dnn_lib::ScalingParamType<TestType> alpha = 1.;
  const dnn_lib::ScalingParamType<TestType> beta = 0.;

  // LRN Descriptor
  dnn_lib::LRNDescriptor normDesc;
  normDesc.set(lrnN, lrnAlpha, lrnBeta, lrnK);

  // Input/Output tensors and descriptors
  dnn_lib::TensorDescriptor xDesc;
  xDesc.set(dnn_lib::get_data_type<TestType>(), {N, c, h, w});
  El::Matrix<TestType, El::Device::GPU> x(c * h * w, N);
  dnn_lib::TensorDescriptor dxDesc;
  dxDesc.set(dnn_lib::get_data_type<TestType>(), {N, c, h, w});
  El::Matrix<TestType, El::Device::GPU> dx(c * h * w, N);
  dnn_lib::TensorDescriptor yDesc;
  yDesc.set(dnn_lib::get_data_type<TestType>(), {N, c, h, w});
  El::Matrix<TestType, El::Device::GPU> y(c * h * w, N);
  dnn_lib::TensorDescriptor dyDesc;
  dyDesc.set(dnn_lib::get_data_type<TestType>(), {N, c, h, w});
  El::Matrix<TestType, El::Device::GPU> dy(c * h * w, N);

  // Workspace
  size_t workspace_size = dnn_lib::get_lrn_ws_size(yDesc);
  El::Matrix<TestType, El::Device::GPU> workSpace(workspace_size, 1);

  SECTION("LRN forward")
  {
    REQUIRE_NOTHROW(dnn_lib::lrn_cross_channel_forward(normDesc,
                                                       alpha,
                                                       xDesc,
                                                       x,
                                                       beta,
                                                       yDesc,
                                                       y,
                                                       workSpace));
  }

  SECTION("LRN backward")
  {
    REQUIRE_NOTHROW(dnn_lib::lrn_cross_channel_backward(normDesc,
                                                        alpha,
                                                        yDesc,
                                                        y,
                                                        dyDesc,
                                                        dy,
                                                        xDesc,
                                                        x,
                                                        beta,
                                                        dxDesc,
                                                        dx,
                                                        workSpace));
  }
}

TEMPLATE_TEST_CASE("Computing pooling layers", "[dnn_lib]", float)
{
  // Parameters describing pooling and tensor sizes
  int N = 128, c = 3, h = 128, w = 128;
  std::vector<int> windowDims{2, 2};
  std::vector<int> padding{1, 1};
  std::vector<int> stride{1, 1};

  // Scaling parameters
  const dnn_lib::ScalingParamType<TestType> alpha = 1.;
  const dnn_lib::ScalingParamType<TestType> beta = 0.;

  // Pooling descriptor
  dnn_lib::PoolingDescriptor poolingDesc;
  poolingDesc.set(pooling_mode::MAX,
                  dnn_lib::DNN_PROPAGATE_NAN,
                  windowDims,
                  padding,
                  stride);

  // Input/Output tensors and descriptors
  dnn_lib::TensorDescriptor xDesc;
  xDesc.set(dnn_lib::get_data_type<TestType>(), {N, c, h, w});
  El::Matrix<TestType, El::Device::GPU> x(c * h * w, N);
  dnn_lib::TensorDescriptor dxDesc;
  dxDesc.set(dnn_lib::get_data_type<TestType>(), {N, c, h, w});
  El::Matrix<TestType, El::Device::GPU> dx(c * h * w, N);
  dnn_lib::TensorDescriptor yDesc;
  yDesc.set(dnn_lib::get_data_type<TestType>(), {N, c, h, w});
  El::Matrix<TestType, El::Device::GPU> y(c * h * w, N);
  dnn_lib::TensorDescriptor dyDesc;
  dyDesc.set(dnn_lib::get_data_type<TestType>(), {N, c, h, w});
  El::Matrix<TestType, El::Device::GPU> dy(c * h * w, N);

  // Workspace
  size_t workspace_size =
    dnn_lib::get_pooling_ws_size(poolingDesc, yDesc) / sizeof(TestType);
  El::Matrix<TestType, El::Device::GPU> workSpace(workspace_size, 1);

  SECTION("Pooling forward")
  {
    REQUIRE_NOTHROW(dnn_lib::pooling_forward(poolingDesc,
                                             alpha,
                                             xDesc,
                                             x,
                                             beta,
                                             yDesc,
                                             y,
                                             workSpace));
  }

  SECTION("Pooling backward")
  {
    REQUIRE_NOTHROW(dnn_lib::pooling_backward(poolingDesc,
                                              alpha,
                                              yDesc,
                                              y,
                                              dyDesc,
                                              dy,
                                              xDesc,
                                              x,
                                              beta,
                                              dxDesc,
                                              dx,
                                              workSpace));
  }
}

TEMPLATE_TEST_CASE("Computing softmax layers", "[dnn_lib]", float)
{
  // Parmeters describing tensor sizes
  int N = 128, labels_n = 10;

  // Scaling parameters
  const dnn_lib::ScalingParamType<TestType> alpha = 1.;
  const dnn_lib::ScalingParamType<TestType> beta = 0.;

  // Input/Output tensors and descriptors
  dnn_lib::TensorDescriptor xDesc;
  xDesc.set(dnn_lib::get_data_type<TestType>(), {N, labels_n, 1});
  El::Matrix<TestType, El::Device::GPU> x(labels_n, N);
  dnn_lib::TensorDescriptor dxDesc;
  dxDesc.set(dnn_lib::get_data_type<TestType>(), {N, labels_n, 1});
  El::Matrix<TestType, El::Device::GPU> dx(labels_n, N);
  dnn_lib::TensorDescriptor yDesc;
  yDesc.set(dnn_lib::get_data_type<TestType>(), {N, labels_n, 1});
  El::Matrix<TestType, El::Device::GPU> y(labels_n, N);
  dnn_lib::TensorDescriptor dyDesc;
  dyDesc.set(dnn_lib::get_data_type<TestType>(), {N, labels_n, 1});
  El::Matrix<TestType, El::Device::GPU> dy(labels_n, N);

  SECTION("softmax forward")
  {
    REQUIRE_NOTHROW(dnn_lib::softmax_forward(alpha,
                                             xDesc,
                                             x,
                                             beta,
                                             yDesc,
                                             y,
                                             softmax_mode::CHANNEL,
                                             softmax_alg::ACCURATE));
  }
  SECTION("softmax backward")
  {
    REQUIRE_NOTHROW(dnn_lib::softmax_backward(alpha,
                                              yDesc,
                                              y,
                                              dyDesc,
                                              dy,
                                              beta,
                                              dxDesc,
                                              dx,
                                              softmax_mode::CHANNEL,
                                              softmax_alg::ACCURATE));
  }
}
