////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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
#ifndef LBANN_UTILS_DNN_LIB_ONEDNN_SOFTMAX_HPP_
#define LBANN_UTILS_DNN_LIB_ONEDNN_SOFTMAX_HPP_

#include "lbann/utils/dnn_enums.hpp"
#include "lbann/utils/dnn_lib/helpers.hpp"
#include "lbann/utils/dnn_lib/onednn.hpp"
#include "lbann/utils/sync_info_helpers.hpp"

#if !defined(LBANN_HAS_ONEDNN)
static_assert(false,
              "This file should not be included unless "
              "OneDNN support is enabled.");
#endif // !defined(LBANN_HAS_ONEDNN)

namespace lbann
{

#if defined LBANN_HAS_ONEDNN

template <El::Device D>
template <typename DataT, typename ScalarT>
void onednn_backend<D>::softmax_forward(
  ScalarT const& alpha_in,
  TensorDescriptor const& xDesc,
  El::Matrix<DataT, D> const& x,
  ScalarT const& beta_in,
  TensorDescriptor const& yDesc,
  El::Matrix<DataT, D>& y,
  El::SyncInfo<D> const& si,
  softmax_mode mode,
  softmax_alg alg)
{
  // Short-circuit the function if actually looking for logsoftmax. Do
  // this first since that function will do all this same
  // error-checking, too.
  if (alg == softmax_alg::LOG)
    return logsoftmax_forward(alpha_in,
                              xDesc,
                              x,
                              beta_in,
                              yDesc,
                              y,
                              si,
                              mode);

  // Start the Softmax operation

  constexpr int softmax_axis = 1; // Columns of x,y

  // For now, only allow INSTANCE mode.
  if (mode != softmax_mode::INSTANCE)
    LBANN_ERROR("Unsupported softmax mode.");

  // Other basic input validation
  LBANN_ASSERT(x.Height() == y.Height());
  LBANN_ASSERT(x.Width() == y.Width());
  LBANN_ASSERT(x.LDim() == y.LDim());

  // Note (trb 01/26/2021): This function gets much more complicated
  // if the data has different strides. It's not prohibitive, but it
  // adds significant complications -- extraneous memory descriptors
  // and extraneous copies. I don't want to worry about it until
  // we know this is a problem in the real world.

  // Get the DNNL engine for this device
  dnnl::engine& engine = onednn::get_device_engine<D>();
  // Get the DNNL stream object correspondning to this SyncInfo object
  dnnl::stream stream = onednn::get_stream(engine, si);

  xDesc.get().set_data_handle(const_cast<DataT*>(x.LockedBuffer()), stream);
  yDesc.get().set_data_handle(y.Buffer(), stream);

  // Create operation descriptor and primitive descriptor.
  //
  // (trb 01/26/2021): I wonder if this is something we should
  // consider storing at the layer level and stubbing into
  // cuDNN/MIOpen operations (make "softmax_forward_descriptor_type =
  // int" for those cases, or whatever). Another option might be some
  // memoization that keeps track of these things. It just seems like
  // standing these up and tearing them down over and over will be a
  // significant overhead in long training runs with many batches.
  //
  // In this case, I expect "typical" cases to have two distinct
  // cases: the "usual" one, and the "last batch" one. And there are
  // not likely that many softmax layers per model (probably just one,
  // but I don't want to assert that is true in all cases). So every
  // batch, we're mostly spinning up something equivalent to one of
  // these two cases.

  auto softmax_desc =
    dnnl::softmax_forward::desc(dnnl::prop_kind::forward_training,
                                xDesc.get().get_desc(),
                                softmax_axis);
  auto softmax_prim_desc =
    dnnl::softmax_forward::primitive_desc(softmax_desc, engine);

  // Create the actual primitive.
  auto softmax_prim = dnnl::softmax_forward(softmax_prim_desc);

  // Primitive execution.
  softmax_prim.execute(stream,
                       { {DNNL_ARG_SRC, xDesc.get()},
                         {DNNL_ARG_DST, yDesc.get()} });

  // FIXME (trb 01/26/2021): Need to figure out how to generalize
  // this. I almost wonder if we'll want this "stream" to be absorbed
  // into `El::SyncInfo` somehow for GPU.
  stream.wait();
}

template <El::Device D>
template <typename DataT, typename ScalarT>
void onednn_backend<D>::softmax_backward(
  ScalarT const& alpha_in,
  TensorDescriptor const& yDesc,
  El::Matrix<DataT, D> const& y,
  TensorDescriptor const& dyDesc,
  El::Matrix<DataT, D> const& dy,
  ScalarT const& beta_in,
  TensorDescriptor const& dxDesc,
  El::Matrix<DataT, D>& dx,
  El::SyncInfo<D> const& si,
  softmax_mode mode,
  softmax_alg alg)
{
  if (alg == softmax_alg::LOG)
    return logsoftmax_backward(alpha_in,
                               yDesc,
                               y,
                               dyDesc,
                               dy,
                               beta_in,
                               dxDesc,
                               dx,
                               si,
                               mode);

  constexpr auto softmax_axis = 1;

  LBANN_ASSERT(dy.Height() == dx.Height());
  LBANN_ASSERT(dy.Width() == dx.Width());
  LBANN_ASSERT(dy.LDim() == dx.LDim()); // See note in softmax_forward

  dnnl::engine& engine = onednn::get_device_engine<D>();
  dnnl::stream stream = onednn::get_stream(engine, si);

  yDesc.get().set_data_handle(const_cast<DataT*>(y.LockedBuffer()), stream);
  dyDesc.get().set_data_handle(const_cast<DataT*>(dy.LockedBuffer()), stream);
  dxDesc.get().set_data_handle(dx.Buffer(), stream);

  // FIXME: HACK -- should cache from the fwd pass.
  auto fwd_softmax_prim_desc =
    dnnl::softmax_forward::primitive_desc(
      { dnnl::prop_kind::forward_training,
        yDesc.get().get_desc(),
        softmax_axis },
      engine);

  auto softmax_prim_desc =
    dnnl::softmax_backward::primitive_desc(
      { dxDesc.get().get_desc(),
        yDesc.get().get_desc(),
        softmax_axis },
      engine,
      fwd_softmax_prim_desc);

  auto softmax_prim = dnnl::softmax_backward(softmax_prim_desc);

  softmax_prim.execute(stream,
                       { {DNNL_ARG_DST, yDesc.get()},
                         {DNNL_ARG_DIFF_DST, dyDesc.get()},
                         {DNNL_ARG_DIFF_SRC, dxDesc.get()} });
  stream.wait();
}
#endif // defined LBANN_HAS_ONEDNN
} // namespace lbann
#endif // LBANN_UTILS_DNN_LIB_ONEDNN_SOFTMAX_HPP_
