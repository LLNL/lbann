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
#ifndef LBANN_UTILS_DNN_LIB_SOFTMAX_HPP
#define LBANN_UTILS_DNN_LIB_SOFTMAX_HPP

#include "lbann_config.hpp"

#if defined LBANN_HAS_CUDNN
#include "lbann/utils/dnn_lib/cudnn/softmax.hpp"
#elif defined LBANN_HAS_MIOPEN
#include "lbann/utils/dnn_lib/miopen/softmax.hpp"
#elif defined LBANN_HAS_GPU && !defined LBANN_HAS_ONEDNN_GPU
static_assert(false,
              "GPU support detected but no valid DNN library implementation. ");
#endif // LBANN_HAS_CUDNN

#if defined LBANN_HAS_ONEDNN
#include "lbann/utils/dnn_lib/onednn/softmax.hpp"
#endif // LBANN_HAS_ONEDNN

#include "lbann/utils/dnn_lib/openmp/softmax.hpp"

#include "lbann/utils/sync_info_helpers.hpp"

namespace lbann {
namespace dnn_lib {

template <typename ScalarT, typename TensorDescT, typename DataT, El::Device D>
void softmax_forward(ScalarT const& alpha_in,
                     TensorDescT const& xDesc,
                     El::Matrix<DataT, D> const& x,
                     ScalarT const& beta_in,
                     TensorDescT const& yDesc,
                     El::Matrix<DataT, D>& y,
                     El::SyncInfo<D> const& si,
                     softmax_mode mode,
                     softmax_alg alg = softmax_alg::ACCURATE)
{
  using backend = typename TensorDescT::backend_type;
  static_assert(backend::device == D, "Mismatched device identifiers.");
  backend::softmax_forward(alpha_in,
                           xDesc,
                           x,
                           beta_in,
                           yDesc,
                           y,
                           si,
                           mode,
                           alg);
}

template <typename ScalarT, typename TensorDescT, typename DataT, El::Device D>
void softmax_forward(ScalarT const& alpha_in,
                     TensorDescT const& xDesc,
                     El::Matrix<DataT, D> const& x,
                     ScalarT const& beta_in,
                     TensorDescT const& yDesc,
                     El::Matrix<DataT, D>& y,
                     softmax_mode mode,
                     softmax_alg alg = softmax_alg::ACCURATE)
{
  auto multisync = El::MakeMultiSync(get_sync_info(y), get_sync_info(x));
  softmax_forward(alpha_in,
                  xDesc,
                  x,
                  beta_in,
                  yDesc,
                  y,
                  force(multisync),
                  mode,
                  alg);
}

template <typename ScalarT, typename TensorDescT, typename DataT, El::Device D>
void softmax_backward(ScalarT const& alpha_in,
                      TensorDescT const& yDesc,
                      El::Matrix<DataT, D> const& y,
                      TensorDescT const& dyDesc,
                      El::Matrix<DataT, D> const& dy,
                      ScalarT const& beta_in,
                      TensorDescT const& dxDesc,
                      El::Matrix<DataT, D>& dx,
                      El::SyncInfo<D> const& si,
                      softmax_mode mode,
                      softmax_alg alg = softmax_alg::ACCURATE)
{
  // Short-circuit if we can
  if (y.IsEmpty())
    return;

  using backend = typename TensorDescT::backend_type;
  static_assert(backend::device == D, "Mismatched device identifiers.");
  backend::softmax_backward(alpha_in,
                            yDesc,
                            y,
                            dyDesc,
                            dy,
                            beta_in,
                            dxDesc,
                            dx,
                            si,
                            mode,
                            alg);
}

template <typename ScalarT, typename TensorDescT, typename DataT, El::Device D>
void softmax_backward(ScalarT const& alpha_in,
                      TensorDescT const& yDesc,
                      El::Matrix<DataT, D> const& y,
                      TensorDescT const& dyDesc,
                      El::Matrix<DataT, D> const& dy,
                      ScalarT const& beta_in,
                      TensorDescT const& dxDesc,
                      El::Matrix<DataT, D>& dx,
                      softmax_mode mode,
                      softmax_alg alg = softmax_alg::ACCURATE)
{
  auto multisync =
    El::MakeMultiSync(get_sync_info(dx), get_sync_info(y), get_sync_info(dy));
  softmax_backward(alpha_in,
                   yDesc,
                   y,
                   dyDesc,
                   dy,
                   beta_in,
                   dxDesc,
                   dx,
                   force(multisync),
                   mode,
                   alg);
}
} // namespace dnn_lib
} // namespace lbann
#endif // LBANN_UTILS_DNN_LIB_SOFTMAX_HPP
