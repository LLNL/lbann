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

#ifndef LBANN_LAYERS_MATH_DFT_ABS_HPP_INCLUDED
#define LBANN_LAYERS_MATH_DFT_ABS_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"
#include "lbann_config.hpp"

// This layer is only supported if LBANN has FFTW support.
#ifdef LBANN_HAS_FFTW

namespace lbann {

// Forward declaration of FFT stuff
template <typename T, El::Device D>
class dft_abs_impl;
class lbann_comm;

/** @class dft_abs_layer
 *  @brief Absolute value of discrete Fourier transform.
 *
 *  One-, two-, or three-dimensional data is allowed.
 *
 *  The implementation is meant to be as flexible as possible. We use
 *  FFTW for the CPU implementation; whichever types your
 *  implementation of FFTW supports will be supported in this layer at
 *  runtime. The GPU implementation uses cuFFT on NVIDIA GPUs and will
 *  support float and double at runtime (assuming CUDA support is
 *  enabled). A future implementation will support rocFFT for AMD
 *  GPUs.
 *
 *  Currently, LBANN only supports outputting the same type that is
 *  used as input. As such, in forward propagation, this will do a
 *  DFT and then compute the absolute value of the output
 *  implicitly. The intention is to support immediate customer need
 *  now; we will generalize this as LBANN learns to support different
 *  input/output data types.
 *
 *  @note The "Layout" template parameter is omitted because
 *        "MODEL_PARALLEL" is not supported at this time.
 */
template <typename TensorDataType, El::Device Device>
class dft_abs_layer : public data_type_layer<TensorDataType>
{
  static const auto Layout = data_layout::DATA_PARALLEL;

public:
  dft_abs_layer(lbann_comm* const comm);
  ~dft_abs_layer();
  dft_abs_layer* copy() const override { return new dft_abs_layer(*this); }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "DFT Abs"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override { return ERROR_SIGNALS; }

  description get_description() const override
  {
    return data_type_layer<TensorDataType>::get_description();
  }

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  dft_abs_layer() : dft_abs_layer(nullptr) {}

  dft_abs_layer(dft_abs_layer const&);
  void setup_dims() override;
  void fp_compute() override;
  void bp_compute() override;

private:
  using impl_type = dft_abs_impl<TensorDataType, Device>;
  std::unique_ptr<impl_type> pimpl_;
}; // class dft_abs_layer

template <typename T, El::Device D>
void dft_abs_layer<T, D>::write_specific_proto(lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  proto.mutable_dft_abs();
}

#ifndef LBANN_DFT_ABS_LAYER_INSTANTIATE

#ifdef LBANN_HAS_FFTW_FLOAT
extern template class dft_abs_layer<float, El::Device::CPU>;
#endif // LBANN_HAS_FFTW_FLOAT
#if defined(LBANN_HAS_DOUBLE) && defined(LBANN_HAS_FFTW_DOUBLE)
extern template class dft_abs_layer<double, El::Device::CPU>;
#endif // defined(LBANN_HAS_DOUBLE) && defined (LBANN_HAS_FFTW_DOUBLE)

#ifdef LBANN_HAS_GPU
// cuFFT always supports both types.
extern template class dft_abs_layer<float, El::Device::GPU>;
#ifdef LBANN_HAS_DOUBLE
extern template class dft_abs_layer<double, El::Device::GPU>;
#endif // LBANN_HAS_DOUBLE
#endif // LBANN_HAS_GPU

#endif // LBANN_DFT_ABS_LAYER_INSTANTIATE

} // namespace lbann
#endif // LBANN_HAS_FFTW
#endif // LBANN_LAYERS_MATH_DFT_ABS_HPP_INCLUDED
