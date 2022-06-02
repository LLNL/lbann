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

#include "lbann/layers/data_type_layer.hpp"
#define LBANN_PERMUTE_LAYER_INSTANTIATE
#include "lbann/layers/transform/permute.hpp"

#include "permute/cutensor_permuteimpl.hpp"

#include "lbann/utils/description.hpp"
#include <memory>

namespace lbann {

template <typename T>
class PermuteLayer<T>::PermuteImpl
{
public:
  using DeviceImplType = cuTENSOR_PermuteImpl;
  using MatType = El::Matrix<T, El::Device::GPU>;
  using DimsType = typename DeviceImplType::DimsType;

public:
  // LBANN uses row-major tensor ordering.
  PermuteImpl(std::vector<int> const& perm_row_major)
    : m_device_impl{RowMajorPerm{perm_row_major}}
  {}
  PermuteImpl(PermuteImpl const& other) = default;

  // Returns the row-major output dims.
  std::vector<int> setup_dims(std::vector<int> const& input_dims)
  {
    using IndexType = typename DimsType::value_type;
    m_device_impl.set_dims(RowMajor(vec_convert<IndexType>(input_dims)));
    return vec_convert<int>(RowMajor(m_device_impl.output_dims()).get());
  }

  void forward_prop(MatType const& input, MatType& output) const
  {
    if (input.Width() == El::Int{0} || output.Width() == El::Int{0})
      return;
    m_device_impl.permute(input, output);
  }

  // Activations don't actually matter here...
  void backward_prop(MatType const& grad_wrt_out, MatType& grad_wrt_in)
  {
    if (grad_wrt_out.Width() == El::Int{0} || grad_wrt_in.Width() == El::Int{0})
      return;
    m_device_impl.inverse_permute(grad_wrt_out, grad_wrt_in);
  }

  std::vector<int> get_perm() const
  {
    return RowMajorPerm{m_device_impl.perm()}.get();
  }

  std::string describe_perm() const
  {
    RowMajorPerm const perm_rm(m_device_impl.perm());
    std::ostringstream oss;
    oss << "(";
    for (size_t ii = 0; ii < perm_rm.size(); ++ii)
      oss << (ii == 0UL ? " " : ", ") << perm_rm.get()[ii];
    oss << " )";
    return oss.str();
  }

private:
  DeviceImplType m_device_impl;

}; // class PermuteImpl

template <typename T>
PermuteLayer<T>::PermuteLayer(std::vector<int> const& axes_rm)
  : data_type_layer<T>(nullptr), m_impl{std::make_unique<PermuteImpl>(axes_rm)}
{
  this->m_expected_num_parent_layers = 1;
}

template <typename T>
PermuteLayer<T>::PermuteLayer(PermuteLayer const& other)
  : data_type_layer<T>{other},
    m_impl{std::make_unique<PermuteImpl>(*other.m_impl)}
{
  this->m_expected_num_parent_layers = 1;
}

template <typename T>
auto PermuteLayer<T>::operator=(PermuteLayer const& other) -> PermuteLayer&
{
  if (std::addressof(other) != this) {
    PermuteLayer{*this}.swap(*this);
  }
  return *this;
}

template <typename T>
PermuteLayer<T>::~PermuteLayer()
{}

template <typename T>
void PermuteLayer<T>::swap(PermuteLayer& other)
{
  std::swap(m_impl, other.m_impl);
}

template <typename T>
description PermuteLayer<T>::get_description() const
{
  auto desc = data_type_layer<T>::get_description();
  desc.add("perm", m_impl->describe_perm());
  return desc;
}

template <typename T>
void PermuteLayer<T>::setup_dims(DataReaderMetaData& dr_metadata)
{
  data_type_layer<T>::setup_dims(dr_metadata);
  this->set_output_dims(m_impl->setup_dims(this->get_input_dims(0)), 0);
}

template <typename T>
void PermuteLayer<T>::fp_compute()
{
  using MatType = El::Matrix<T, El::Device::GPU>;

  auto const& input = this->get_local_prev_activations();
  auto& output = this->get_local_activations();

  LBANN_ASSERT(input.GetDevice() == El::Device::GPU);
  LBANN_ASSERT(output.GetDevice() == El::Device::GPU);

  if (input.Width()) {
    m_impl->forward_prop(static_cast<MatType const&>(input),
                         static_cast<MatType&>(output));
  }
}

template <typename T>
void PermuteLayer<T>::bp_compute()
{
  using MatType = El::Matrix<T, El::Device::GPU>;

  auto const& grad_wrt_output = this->get_local_prev_error_signals();
  auto& grad_wrt_input = this->get_local_error_signals();

  LBANN_ASSERT(grad_wrt_output.GetDevice() == El::Device::GPU);
  LBANN_ASSERT(grad_wrt_input.GetDevice() == El::Device::GPU);

  if (grad_wrt_output.Width()) {
    m_impl->backward_prop(static_cast<MatType const&>(grad_wrt_output),
                          static_cast<MatType&>(grad_wrt_input));
  }
}

#define PROTO(T) template class PermuteLayer<T>
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
