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

#ifndef LBANN_LAYERS_DATA_TYPE_DISTCONV_ADAPTER_HPP_INCLUDED
#define LBANN_LAYERS_DATA_TYPE_DISTCONV_ADAPTER_HPP_INCLUDED

#include "distconv/tensor/shuffle_mpi.hpp"
#include "distconv/tensor/shuffle_mpi_cuda.hpp"
#include "lbann/layers/distconv_adapter.hpp"

namespace lbann {

// Forward Declarations
class Layer;

namespace dc {
using Shape = ::distconv::tensor::Shape;

using LocaleMPI = ::distconv::tensor::LocaleMPI;

template <typename TensorDataType>
using TensorDev = ::distconv::tensor::
  Tensor<TensorDataType, LocaleMPI, ::distconv::tensor::CUDAAllocator>;

template <typename TensorDataType>
using TensorShuffler =
  ::distconv::tensor::TensorMPICUDAShuffler<TensorDataType>;

} // namespace dc

template <typename InputTensorDataType,
          typename OutputTensorDataType = InputTensorDataType>
class data_type_distconv_adapter : public distconv_adapter
{
public:
  // Keep the older TensorDevType around for downstream layers
  using TensorDevType = dc::TensorDev<OutputTensorDataType>;
  using InputTensorDevType = dc::TensorDev<InputTensorDataType>;
  using OutputTensorDevType = dc::TensorDev<OutputTensorDataType>;
  using InputTensorShufflerType = dc::TensorShuffler<InputTensorDataType>;
  using OutputTensorShufflerType = dc::TensorShuffler<OutputTensorDataType>;

  data_type_distconv_adapter(Layer& layer) : distconv_adapter(layer) {}
  virtual ~data_type_distconv_adapter() = default;

  /** Get activation tensor corresponding to child layer. */
  const OutputTensorDevType& get_activations(const Layer& child) const override;
  /** Get error signal tensor corresponding to parent layer. */
  const InputTensorDevType&
  get_error_signals(const Layer& parent) const override;

  /** Get activation tensor. */
  const OutputTensorDevType& get_activations(int child_index = 0) const;
  /** Get activation tensor. */
  OutputTensorDevType& get_activations(int child_index = 0);
  /** Get original activation tensor. */
  const OutputTensorDevType&
  get_original_activations(int child_index = 0) const;
  /** Get original activation tensor. */
  OutputTensorDevType& get_original_activations(int child_index = 0);

  /** Get previous activation tensor. */
  const InputTensorDevType& get_prev_activations(int parent_index = 0) const;
  /** Get previous activation tensor. */
  InputTensorDevType& get_prev_activations(int parent_index = 0);
  /** Get original previous activation tensor. */
  const InputTensorDevType&
  get_original_prev_activations(int parent_index = 0) const;
  /** Get original previous activation tensor. */
  InputTensorDevType& get_original_prev_activations(int parent_index = 0);

  /** Get error signal tensor. */
  const InputTensorDevType& get_error_signals(int parent_index = 0) const;
  /** Get error signal tensor. */
  InputTensorDevType& get_error_signals(int parent_index = 0);
  /** Get original error signal tensor. */
  const InputTensorDevType&
  get_original_error_signals(int parent_index = 0) const;
  /** Get original error signal tensor. */
  InputTensorDevType& get_original_error_signals(int parent_index = 0);

  /** Get previous error siganl tensor. */
  const OutputTensorDevType& get_prev_error_signals(int child_index = 0) const;
  /** Get previous error siganl tensor. */
  OutputTensorDevType& get_prev_error_signals(int child_index = 0);
  /** Get original previous error signal tensor. */
  const OutputTensorDevType&
  get_original_prev_error_signals(int child_index = 0) const;
  /** Get original previous error signal tensor. */
  OutputTensorDevType& get_original_prev_error_signals(int child_index = 0);

  void fp_setup() override;
  void fp_postprocess() override;
  void bp_setup() override;
  void bp_postprocess() override;

  void dump_activations() const override;
  void dump_original_activations() override;
  void dump_error_signals() const override;
  void dump_original_error_signals() override;

protected:
  // Setup fp tensors
  void setup_prev_activations() override;
  virtual std::unique_ptr<InputTensorDevType>
  setup_prev_activations_i(int index) const;
  void setup_original_prev_activations() override;
  virtual std::unique_ptr<InputTensorDevType>
  setup_original_prev_activations_i(int index) const;
  void setup_activations() override;
  virtual std::unique_ptr<OutputTensorDevType>
  setup_activations_i(int index) const;
  void setup_original_activations() override;
  virtual std::unique_ptr<OutputTensorDevType>
  setup_original_activations_i(int index) const;

  // Setup bp tensors
  void setup_prev_error_signals() override;
  virtual std::unique_ptr<OutputTensorDevType>
  setup_prev_error_signals_i(int index) const;
  void setup_original_prev_error_signals() override;
  virtual std::unique_ptr<OutputTensorDevType>
  setup_original_prev_error_signals_i(int index) const;
  void setup_error_signals() override;
  virtual std::unique_ptr<InputTensorDevType>
  setup_error_signals_i(int index) const;
  void setup_original_error_signals() override;
  virtual std::unique_ptr<InputTensorDevType>
  setup_original_error_signals_i(int index) const;

  virtual dc::Shape get_prev_activations_shape(int input_index = 0) const;
  virtual dc::Shape get_prev_activations_local_shape(int input_index = 0) const;
  virtual dc::Shape get_activations_shape(int index = 0) const;
  virtual dc::Shape get_activations_local_shape(int index = 0) const;

  virtual dc::Shape get_prev_error_signals_shape(int index = 0) const;
  virtual dc::Shape get_prev_error_signals_local_shape(int index = 0) const;
  virtual dc::Shape get_error_signals_shape(int index = 0) const;
  virtual dc::Shape get_error_signals_local_shape(int index = 0) const;

  void ensure_prev_activations() override;
  void copy_out_activations() override;
  void ensure_prev_error_signals() override;
  void copy_out_error_signals() override;

  InputTensorShufflerType&
  get_prev_activations_shuffler(const InputTensorDevType& src,
                                const InputTensorDevType& dst);
  OutputTensorShufflerType&
  get_activations_shuffler(const OutputTensorDevType& src,
                           const OutputTensorDevType& dst);
  OutputTensorShufflerType&
  get_prev_error_signals_shuffler(const OutputTensorDevType& src,
                                  const OutputTensorDevType& dst);
  InputTensorShufflerType&
  get_error_signals_shuffler(const InputTensorDevType& src,
                             const InputTensorDevType& dst);

private:
  std::vector<std::unique_ptr<InputTensorDevType>> m_inputs;
  std::vector<std::unique_ptr<InputTensorDevType>> m_original_inputs;
  std::vector<std::unique_ptr<OutputTensorDevType>> m_outputs;
  std::vector<std::unique_ptr<OutputTensorDevType>> m_original_outputs;

  std::vector<std::unique_ptr<InputTensorDevType>> m_gradient_wrt_inputs;
  std::vector<std::unique_ptr<InputTensorDevType>>
    m_original_gradient_wrt_inputs;
  std::vector<std::unique_ptr<OutputTensorDevType>> m_gradient_wrt_outputs;
  std::vector<std::unique_ptr<OutputTensorDevType>>
    m_original_gradient_wrt_outputs;

  // TODO: Use unique_ptr
  std::array<InputTensorShufflerType*, 4> m_prev_activations_shufflers{
    {nullptr, nullptr, nullptr, nullptr}};
  std::array<OutputTensorShufflerType*, 4> m_activations_shufflers{
    {nullptr, nullptr, nullptr, nullptr}};
  std::array<OutputTensorShufflerType*, 4> m_prev_error_signals_shufflers{
    {nullptr, nullptr, nullptr, nullptr}};
  std::array<InputTensorShufflerType*, 4> m_error_signals_shufflers{
    {nullptr, nullptr, nullptr, nullptr}};

  void set_activations_outermost_dimension(size_t dim);
  void set_error_signals_outermost_dimension(size_t dim);

  uint64_t get_max_mini_batch_size() const;
};

} // namespace lbann

#endif // LBANN_LAYERS_DATA_TYPE_DISTCONV_ADAPTER_HPP_INCLUDED
