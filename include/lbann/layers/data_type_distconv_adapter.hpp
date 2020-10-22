////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#include "lbann/layers/distconv_adapter.hpp"
#include "lbann/layers/layer.hpp"

namespace lbann {

template <typename TensorDataType>
class data_type_distconv_adapter: public distconv_adapter {
public:
  using TensorDevType = dc::TensorDev<TensorDataType>;
  using TensorShufflerType = dc::TensorShuffler<TensorDataType>;

  data_type_distconv_adapter(Layer& layer): distconv_adapter(layer) {}
  virtual ~data_type_distconv_adapter() = default;

  /** Get activation tensor corresponding to child layer. */
  const TensorDevType& get_activations(const Layer& child) const override;
  /** Get error signal tensor corresponding to parent layer. */
  const TensorDevType& get_error_signals(const Layer& parent) const override;

  /** Get activation tensor. */
  const TensorDevType& get_activations(int child_index = 0) const;
  /** Get activation tensor. */
  TensorDevType& get_activations(int child_index = 0);
  /** Get original activation tensor. */
  const TensorDevType& get_original_activations(int child_index = 0) const;
  /** Get original activation tensor. */
  TensorDevType& get_original_activations(int child_index = 0);

  /** Get previous activation tensor. */
  const TensorDevType& get_prev_activations(int parent_index = 0) const;
  /** Get previous activation tensor. */
  TensorDevType& get_prev_activations(int parent_index = 0);
  /** Get original previous activation tensor. */
  const TensorDevType& get_original_prev_activations(int parent_index = 0) const;
  /** Get original previous activation tensor. */
  TensorDevType& get_original_prev_activations(int parent_index = 0);

  /** Get error signal tensor. */
  const TensorDevType& get_error_signals(int parent_index = 0) const;
  /** Get error signal tensor. */
  TensorDevType& get_error_signals(int parent_index = 0);
  /** Get original error signal tensor. */
  const TensorDevType& get_original_error_signals(int parent_index = 0) const;
  /** Get original error signal tensor. */
  TensorDevType& get_original_error_signals(int parent_index = 0);

  /** Get previous error siganl tensor. */
  const TensorDevType& get_prev_error_signals(int child_index = 0) const;
  /** Get previous error siganl tensor. */
  TensorDevType& get_prev_error_signals(int child_index = 0);
  /** Get original previous error signal tensor. */
  const TensorDevType& get_original_prev_error_signals(int child_index = 0) const;
  /** Get original previous error signal tensor. */
  TensorDevType& get_original_prev_error_signals(int child_index = 0);

  void fp_setup(El::Int mini_batch_size) override;
  void fp_postprocess() override;
  void bp_setup(El::Int mini_batch_size) override;
  void bp_postprocess() override;

  void dump_activations() const override;
  void dump_original_activations() override;
  void dump_error_signals() const override;
  void dump_original_error_signals() override;

 protected:
  // Setup fp tensors
  void setup_prev_activations() override;
  virtual std::unique_ptr<TensorDevType> setup_prev_activations_i(int index) const;
  void setup_original_prev_activations() override;
  virtual std::unique_ptr<TensorDevType> setup_original_prev_activations_i(int index) const;
  void setup_activations() override;
  virtual std::unique_ptr<TensorDevType> setup_activations_i(int index) const;
  void setup_original_activations() override;
  virtual std::unique_ptr<TensorDevType> setup_original_activations_i(int index) const;

  // Setup bp tensors
  void setup_prev_error_signals() override;
  virtual std::unique_ptr<TensorDevType> setup_prev_error_signals_i(int index) const;
  void setup_original_prev_error_signals() override;
  virtual std::unique_ptr<TensorDevType> setup_original_prev_error_signals_i(int index) const;
  void setup_error_signals() override;
  virtual std::unique_ptr<TensorDevType> setup_error_signals_i(int index) const;
  void setup_original_error_signals() override;
  virtual std::unique_ptr<TensorDevType> setup_original_error_signals_i(int index) const;

  virtual dc::Shape get_prev_activations_shape(int input_index=0) const;
  virtual dc::Shape get_prev_activations_local_shape(int input_index=0) const;
  virtual dc::Shape get_activations_shape(int index=0) const;
  virtual dc::Shape get_activations_local_shape(int index=0) const;

  virtual dc::Shape get_prev_error_signals_shape(int index=0) const;
  virtual dc::Shape get_prev_error_signals_local_shape(int index=0) const;
  virtual dc::Shape get_error_signals_shape(int index=0) const;
  virtual dc::Shape get_error_signals_local_shape(int index=0) const;

  void ensure_prev_activations() override;
  void copy_out_activations() override;
  void ensure_prev_error_signals() override;
  void copy_out_error_signals() override;

  TensorShufflerType& get_prev_activations_shuffler(
      const TensorDevType &src, const TensorDevType &dst);
  TensorShufflerType& get_activations_shuffler(
      const TensorDevType &src, const TensorDevType &dst);
  TensorShufflerType& get_prev_error_signals_shuffler(
      const TensorDevType &src, const TensorDevType &dst);
  TensorShufflerType& get_error_signals_shuffler(
      const TensorDevType &src, const TensorDevType &dst);

 private:
  std::vector<std::unique_ptr<TensorDevType>> m_inputs;
  std::vector<std::unique_ptr<TensorDevType>> m_original_inputs;
  std::vector<std::unique_ptr<TensorDevType>> m_outputs;
  std::vector<std::unique_ptr<TensorDevType>> m_original_outputs;

  std::vector<std::unique_ptr<TensorDevType>> m_gradient_wrt_inputs;
  std::vector<std::unique_ptr<TensorDevType>> m_original_gradient_wrt_inputs;
  std::vector<std::unique_ptr<TensorDevType>> m_gradient_wrt_outputs;
  std::vector<std::unique_ptr<TensorDevType>> m_original_gradient_wrt_outputs;

  // TODO: Use unique_ptr
  std::array<TensorShufflerType*, 4> m_prev_activations_shufflers{ {nullptr, nullptr, nullptr, nullptr} };
  std::array<TensorShufflerType*, 4> m_activations_shufflers{ {nullptr, nullptr, nullptr, nullptr} };
  std::array<TensorShufflerType*, 4> m_prev_error_signals_shufflers{ {nullptr, nullptr, nullptr, nullptr} };
  std::array<TensorShufflerType*, 4> m_error_signals_shufflers{ {nullptr, nullptr, nullptr, nullptr} };

  void set_activations_outermost_dimension(size_t dim);
  void set_error_signals_outermost_dimension(size_t dim);

  size_t get_max_mini_batch_size() const;
};

} // namespace lbann

#endif // LBANN_LAYERS_DATA_TYPE_DISTCONV_ADAPTER_HPP_INCLUDED
