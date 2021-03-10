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

#ifndef LBANN_LAYERS_INPUT_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_INPUT_LAYER_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/data_coordinator/buffered_data_coordinator.hpp"
#include "lbann/execution_contexts/sgd_execution_context.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/distconv.hpp"
#include "lbann/models/model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType,
          data_layout T_layout, El::Device Dev>
class input_distconv_adapter: public data_type_distconv_adapter<TensorDataType> {
 public:
  using TensorDevType = typename data_type_distconv_adapter<TensorDataType>::TensorDevType;
  using TensorHost = dc::TensorHost<TensorDataType>;
  using TensorHostShuffler = dc::TensorHostShuffler<TensorDataType>;

  input_distconv_adapter(Layer& layer, const bool shuffle_required);
  virtual ~input_distconv_adapter() = default;

  TensorHostShuffler &get_shuffler(const TensorHost &src, const TensorHost &dst,
                                   int mat_idx);
  void setup_fp_tensors() override;
  std::unique_ptr<TensorDevType> setup_activations_i(int index) const override;
  dc::Shape get_activations_local_shape(int index) const override;
  dc::Shape get_activations_shape(int index) const;
  void setup_shuffler_buffers(const TensorHost &src, const TensorHost &dst);

  // No bp tensors needed for this layer.
  void setup_prev_error_signals() override {}
  void setup_original_prev_error_signals() override {}
  void setup_error_signals() override {}
  void setup_original_error_signals() override {}
  void setup_bp_tensors() override {}

  bool child_copy_required(size_t output_index) const override;
  bool child_shuffle_required(size_t output_index) const override;

  // Nothing to do here as everything is done in fp_compute_distconv.
  void fp_setup(El::Int mini_batch_size) override {}
  void fp_compute();
  bool is_input_processed(size_t index) const;

 private:
  std::vector<bool> m_is_input_processed;
  std::vector<std::unique_ptr<TensorHost>> m_original_host_tensors;
  std::vector<std::unique_ptr<TensorHost>> m_host_tensors;

  const bool m_shuffle_required;
  std::vector<std::array<std::unique_ptr<TensorHostShuffler>, 4>> m_shufflers;
  std::unique_ptr<TensorDataType> m_shuffler_src_buf;
  size_t m_shuffler_src_buf_size = 0;
  std::unique_ptr<TensorDataType> m_shuffler_dst_buf;
  size_t m_shuffler_dst_buf_size = 0;

  // TODO: Use pinned memory pool
  TensorDataType *m_copy_pinned_buffer = nullptr;
};
#endif // LBANN_HAS_DISTCONV

/** @brief Interface with data reader. */
template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class input_layer : public data_type_layer<TensorDataType> {
  static_assert(T_layout == data_layout::DATA_PARALLEL,
                "input layer only supports DATA_PARALLEL data layout");
 public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  ///@}
 public:

  /// @todo make the map and vector references
  input_layer(lbann_comm *comm,
              data_reader_target_mode dr_mode = data_reader_target_mode::NA)
    : data_type_layer<TensorDataType>(comm),
    m_data_reader_mode(dr_mode) {

    // Input layers have no parents
    this->m_expected_num_parent_layers = 0;
    if(dr_mode == data_reader_target_mode::NA) {
      this->m_expected_num_child_layers = 1;
    }else {
      // Input layers output a sample and target, which could be the
      // original value, categorical label, or regression value
      this->m_expected_num_child_layers = 2;
    }
  }

  // This is to track if samples are loaded with set_samples(), if so the
  // fp_compute() sample loading is no longer necessary
  bool m_samples_loaded = false;

  input_layer(const input_layer&) = default;
  input_layer& operator=(const input_layer&) = default;
  input_layer* copy() const override {
    return new input_layer(*this);
  }

  std::string get_type() const override { return "input"; }
  // description get_description() const override {
  //   auto desc = io_layer<TensorDataType>::get_description();
  //   return desc;
  // }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }


  void setup_dims(DataReaderMetaData& dr_metadata) override;

  void setup_data(size_t max_mini_batch_size) override;

  /** Setup output tensors.
   *  Sets up the effective (global) mini-batch size.
   */
  void fp_setup_outputs(El::Int mini_batch_size) override;

  void fp_compute() override;

  void set_samples(const El::AbstractDistMatrix<TensorDataType>& samples);

  /**
   * Get the dimensions of the underlying data.
   */
  std::vector<int> get_data_dims(DataReaderMetaData& dr_metadata, int child_index = 0) const;

  bool is_for_regression() const {
    return (m_data_reader_mode == data_reader_target_mode::REGRESSION);
  }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}
 protected:
  data_reader_target_mode m_data_reader_mode;

 private:
  friend cereal::access;
  input_layer()
    : input_layer(nullptr, data_reader_target_mode::NA)
  {}

#ifdef LBANN_HAS_DISTCONV
 public:
  /** @brief Extensions for distributed convolutions */
///{@
  using distconv_adapter_type = input_distconv_adapter<TensorDataType, T_layout, Dev>;
  friend distconv_adapter_type;
 protected:
  bool is_distconv_supported() const override {
    return Dev == El::Device::CPU && T_layout == data_layout::DATA_PARALLEL;
  }
  void setup_distconv_adapter(const DataReaderMetaData& dr_metadata) override {
    this->get_distconv_adapter_ptr() = make_unique<distconv_adapter_type>(
        *this, dr_metadata.shuffle_required);
  }
  distconv_adapter_type& get_distconv_adapter() override;
  const distconv_adapter_type& get_distconv_adapter() const override;
  bool keep_original_outputs(int index) const override;
///@}
#endif // LBANN_HAS_DISTCONV
};

#ifndef LBANN_INPUT_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device)         \
  extern template class input_layer<    \
    T, data_layout::DATA_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_INPUT_LAYER_INSTANTIATE

} // namespace lbann

#endif  // LBANN_LAYERS_INPUT_LAYER_HPP_INCLUDED
