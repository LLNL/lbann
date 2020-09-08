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

#include "lbann/layers/io/input/generic_input_layer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/distconv.hpp"
#include "lbann/models/model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, typename T_io_buffer,
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

  bool m_shuffle_required;
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
          typename T_io_buffer,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class input_layer : public generic_input_layer<TensorDataType> {
  static_assert(T_layout == data_layout::DATA_PARALLEL,
                "input layer only supports DATA_PARALLEL data layout");
 public:
  /** @name Public Types */
  ///@{

  /** @brief The local tensor type expected for IO in this object. */
  using IODataType = DataType;

  ///@}
 public:

  /// @todo make the map and vector references
  input_layer(lbann_comm *comm, int num_parallel_readers,
    data_reader_target_mode target_mode = data_reader_target_mode::CLASSIFICATION)
    : generic_input_layer<TensorDataType>(comm, num_parallel_readers, target_mode) {
    // Initialize two buffers
    initialize_io_buffer(comm, std::min(num_parallel_readers, data_type_layer<TensorDataType>::m_comm->get_procs_per_trainer()));
    initialize_io_buffer(comm, std::min(num_parallel_readers, data_type_layer<TensorDataType>::m_comm->get_procs_per_trainer()));
    for (auto io_buffer : this->m_io_buffers) {
      io_buffer->fetch_data_fn = new fetch_data_functor<IODataType>(target_mode);
      io_buffer->update_data_reader_fn = new update_data_reader_functor();
    }
  }
  input_layer(const input_layer&) = default;
  input_layer& operator=(const input_layer&) = default;
  input_layer* copy() const override {
    return new input_layer(*this);
  }

  inline void initialize_io_buffer(lbann_comm *comm, int num_parallel_readers) {
    generic_input_layer<TensorDataType>::template initialize_io_buffer<T_io_buffer>(comm, num_parallel_readers);
  }

  std::string get_type() const override { return "input"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

#ifdef LBANN_HAS_DISTCONV
  void fp_compute () override;
  using distconv_adapter_type = input_distconv_adapter<TensorDataType, T_io_buffer, T_layout, Dev>;
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
#endif // LBANN_HAS_DISTCONV
};

#ifndef LBANN_INPUT_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device)         \
  extern template class input_layer<    \
    T, partitioned_io_buffer<T>,        \
    data_layout::DATA_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_INPUT_LAYER_INSTANTIATE

} // namespace lbann

#endif  // LBANN_LAYERS_INPUT_LAYER_HPP_INCLUDED
