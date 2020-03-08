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

#ifndef LBANN_LAYERS_INPUT_LAYER_DISTCONV_HPP_INCLUDED
#define LBANN_LAYERS_INPUT_LAYER_DISTCONV_HPP_INCLUDED

#include "lbann/layers/io/input/input_layer.hpp"
#include "lbann/utils/distconv.hpp"
#include <cstdint>

namespace lbann {

/** @brief Interface with data reader. */
template <typename TensorDataType,
          typename T_io_buffer,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU,
          typename InputType = TensorDataType>
class input_layer_distconv : public input_layer<TensorDataType, T_io_buffer, T_layout, Dev> {
 public:

  /// @todo make the map and vector references
  input_layer_distconv(lbann_comm *comm, int num_parallel_readers,
                       std::map<execution_mode, generic_data_reader *> data_readers,
                       bool data_set_spans_models = true,
                       data_reader_target_mode target_mode = data_reader_target_mode::CLASSIFICATION)
      : input_layer<TensorDataType, T_io_buffer, T_layout, Dev>(
          comm, num_parallel_readers, data_readers,
          data_set_spans_models, target_mode) {
  }

  void fp_compute() override {
    input_layer<TensorDataType, T_io_buffer, T_layout, Dev>::fp_compute();
#ifdef LBANN_HAS_DISTCONV
    // When enabled, shuffle the input samples and copy them to a device tensor
    if (this->distconv_enabled()) {
      fp_compute_distconv();
    }
#endif
  }

#ifdef LBANN_HAS_DISTCONV
 public:
  using TensorDevType = typename data_type_layer<TensorDataType>::TensorDevType;

  int get_num_dims() const {
    return this->get_output_dims().size() + 1;
  }

  void setup_tensors_fwd(const std::array<dc::Dist, dc::num_dists> &dists) override {
    using namespace dc;
    input_layer<TensorDataType, T_io_buffer, T_layout, Dev>::setup_tensors_fwd(dists);
    if (!this->distconv_enabled()) return;

    // copies the label data as well when the second child layer is
    // also enabled for distconv
    if (this->get_num_children() == 2 &&
        this->get_child_layers()[1]->using_distconv()) {
      m_copy_labels_dc = true;
      dc::MPIRootPrintStreamInfo() << "Copy label/response data to Distconv as well";
    }

    const auto tensor_shape = this->get_output_tensor_shape();
    const auto sample_dist = dc::get_hydrogen_data_parallel_distribution(get_num_dims());
    auto local_shape = tensor_shape;
    // Set the sample dimension as 0 so that its actual value is
    // calculated by Distconv
    local_shape[dc::get_sample_dim()] = 0;
    const auto &dist = dists[1];
    auto dist_no_halo = dist;
    dist_no_halo.clear_overlap();

    // Use the same MPI communicator for both IO buffers. This seems
    // to work around MPI errors likely caused with the alltoallv for
    // shuffling.
    const LocaleMPI loc(dc::get_mpi_comm(), false);

    if (dc::is_cosmoflow_parallel_io_enabled()) {
      // Assumes the input buffer is already partitioned for
      // Distconv
      m_input_host_view = TensorHost(tensor_shape, loc, dist_no_halo);
      // Create a Distconv tensor at host memory.
      m_input_host_tensor = TensorHost(tensor_shape, loc, dist_no_halo);
    } else {
      // Create a view to the host Elemental matrix
      m_input_host_view = TensorHost(tensor_shape, loc, sample_dist, local_shape);
      // Create a Distconv tensor at host memory.
      m_input_host_tensor = TensorHost(tensor_shape, loc, dist);
    }
    if (!dc::is_cosmoflow_parallel_io_enabled()) {
      // TODO: This is a temporary hack. Should use
      // CUDAHostPooledAllocator, but the shuffler is
      // only specialized for BaseAllocator.
#if 0
      assert0(m_input_host_tensor.allocate());
#else
      size_t buf_size = m_input_host_tensor.get_local_real_size()
          * sizeof(InputType);
      dc::MPIPrintStreamInfo() << "buf size: " << buf_size;
      InputType *buf = nullptr;
      CHECK_CUDA(cudaMallocHost(&buf, buf_size));
      // Note buf should be deallocated.
      dc::tensor::View(m_input_host_tensor, buf);
#endif
    }

    if (!dc::is_cosmoflow_parallel_io_enabled()) {
      setup_shuffler_buffers(m_input_host_view, m_input_host_tensor);
    }

    // Layer::setup_activations_tensor does not work as it assumes
    // prev_activations_tensor is already
    // setup. prev_activations_tensor is not necessary for input.
    //const LocaleMPI loc(dc::get_mpi_comm(), false);
    this->get_activations_t() = TensorDevType(tensor_shape, loc, dist);
    assert0(this->get_activations_t().allocate());
    this->get_activations_t().zero(dc::get_stream());

    // Keeps the same input type and convert to float on GPU
    m_input_dev = TensorDevInput(tensor_shape, loc, dist);
    assert0(m_input_dev.allocate());
    m_input_dev.zero(dc::get_stream());

    // Allocate pinned memory buffer for copying input
    if (dc::is_cosmoflow_parallel_io_enabled()) {
      auto req_size = m_input_dev.get_local_real_size() * sizeof(InputType);
      CHECK_CUDA(cudaMallocHost(&m_copy_pinned_buffer, req_size));
    }

    if (m_copy_labels_dc) {
      setup_label_tensors();
    }
  }

  void setup_tensors_bwd(
      const std::array<dc::Dist, dc::num_dists> &dists) override {
    input_layer<TensorDataType, T_io_buffer, T_layout, Dev>::setup_tensors_bwd(dists);
    if (!this->distconv_enabled()) return;

    // Nothing to do as this is an input layer
  }

  void fp_setup_distconv(El::Int mini_batch_size) override {
    if (!this->distconv_enabled()) return;
    // Nothing to do here as everything is done in fp_compute_distconv.
  }

 protected:

  bool m_copy_labels_dc = false;

  using TensorHost = dc::TensorHost<InputType>;
  using TensorShuffler = dc::TensorHostShuffler<InputType>;
  using TensorDevInput = ::distconv::tensor::Tensor<
    InputType, ::distconv::tensor::LocaleMPI,
    ::distconv::tensor::CUDAAllocator>;

  dc::TensorHost<InputType> m_input_host_view;
  dc::TensorHost<InputType> m_input_host_tensor;
  TensorDevInput m_input_dev;
  // shufflers for the input data
  std::unique_ptr<TensorShuffler> m_input_shuffler;
  // 3 last-MB shufflers for training/validation/testing
  std::array<std::unique_ptr<TensorShuffler>, 3> m_input_shuffler_last_mb;
  std::unique_ptr<InputType> m_shuffler_src_buf;
  size_t m_shuffler_src_buf_size = 0;
  std::unique_ptr<InputType> m_shuffler_dst_buf;
  size_t m_shuffler_dst_buf_size = 0;

  dc::TensorHost<InputType> m_labels_host_view;
  dc::TensorHost<InputType> m_labels_host_tensor;
  TensorDevType m_labels_dev;
  TensorDevInput m_labels_input_type;
  // shufflers for the labels
  std::unique_ptr<TensorShuffler> m_label_shuffler;
  std::array<std::unique_ptr<TensorShuffler>, 3> m_label_shuffler_last_mb;

  InputType *m_copy_pinned_buffer = nullptr;

  using input_layer<TensorDataType, T_io_buffer, T_layout, Dev>::get_activations_t;

  const TensorDevType &get_activations_t(const Layer &child) const override {
    const int child_index = std::find(this->get_child_layers().begin(),
                                      this->get_child_layers().end(),
                                      &child) - this->get_child_layers().begin();
    if (child_index >= this->get_num_children()) {
      LBANN_ERROR("Invalid child layer");
    }
    if (child_index == 0) {
      return this->get_activations_t();
    } else {
      assert_eq(child_index, 1);
      return m_labels_dev;
    }
  }

  void setup_shuffler_buffers(const TensorHost &src, const TensorHost &dst) {
    auto shuffler_src_size = TensorShuffler::get_buf_size(src);
    if (m_shuffler_src_buf_size < shuffler_src_size) {
      m_shuffler_src_buf_size = shuffler_src_size;
      m_shuffler_src_buf =
          std::unique_ptr<InputType>(static_cast<InputType*>(
              dc::util::aligned_malloc(m_shuffler_src_buf_size)));
    }
    auto shuffler_dst_size = TensorShuffler::get_buf_size(dst);
    if (m_shuffler_dst_buf_size < shuffler_dst_size) {
      m_shuffler_dst_buf_size = shuffler_dst_size;
      m_shuffler_dst_buf =
          std::unique_ptr<InputType>(static_cast<InputType*>(
              dc::util::aligned_malloc(m_shuffler_dst_buf_size)));
    }
  }

  TensorShuffler &get_shuffler(const TensorHost &src, const TensorHost &dst,
                               bool is_label) {
    size_t cur_mb_size = src.get_shape()[dc::get_sample_dim()];
    auto src_buf = m_shuffler_src_buf.get();
    auto dst_buf = m_shuffler_dst_buf.get();
    if (cur_mb_size == this->get_model()->get_max_mini_batch_size()) {
      auto &shfl = is_label ? m_label_shuffler : m_input_shuffler;
      if (shfl == nullptr) {
        dc::MPIPrintStreamDebug() << "Creating host shuffler: "
                                  << src << " -> " << dst;
        shfl.reset(new TensorShuffler(
            src, dst, src_buf, dst_buf));
      }
      return *shfl;
    } else {
      // The last remaining mini-batches for the train, validation, and
      // testing modes
      auto mode = this->m_model->get_execution_context().get_execution_mode();
      int shfl_idx = static_cast<int>(mode);
      assert_always(shfl_idx >= 0 && shfl_idx < 3);
      auto &shfl = is_label ? m_label_shuffler_last_mb.at(shfl_idx) :
          m_input_shuffler_last_mb.at(shfl_idx);
      if (shfl == nullptr) {
        dc::MPIPrintStreamDebug() << "Creating host last-mb shuffler: "
                                  << src << " -> " << dst;
        shfl.reset(new TensorShuffler(
            src, dst, src_buf, dst_buf));
      }
      return *shfl;
    }
  }

  void fp_compute_distconv() {
    if (!this->distconv_enabled()) return;

    // Note that the mini-batch size of the data reader is not
    // actually the one for the current mini-batch as the mini-batch
    // index is already updated by fp_compute.
    const int mb_size = static_cast<sgd_execution_context&>(
        this->get_model()->get_execution_context()).get_current_mini_batch_size();
    auto &input_view = m_input_host_view;
    auto &input_tensor = m_input_host_tensor;

    this->get_activations_t().set_outermost_dimension(mb_size);
    m_input_dev.set_outermost_dimension(mb_size);

    assert_eq(mb_size * dc::get_number_of_io_partitions(),
              this->get_activations().Width());
    input_view.set_outermost_dimension(mb_size);
    input_tensor.set_outermost_dimension(mb_size);

    // Setup view
    assert0(dc::tensor::View(
        input_view,
        reinterpret_cast<const InputType*>(
            this->get_activations().LockedBuffer())));

    if (dc::is_cosmoflow_parallel_io_enabled()) {
      // The input buffer is assumed to be already partitioned
      assert0(dc::tensor::View(
          input_tensor, input_view.get_const_buffer()));
    } else {
      dc::MPIPrintStreamDebug()
          << this->get_name()
          << ": Shuffle the input LBANN tensor to Distconv tensor";
      get_shuffler(input_view, input_tensor, false).shuffle_forward(
          input_view.get_const_base_ptr(),
          input_tensor.get_base_ptr());
    }

    // After this, there is no inter-process communication, so it's
    // safe to exit if the local tensor is empty.
    if (input_tensor.get_local_size() == 0) {
      copy_label_distconv(mb_size);
      return;
    }

    dc::MPIPrintStreamDebug()
        << this->get_name()
        << ": Copy the host tensor to device tensor";
    // This should not incur communication as the distributions should
    // be the same except for overlapping width. Device copy should be
    // done with cudaMemcpy3D.
    prof_region_begin("copy-to-device", prof_colors[1], false);
    // TODO: Copy doesn't seem to be working correctly, likely because
    // of the additional halo region in the destination buffer. For
    // now, avoid this with the manual copy below. Also, in the
    // Cosmoflow case, "input_tensor" is not a pinned buffer.
    if (!dc::is_cosmoflow_parallel_io_enabled()) {
      assert0(dc::tensor::Copy(m_input_dev, input_tensor, dc::get_stream()));
    } else {
      int chan_dim = input_tensor.get_local_shape()[::distconv::get_channel_dim()];
      size_t block_size = input_tensor.get_local_size() / chan_dim;
      for (int i = 0; i < chan_dim; ++i) {
        auto dev_off =
            m_input_dev.get_local_offset(dc::IndexVector({0,0,0,i,0}));
        auto host_off = block_size * i;
        // First copy to temporary pinned buffer
        std::memcpy(m_copy_pinned_buffer + dev_off,
                    input_tensor.get_const_buffer() + host_off,
                    sizeof(short) * block_size);
      }
      CHECK_CUDA(cudaMemcpyAsync(
          m_input_dev.get_buffer(),  m_copy_pinned_buffer,
          m_input_dev.get_local_real_size() * sizeof(InputType),
          cudaMemcpyHostToDevice, dc::get_stream()));
    }
    prof_region_end("copy-to-device", false);

    {
      const auto norm_alpha_p = std::getenv("COSMOFLOW_NORMALIZE_ALPHA");
      const auto norm_beta_p  = std::getenv("COSMOFLOW_NORMALIZE_BETA");
      if(norm_alpha_p != nullptr) {
        const auto norm_alpha = std::stod(norm_alpha_p);
        const auto norm_beta = std::stod(norm_beta_p);
        prof_region_begin("cast-scale-bias-from-int16", prof_colors[1], false);
        dc::tensor::CastScaleBias(this->get_activations_t(),
                                  m_input_dev,
                                  (TensorDataType) norm_alpha,
                                  (TensorDataType) norm_beta,
                                  dc::get_stream());
        prof_region_end("cast-scale-bias-from-int16", false);
      } else {
        prof_region_begin("cast-from-int16", prof_colors[1], false);
        dc::tensor::Cast(this->get_activations_t(), m_input_dev, dc::get_stream());
        prof_region_end("cast-from-int16", false);
      }
    }
    // Note: no copy out for activation is necessary as the original
    // LBANN tensor is valid.

    // Copy label as well if necessary
    copy_label_distconv(mb_size);
  }

  // TODO: This is a temporary hack. The label tensor shape should
  //be set based on the shape set by the data reader, but the data
  //reader does not provide it. Using the shape shape as the data
  //tensor works fine for the U-Net model.
  dc::Shape get_unet_label_shape() const {
    auto shape = this->get_output_tensor_shape(0);
    auto label_size = this->get_output_tensor_shape(1).reduce_prod();
    auto num_channels = label_size / shape.reduce_prod();
    shape[-2] = num_channels;
    return shape;
  }

  void setup_label_tensors() {
    using namespace dc;
    assert_always(m_copy_labels_dc);
    //const auto tensor_shape = get_output_tensor_shape(1);
    const auto tensor_shape = get_unet_label_shape();
    const auto sample_dist = dc::get_hydrogen_data_parallel_distribution(get_num_dims());
    auto local_shape = tensor_shape;
    // calculated by Distconv
    local_shape[dc::get_sample_dim()] = 0;
    auto dist = this->get_activations_t().get_distribution();
    // Assumes no halo required.
    dist.clear_overlap();

    const LocaleMPI loc(dc::get_mpi_comm(), false);

    if (dc::is_cosmoflow_parallel_io_enabled()) {
      // Assumes the input buffer is already partitioned for
      // Distconv
      m_labels_host_view = TensorHost(tensor_shape, loc, dist);
      // Create a Distconv tensor at host memory.
      m_labels_host_tensor = TensorHost(tensor_shape, loc, dist);
    } else {
      // Create a view to the host Elemental matrix
      m_labels_host_view = TensorHost(tensor_shape, loc, sample_dist, local_shape);
      // Create a Distconv tensor at host memory.
      m_labels_host_tensor = TensorHost(tensor_shape, loc, dist);
    }

    // When not partitioned yet, setup an intermediate tensor and shuffler
    if (!dc::is_cosmoflow_parallel_io_enabled()) {
      // TODO: This is a temporary hack. Should use
      // CUDAHostPooledAllocator, but the shuffler is
      // only specialized for BaseAllocator.
      size_t buf_size = m_labels_host_tensor.get_local_real_size()
          * sizeof(InputType);
      InputType *buf = nullptr;
      CHECK_CUDA(cudaMallocHost(&buf, buf_size));
      // Note buf should be deallocated.
      dc::tensor::View(m_labels_host_tensor, buf);
      setup_shuffler_buffers(m_labels_host_view, m_labels_host_tensor);
    }

    // Data may be type InputType. Use an intermediate buffer of type
    // InputType, which will be copied to the actual final label
    // tensor with casting to DataType
    m_labels_input_type = TensorDevInput(tensor_shape, loc, dist);
    assert0(m_labels_input_type.allocate());
    m_labels_input_type.zero(dc::get_stream());

    // The final label tensor
    m_labels_dev = TensorDevType(tensor_shape, loc, dist);
    assert0(m_labels_dev.allocate());
    m_labels_dev.zero(dc::get_stream());

    dc::MPIRootPrintStreamInfo() << "label tensor: " << m_labels_dev;
  }

  void copy_label_distconv(int mb_size) {
    if (!m_copy_labels_dc) return;
    constexpr int mat_idx = 1;
    assert_eq(mb_size * dc::get_number_of_io_partitions(),
              this->get_activations(mat_idx).Width());

    // Adjust the sample size
    m_labels_host_view.set_outermost_dimension(mb_size);
    m_labels_host_tensor.set_outermost_dimension(mb_size);
    m_labels_dev.set_outermost_dimension(mb_size);
    m_labels_input_type.set_outermost_dimension(mb_size);

    // Setup view to the LBANN matrix
    assert0(dc::tensor::View(
        m_labels_host_view,
        reinterpret_cast<const InputType*>(
            this->get_activations(mat_idx).LockedBuffer())));

    // Shuffle if necessary
    if (dc::is_cosmoflow_parallel_io_enabled()) {
      // The input buffer is assumed to be already partitioned
      assert0(dc::tensor::View(
          m_labels_host_tensor, m_labels_host_view.get_const_buffer()));
    } else {
      dc::MPIPrintStreamDebug()
          << this->get_name()
          << ": Shuffle the label LBANN tensor to Distconv tensor";
      get_shuffler(m_labels_host_view, m_labels_host_tensor, true).shuffle_forward(
          m_labels_host_view.get_const_base_ptr(),
          m_labels_host_tensor.get_base_ptr());
    }

    // After this, there is no inter-process communication, so it's
    // safe to exit if the local tensor is empty.
    if (m_labels_host_tensor.get_local_size() == 0) {
      return;
    }

    // Cpoy the host tensor to device
    dc::MPIPrintStreamDebug() << "Copy the host label to device tensor";
    prof_region_begin("label-copy-to-device", prof_colors[1], false);
    assert0(dc::tensor::Copy(m_labels_input_type, m_labels_host_tensor, dc::get_stream()));
    prof_region_end("label-copy-to-device", false);

    // Cast to DataType. Just a copy if both tensors are in the same type.
    prof_region_begin("label-cast-from-int16", prof_colors[1], false);
    dc::tensor::Cast(m_labels_dev, m_labels_input_type, dc::get_stream());
    prof_region_end("label-cast-from-int16", false);
  }
#endif // LBANN_HAS_DISTCONV
};

#ifndef LBANN_INPUT_LAYER_DISTCONV_INSTANTIATE
extern template class input_layer_distconv<
  DataType, partitioned_io_buffer<DataType>,
  data_layout::DATA_PARALLEL, El::Device::CPU,
  DataType>;
extern template class input_layer_distconv<
  DataType, partitioned_io_buffer<DataType>,
  data_layout::DATA_PARALLEL, El::Device::CPU,
  int16_t>;
#ifdef LBANN_HAS_GPU
extern template class input_layer_distconv<
  DataType, partitioned_io_buffer<DataType>,
  data_layout::DATA_PARALLEL, El::Device::GPU,
  DataType>;
extern template class input_layer_distconv<
  DataType, partitioned_io_buffer<DataType>,
  data_layout::DATA_PARALLEL, El::Device::GPU,
  int16_t>;
#endif // LBANN_HAS_GPU
#endif // LBANN_INPUT_LAYER_INSTANTIATE


} // namespace lbann

#endif  // LBANN_LAYERS_INPUT_LAYER_DISTCONV_HPP_INCLUDED
