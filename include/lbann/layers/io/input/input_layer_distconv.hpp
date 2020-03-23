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

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, typename T_io_buffer,
          data_layout T_layout, El::Device Dev, typename InputType>
class input_adapter: public data_type_distconv_adapter<TensorDataType> {
 public:
  using TensorDevType = typename data_type_distconv_adapter<TensorDataType>::TensorDevType;
  using TensorHost = dc::TensorHost<InputType>;
  using TensorHostShuffler = dc::TensorHostShuffler<InputType>;
  using TensorDevInput = ::distconv::tensor::Tensor<
    InputType, ::distconv::tensor::LocaleMPI,
    ::distconv::tensor::CUDAAllocator>;

  TensorHost m_input_host_view;
  TensorHost m_input_host_tensor;
  TensorDevInput m_input_dev;
  std::array<std::unique_ptr<TensorHostShuffler>, 4> m_input_shufflers;
  std::unique_ptr<InputType> m_shuffler_src_buf;
  size_t m_shuffler_src_buf_size = 0;
  std::unique_ptr<InputType> m_shuffler_dst_buf;
  size_t m_shuffler_dst_buf_size = 0;
  // TODO: Use pinned memory pool
  InputType *m_copy_pinned_buffer = nullptr;

  bool m_copy_labels = false;
  TensorHost m_labels_host_view;
  TensorHost m_labels_host_tensor;
  TensorDevInput m_labels_input_type;
  std::array<std::unique_ptr<TensorHostShuffler>, 4> m_label_shufflers;

  input_adapter(Layer& layer): data_type_distconv_adapter<TensorDataType>(layer) {}
  virtual ~input_adapter() = default;

  TensorHostShuffler &get_shuffler(const TensorHost &src, const TensorHost &dst,
                                   bool is_label) {
    size_t cur_mb_size = src.get_shape()[dc::get_sample_dim()];
    auto src_buf = m_shuffler_src_buf.get();
    auto dst_buf = m_shuffler_dst_buf.get();
    int shfl_idx = -1;
    const auto &model = *(this->layer().get_model());
    if (cur_mb_size == model.get_max_mini_batch_size()) {
      shfl_idx = 0;
    } else {
      // The last remaining mini-batches for the train, validation, and
      // testing modes
      auto mode = model.get_execution_context().get_execution_mode();
      shfl_idx = 1 + static_cast<int>(mode);
    }
    assert_always(shfl_idx >= 0 && shfl_idx < 4);
    auto &shfl = is_label ? m_label_shufflers[shfl_idx] :
        m_input_shufflers[shfl_idx];
    if (shfl == nullptr) {
      dc::MPIPrintStreamDebug() << "Creating host shuffler: "
                                << src << " -> " << dst;
      shfl = make_unique<TensorHostShuffler>(
          src, dst, src_buf, dst_buf);
    }
    return *shfl;
  }

  void setup_fp_tensors() override {
    const auto &output_dist = this->get_activations_dist();
    const auto tensor_shape = this->get_activations_shape();
    const auto sample_dist = dc::get_hydrogen_data_parallel_distribution(
        this->get_num_dims());
    auto local_shape = tensor_shape;
    // Set the sample dimension as 0 so that its actual value is
    // calculated by Distconv
    local_shape[dc::get_sample_dim()] = 0;
    auto dist_no_halo = output_dist;
    dist_no_halo.clear_overlap();

    // Use the same MPI communicator for both IO buffers. This seems
    // to work around MPI errors likely caused with the alltoallv for
    // shuffling.
    const dc::LocaleMPI loc(dc::get_mpi_comm(), false);

    if (dc::is_cosmoflow_parallel_io_enabled()) {
      // Create a view to the host Elemental matrix that is already
      // partitioned.
      m_input_host_view = TensorHost(tensor_shape, loc, dist_no_halo);
      // Create a Distconv tensor at host memory. This will be just a
      // view to m_input_host_view.
      m_input_host_tensor = TensorHost(tensor_shape, loc, dist_no_halo);
    } else {
      // Create a view to the host Elemental matrix
      m_input_host_view = TensorHost(tensor_shape, loc, sample_dist, local_shape);
      // Create a Distconv tensor at host memory.
      m_input_host_tensor = TensorHost(tensor_shape, loc, output_dist);
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

    this->setup_activations();

    // Keeps the same input type and convert to float on GPU
    m_input_dev = TensorDevInput(tensor_shape, loc, output_dist);
    assert0(m_input_dev.allocate());
    m_input_dev.zero(El::GPUManager::Stream());

    // Allocate pinned memory buffer for copying input
    if (dc::is_cosmoflow_parallel_io_enabled()) {
      auto req_size = m_input_dev.get_local_real_size() * sizeof(InputType);
      CHECK_CUDA(cudaMallocHost(&m_copy_pinned_buffer, req_size));
    }

    // copies the label data as well when the second child layer is
    // also enabled for distconv
    if (this->layer().get_num_children() == 2 &&
        this->layer().get_child_layers()[1]->distconv_enabled()) {
      m_copy_labels = true;
      dc::MPIRootPrintStreamInfo() << "Copy label/response data to Distconv as well";
      setup_label_tensors(output_dist);
    }
  }

  void setup_shuffler_buffers(const TensorHost &src, const TensorHost &dst) {
    auto shuffler_src_size = TensorHostShuffler::get_buf_size(src);
    if (m_shuffler_src_buf_size < shuffler_src_size) {
      m_shuffler_src_buf_size = shuffler_src_size;
      m_shuffler_src_buf =
          std::unique_ptr<InputType>(static_cast<InputType*>(
              dc::util::aligned_malloc(m_shuffler_src_buf_size)));
    }
    auto shuffler_dst_size = TensorHostShuffler::get_buf_size(dst);
    if (m_shuffler_dst_buf_size < shuffler_dst_size) {
      m_shuffler_dst_buf_size = shuffler_dst_size;
      m_shuffler_dst_buf =
          std::unique_ptr<InputType>(static_cast<InputType*>(
              dc::util::aligned_malloc(m_shuffler_dst_buf_size)));
    }
  }

  // No enforced local shape as the activations tensor is always
  // copied from the El matrix.
  dc::Shape get_activations_local_shape(int index=0) const override {
    return dc::Shape(this->get_num_dims(), 0);
  }

  // TODO: This is a temporary hack. The label tensor shape should
  //be set based on the shape set by the data reader, but the data
  //reader does not provide it. Using the shape shape as the data
  //tensor works fine for the U-Net model.
  dc::Shape get_unet_label_shape() const {
    auto shape = this->get_activations_shape(0);
    auto label_size = this->get_activations_shape(1).reduce_prod();
    auto num_channels = label_size / shape.reduce_prod();
    shape[-2] = num_channels;
    return shape;
  }

  void setup_label_tensors(const dc::Dist &output_dist) {
    using namespace dc;
    assert_always(m_copy_labels);
    //const auto tensor_shape = get_output_tensor_shape(1);
    const auto tensor_shape = get_unet_label_shape();
    const auto sample_dist = dc::get_hydrogen_data_parallel_distribution(
        this->get_num_dims());
    auto local_shape = tensor_shape;
    // calculated by Distconv
    local_shape[dc::get_sample_dim()] = 0;
    auto label_dist = output_dist;
    // Assumes no halo required.
    label_dist.clear_overlap();

    const LocaleMPI loc(dc::get_mpi_comm(), false);

    if (dc::is_cosmoflow_parallel_io_enabled()) {
      // Assumes the input buffer is already partitioned for
      // Distconv
      m_labels_host_view = TensorHost(tensor_shape, loc, label_dist);
      // Create a Distconv tensor at host memory.
      m_labels_host_tensor = TensorHost(tensor_shape, loc, label_dist);
    } else {
      // Create a view to the host Elemental matrix
      m_labels_host_view = TensorHost(tensor_shape, loc, sample_dist, local_shape);
      // Create a Distconv tensor at host memory.
      m_labels_host_tensor = TensorHost(tensor_shape, loc, label_dist);
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
    m_labels_input_type = TensorDevInput(tensor_shape, loc, label_dist);
    assert0(m_labels_input_type.allocate());
    m_labels_input_type.zero(El::GPUManager::Stream());

    // The final label tensor
    this->m_outputs.emplace_back(
        make_unique<TensorDevType>(tensor_shape, loc, label_dist));
    auto &label_tensor = *(this->m_outputs.back());
    assert0(label_tensor.allocate());
    label_tensor.zero(El::GPUManager::Stream());

    dc::MPIRootPrintStreamInfo() << "label tensor: " << label_tensor;
  }

  // No bp tensors needed for this layer.
  void setup_prev_error_signals(const dc::Dist& dist) {}
  void setup_original_prev_error_signals() {}
  void setup_error_signals(const dc::Dist& dist) {}
  void setup_original_error_signals() {}
  void setup_bp_tensors() override {}

  // Nothing to do here as everything is done in fp_compute_distconv.
  void fp_setup(El::Int mini_batch_size) override {}
};
#endif // LBANN_HAS_DISTCONV

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
  friend class input_adapter<TensorDataType, T_io_buffer, T_layout, Dev, InputType>;
 protected:
  bool is_distconv_supported() const override { return true; }

  input_adapter<TensorDataType, T_io_buffer, T_layout, Dev, InputType>& dc() override;
  const input_adapter<TensorDataType, T_io_buffer, T_layout, Dev, InputType>& dc() const override;

  void setup_distconv_adapter() override {
    this->get_dc() = make_unique<
      input_adapter<TensorDataType, T_io_buffer, T_layout, Dev, InputType>>(*this);
  }

  void fp_compute_distconv() {
    if (!this->distconv_enabled()) return;

    // Note that the mini-batch size of the data reader is not
    // actually the one for the current mini-batch as the mini-batch
    // index is already updated by fp_compute.
    const int mb_size = static_cast<sgd_execution_context&>(
        this->get_model()->get_execution_context()).get_current_mini_batch_size();
    auto &input_view = dc().m_input_host_view;
    auto &input_tensor = dc().m_input_host_tensor;

    this->dc().get_activations().set_outermost_dimension(mb_size);
    dc().m_input_dev.set_outermost_dimension(mb_size);

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
      dc().get_shuffler(input_view, input_tensor, false).shuffle_forward(
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
      assert0(dc::tensor::Copy(dc().m_input_dev, input_tensor, El::GPUManager::Stream()));
    } else {
      int chan_dim = input_tensor.get_local_shape()[::distconv::get_channel_dim()];
      size_t block_size = input_tensor.get_local_size() / chan_dim;
      for (int i = 0; i < chan_dim; ++i) {
        auto dev_off =
            dc().m_input_dev.get_local_offset(dc::IndexVector({0,0,0,i,0}));
        auto host_off = block_size * i;
        // First copy to temporary pinned buffer
        std::memcpy(dc().m_copy_pinned_buffer + dev_off,
                    input_tensor.get_const_buffer() + host_off,
                    sizeof(short) * block_size);
      }
      CHECK_CUDA(cudaMemcpyAsync(
          dc().m_input_dev.get_buffer(),  dc().m_copy_pinned_buffer,
          dc().m_input_dev.get_local_real_size() * sizeof(InputType),
          cudaMemcpyHostToDevice, El::GPUManager::Stream()));
    }
    prof_region_end("copy-to-device", false);

    {
      const auto norm_alpha_p = std::getenv("COSMOFLOW_NORMALIZE_ALPHA");
      const auto norm_beta_p  = std::getenv("COSMOFLOW_NORMALIZE_BETA");
      if(norm_alpha_p != nullptr) {
        const auto norm_alpha = std::stod(norm_alpha_p);
        const auto norm_beta = std::stod(norm_beta_p);
        prof_region_begin("cast-scale-bias-from-int16", prof_colors[1], false);
        dc::tensor::CastScaleBias(this->dc().get_activations(),
                                  dc().m_input_dev,
                                  (TensorDataType) norm_alpha,
                                  (TensorDataType) norm_beta,
                                  El::GPUManager::Stream());
        prof_region_end("cast-scale-bias-from-int16", false);
      } else {
        prof_region_begin("cast-from-int16", prof_colors[1], false);
        dc::tensor::Cast(this->dc().get_activations(), dc().m_input_dev, El::GPUManager::Stream());
        prof_region_end("cast-from-int16", false);
      }
    }
    // Note: no copy out for activation is necessary as the original
    // LBANN tensor is valid.

    // Copy label as well if necessary
    copy_label_distconv(mb_size);
  }

  void copy_label_distconv(int mb_size) {
    if (!dc().m_copy_labels) return;
    constexpr int mat_idx = 1;
    assert_eq(mb_size * dc::get_number_of_io_partitions(),
              this->get_activations(mat_idx).Width());

    auto &labels_dev = this->dc().get_activations(1);

    // Adjust the sample size
    dc().m_labels_host_view.set_outermost_dimension(mb_size);
    dc().m_labels_host_tensor.set_outermost_dimension(mb_size);
    labels_dev.set_outermost_dimension(mb_size);
    dc().m_labels_input_type.set_outermost_dimension(mb_size);

    // Setup view to the LBANN matrix
    assert0(dc::tensor::View(
        dc().m_labels_host_view,
        reinterpret_cast<const InputType*>(
            this->get_activations(mat_idx).LockedBuffer())));

    // Shuffle if necessary
    if (dc::is_cosmoflow_parallel_io_enabled()) {
      // The input buffer is assumed to be already partitioned
      assert0(dc::tensor::View(
          dc().m_labels_host_tensor,
          dc().m_labels_host_view.get_const_buffer()));
    } else {
      dc::MPIPrintStreamDebug()
          << this->get_name()
          << ": Shuffle the label LBANN tensor to Distconv tensor";
      dc().get_shuffler(dc().m_labels_host_view,
                        dc().m_labels_host_tensor, true).shuffle_forward(
                            dc().m_labels_host_view.get_const_base_ptr(),
                            dc().m_labels_host_tensor.get_base_ptr());
    }

    // After this, there is no inter-process communication, so it's
    // safe to exit if the local tensor is empty.
    if (dc().m_labels_host_tensor.get_local_size() == 0) {
      return;
    }

    // Cpoy the host tensor to device
    dc::MPIPrintStreamDebug() << "Copy the host label to device tensor";
    prof_region_begin("label-copy-to-device", prof_colors[1], false);
    assert0(dc::tensor::Copy(
        dc().m_labels_input_type, dc().m_labels_host_tensor, El::GPUManager::Stream()));
    prof_region_end("label-copy-to-device", false);

    // Cast to DataType. Just a copy if both tensors are in the same type.
    prof_region_begin("label-cast-from-int16", prof_colors[1], false);
    dc::tensor::Cast(labels_dev, dc().m_labels_input_type, El::GPUManager::Stream());
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
