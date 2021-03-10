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

#define LBANN_INPUT_LAYER_INSTANTIATE
#include "lbann/layers/io/input_layer.hpp"

#include "lbann/callbacks/imcomm.hpp"
#include "lbann/execution_contexts/execution_context.hpp"
#include "lbann/execution_contexts/sgd_execution_context.hpp"
#include "lbann/utils/profiling.hpp"
#include "lbann/utils/serialize.hpp"

namespace lbann {

template <typename TensorDataType,
          data_layout T_layout,
          El::Device Dev>
void input_layer<TensorDataType, T_layout, Dev>::
setup_dims(DataReaderMetaData& dr_metadata) {
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);
  for (int i = 0; i < this->get_num_children(); ++i) {
    this->set_output_dims(get_data_dims(dr_metadata, i), i);
  }
}

template <typename TensorDataType,
          data_layout T_layout,
          El::Device Dev>
void input_layer<TensorDataType, T_layout, Dev>::setup_data(size_t max_mini_batch_size) {
  data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);

  // Resize output to maximum mini-batch size
  for (int i = 0; i < this->get_num_children(); ++i) {
    auto& output = this->get_activations(i);
    output.Resize(output.Height(), max_mini_batch_size);
  }
}

template <typename TensorDataType,
          data_layout T_layout,
          El::Device Dev>
void input_layer<TensorDataType, T_layout, Dev>::fp_setup_outputs(El::Int mini_batch_size) {
  /// During model setup there is no valid execution context, but
  /// during execution there is a context
  if(this->m_model->has_valid_execution_context()) {
    auto& c = dynamic_cast<sgd_execution_context&>(this->m_model->get_execution_context());
    auto mode = c.get_execution_mode();
    data_coordinator& dc = get_trainer().get_data_coordinator();
    // Determine model mini-batch size and effective mini-batch size
    // Note: If inter-model communication is activated, the effective
    // mini-batch is equal to the global mini-batch size.
    /// @todo This functionality should probably be moved elsewhere
    mini_batch_size = dc.get_current_mini_batch_size(mode);

    auto effective_mini_batch_size = mini_batch_size;
    for (auto&& cb : this->m_model->get_callbacks()) {
      if (dynamic_cast<callback::imcomm*>(cb) != nullptr) {
        effective_mini_batch_size = dc.get_current_global_mini_batch_size(mode);
        break;
      }
    }

    // Set mini-batch size in model
    c.set_current_mini_batch_size(mini_batch_size);
    c.set_effective_mini_batch_size(effective_mini_batch_size);
  }

  // Initialize matrices
  data_type_layer<TensorDataType>::fp_setup_outputs(mini_batch_size);
}

template <typename TensorDataType,
          data_layout T_layout,
          El::Device Dev>
void input_layer<TensorDataType, T_layout, Dev>::fp_compute()
{
  execution_mode const mode =
    this->m_model->get_execution_context().get_execution_mode();
  buffered_data_coordinator<TensorDataType>& dc =
    static_cast<buffered_data_coordinator<TensorDataType>&>(
      get_trainer().get_data_coordinator());

  //  partitioned_io_buffer<TensorDataType>* io_buffer = dc.get_active_buffer(mode);
  // generic_io_buffer<TensorDataType>* io_buffer = dc.m_io_buffers[dc.get_active_buffer_idx(mode) % dc.m_io_buffers.size()];

  // if(dynamic_cast<partitioned_io_buffer<TensorDataType>*>(io_buffer) != nullptr) {
  // Use the predetermined size of the mini-batch to set the current
  // batch size for the neural network
  int num_samples_in_batch = dc.get_current_mini_batch_size(mode);

  dc.update_num_samples_processed(mode, num_samples_in_batch);
  std::map<input_data_type, AbsDistMatrixType*> input_buffers;
  input_buffers[input_data_type::SAMPLES] = &(this->get_activations(0));
  if(this->m_expected_num_child_layers > 1) {
    if(is_for_regression()) {
      input_buffers[input_data_type::RESPONSES] = &(this->get_activations(1));
    }else {
      input_buffers[input_data_type::LABELS] = &(this->get_activations(1));
    }

    dc.distribute_from_local_matrix(mode, input_buffers);

#ifdef LBANN_HAS_DISTCONV
    if (this->distconv_enabled()) {
      get_distconv_adapter().fp_compute();
    }
#endif // LBANN_HAS_DISTCONV
  }
}

template <typename TensorDataType,
          data_layout T_layout,
          El::Device Dev>
void input_layer<TensorDataType, T_layout, Dev>::
set_samples(const El::AbstractDistMatrix<TensorDataType>& samples) {
  El::Copy(samples, this->get_activations(0));
  this->m_samples_loaded = true;
}

template <typename TensorDataType,
          data_layout T_layout,
          El::Device Dev>
std::vector<int> input_layer<TensorDataType, T_layout, Dev>::
get_data_dims(DataReaderMetaData& dr_metadata, int child_index) const {
  if(child_index == 0) {
    return dr_metadata.data_dims[data_reader_target_mode::INPUT];
  }else if(child_index == 1) {
    return dr_metadata.data_dims[this->m_data_reader_mode];
  }else {
    LBANN_ERROR("get_data_dims: Invalid child index");
  }
  return std::vector<int>(1, 0);
}

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType,
          data_layout T_layout, El::Device Dev>
input_distconv_adapter<TensorDataType, T_layout, Dev>::
input_distconv_adapter(Layer& layer, const bool shuffle_required)
  : data_type_distconv_adapter<TensorDataType>(layer),
  m_shuffle_required(shuffle_required) {
  // Input data is only processed when its consumer layer is also
  // enabled for distconv
  for (int i = 0; i < layer.get_num_children(); ++i) {
    m_is_input_processed.push_back(layer.get_child_layers()[i]->distconv_enabled());
  }
  if (m_shuffle_required) {
    m_shufflers.resize(layer.get_num_children());
  }
}

template <typename TensorDataType,
          data_layout T_layout, El::Device Dev>
bool input_distconv_adapter<TensorDataType, T_layout, Dev>::
is_input_processed(size_t index) const {
  if (index >= m_is_input_processed.size()) {
    LBANN_ERROR("Invalid index: ", index);
  }
  return m_is_input_processed[index];
}

template <typename TensorDataType,
          data_layout T_layout, El::Device Dev>
typename input_distconv_adapter<TensorDataType, T_layout, Dev>::TensorHostShuffler&
input_distconv_adapter<TensorDataType, T_layout, Dev>::get_shuffler(
    const TensorHost &src, const TensorHost &dst, int mat_idx) {
  size_t cur_mb_size = src.get_shape()[dc::get_sample_dim()];
  auto src_buf = m_shuffler_src_buf.get();
  auto dst_buf = m_shuffler_dst_buf.get();
  int shfl_idx = -1;
  if (cur_mb_size == get_trainer().get_max_mini_batch_size()) {
    shfl_idx = 0;
  } else {
    // The last remaining mini-batches for the train, validation, and
    // testing modes
    auto mode =
      this->layer().get_model()->get_execution_context().get_execution_mode();
    shfl_idx = 1 + static_cast<int>(mode);
  }
  assert_always(shfl_idx >= 0 && shfl_idx < 4);
  auto &shfl = m_shufflers[mat_idx][shfl_idx];
  if (shfl == nullptr) {
    shfl = make_unique<TensorHostShuffler>(
        src, dst, src_buf, dst_buf);
  }
  return *shfl;
}

template <typename TensorDataType,
          data_layout T_layout, El::Device Dev>
void input_distconv_adapter<TensorDataType, T_layout, Dev>::setup_fp_tensors() {
  const auto sample_dist = dc::get_hydrogen_data_parallel_distribution(
      dc::get_num_dims(this->layer()));
  for (int mat_idx = 0; mat_idx < this->layer().get_num_children(); ++mat_idx) {
    if (!is_input_processed(mat_idx)) continue;

    const auto shape = this->get_activations_shape(mat_idx);
    auto local_shape = shape;
    if (m_shuffle_required) {
      local_shape[dc::get_sample_dim()] = 0;
    } else {
      local_shape = 0;
    }

    // Use the same MPI communicator for both IO buffers. This seems
    // to work around MPI errors likely caused with the alltoallv for
    // shuffling.
    const dc::LocaleMPI loc(dc::get_mpi_comm(), false);

    auto dist = this->get_activations_dist();
    if (mat_idx == 1) {
      // assumes no halo for the ground-truth data
      dist.clear_overlap();
    }
    auto dist_no_halo = dist;
    dist_no_halo.clear_overlap();

    const auto original_host_tensor_dist = m_shuffle_required ?
        sample_dist : dist_no_halo;
    // Create a view to the host LBANN matrix
    m_original_host_tensors.emplace_back(
        make_unique<TensorHost>(shape, loc, original_host_tensor_dist, local_shape));

    // When shuffled, host tensor will have the same distribution as
    // the final output; otherwise, it is just a view to the host
    // LBANN matrix, so no overlap.
    auto host_tensor_dist = m_shuffle_required ? dist : dist_no_halo;
    m_host_tensors.emplace_back(
        make_unique<TensorHost>(shape, loc, host_tensor_dist));

    if (m_shuffle_required) {
      // TODO: This is a temporary hack. Should use
      // CUDAHostPooledAllocator, but the shuffler is
      // only specialized for BaseAllocator.
      size_t buf_size = m_host_tensors.back()->get_local_real_size()
          * sizeof(TensorDataType);
      TensorDataType *buf = nullptr;
      CHECK_CUDA(cudaMallocHost(&buf, buf_size));
      // Note buf should be deallocated.
      dc::tensor::View(*m_host_tensors.back(), buf);
      setup_shuffler_buffers(*m_original_host_tensors.back(),
                             *m_host_tensors.back());
    }
  }

  this->setup_activations();
}

template <typename TensorDataType,
          data_layout T_layout, El::Device Dev>
std::unique_ptr<typename input_distconv_adapter<TensorDataType, T_layout, Dev>::TensorDevType>
input_distconv_adapter<TensorDataType, T_layout, Dev>::
setup_activations_i(int index) const {
  if (!is_input_processed(index)) return nullptr;
  if (index == 0) {
    return data_type_distconv_adapter<TensorDataType>::
        setup_activations_i(index);
  } else {
    assert_eq(index, 1);
    // Note: the default setup_activations_i can't be used because
    // the distribution might need to be changed to remove
    // overlap. This can be fixed by making each tensor hav a
    // different distribution.
    const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
    auto dist = this->get_activations_dist();
    dist.clear_overlap();
    const auto shape = get_activations_shape(index);
    const auto local_shape = get_activations_local_shape(index);
    auto t = make_unique<TensorDevType>(shape, loc, dist, local_shape);
    assert0(t->allocate());
    t->zero(hydrogen::cuda::GetDefaultStream());
    return t;
  }
}

template <typename TensorDataType,
          data_layout T_layout, El::Device Dev>
dc::Shape input_distconv_adapter<TensorDataType, T_layout, Dev>::
get_activations_local_shape(int index) const {
  // No enforced local shape as the activations tensor is always
  // copied from the El matrix.
  return dc::Shape(dc::get_num_dims(this->layer()), 0);
}

template <typename TensorDataType,
          data_layout T_layout, El::Device Dev>
dc::Shape input_distconv_adapter<TensorDataType, T_layout, Dev>::
get_activations_shape(int index) const {
  if (index == 0) {
    return data_type_distconv_adapter<TensorDataType>::
        get_activations_shape(index);
  } else {
    assert_eq(index, 1);
    // TODO: This is a temporary hack. The label tensor shape should
    //be set based on the shape set by the data reader, but the data
    //reader does not provide it. Using the shape shape as the data
    //tensor works fine for the U-Net model.
    auto shape = this->get_activations_shape(0);
    auto label_size = data_type_distconv_adapter<TensorDataType>::
        get_activations_shape(1).reduce_prod();
    const std::string env = std::getenv("DISTCONV_LABEL_NUM_CHANNELS");
    auto num_channels = env != ""
        ? std::stoi(env) : label_size / shape.reduce_prod();
    shape[-2] = num_channels;
    return shape;
  }
}

template <typename TensorDataType,
          data_layout T_layout, El::Device Dev>
void input_distconv_adapter<TensorDataType, T_layout, Dev>::
setup_shuffler_buffers(const TensorHost &src, const TensorHost &dst) {
  auto shuffler_src_size = TensorHostShuffler::get_buf_size(src);
  if (m_shuffler_src_buf_size < shuffler_src_size) {
    m_shuffler_src_buf_size = shuffler_src_size;
    m_shuffler_src_buf =
        std::unique_ptr<TensorDataType>(static_cast<TensorDataType*>(
            dc::util::aligned_malloc(m_shuffler_src_buf_size)));
  }
  auto shuffler_dst_size = TensorHostShuffler::get_buf_size(dst);
  if (m_shuffler_dst_buf_size < shuffler_dst_size) {
    m_shuffler_dst_buf_size = shuffler_dst_size;
    m_shuffler_dst_buf =
        std::unique_ptr<TensorDataType>(static_cast<TensorDataType*>(
            dc::util::aligned_malloc(m_shuffler_dst_buf_size)));
  }
}

template <typename TensorDataType,
          data_layout T_layout, El::Device Dev>
bool input_distconv_adapter<TensorDataType, T_layout, Dev>::
child_copy_required(size_t output_index) const {
  // Not required when label is not handled.
  if (output_index == 1 && !is_input_processed(1)) {
    return false;
  } else {
    return data_type_distconv_adapter<TensorDataType>::
        child_copy_required(output_index);
  }
}

template <typename TensorDataType,
          data_layout T_layout, El::Device Dev>
bool input_distconv_adapter<TensorDataType, T_layout, Dev>::
child_shuffle_required(size_t output_index) const {
  // Not required when label is not handled.
  if (output_index == 1 && !is_input_processed(1)) {
    return false;
  } else {
    return data_type_distconv_adapter<TensorDataType>::
        child_shuffle_required(output_index);
  }
}

template <typename TensorDataType,
          data_layout T_layout, El::Device Dev>
void input_distconv_adapter<TensorDataType, T_layout, Dev>::fp_compute() {
  auto &l = dynamic_cast<input_layer<
    TensorDataType, T_layout, Dev>&>(this->layer());
  auto stream = hydrogen::cuda::GetDefaultStream();
  // Note that the mini-batch size of the data reader is not
  // actually the one for the current mini-batch as the mini-batch
  // index is already updated by fp_compute.
  const int mb_size = static_cast<sgd_execution_context&>(
      l.get_model()->get_execution_context()).get_current_mini_batch_size();

  for (int mat_idx = 0; mat_idx < l.get_num_children(); ++mat_idx) {
    if (!is_input_processed(mat_idx)) continue;

    // TODO: This is diabled as it raises an error when the HDF5 data
    // reader with hyperslab labels is used. Remove this assertion or
    // reshape the actiavtion tensor (mat_idx=1).
    // assert_eq(mb_size * dc::get_number_of_io_partitions(),
    //           l.get_activations(mat_idx).Width());

    auto &original_tensor = *m_original_host_tensors[mat_idx];
    auto &host_tensor = *m_host_tensors[mat_idx];
    auto &device_tensor = this->get_activations(mat_idx);

    // Adjust the mini-batch size
    original_tensor.set_outermost_dimension(mb_size);
    host_tensor.set_outermost_dimension(mb_size);
    device_tensor.set_outermost_dimension(mb_size);

    // Setup view
    assert0(dc::tensor::View(
        original_tensor,
        l.get_activations(mat_idx).LockedBuffer()));

    // Shuffle if necessary
    if (m_shuffle_required) {
      get_shuffler(
          original_tensor, host_tensor, mat_idx).shuffle_forward(
              original_tensor.get_const_base_ptr(),
              host_tensor.get_base_ptr());
    } else {
      // The input buffer is already partitioned
      assert0(dc::tensor::View(
          host_tensor, original_tensor.get_const_buffer()));
    }

    // After this, there is no inter-process communication, so it's
    // safe to exit if the local tensor is empty.
    if (host_tensor.get_local_size() == 0) {
      continue;
    }

    prof_region_begin("copy-to-device", prof_colors[1], false);
    assert0(dc::tensor::Copy(
        device_tensor, host_tensor, stream));
    prof_region_end("copy-to-device", false);
  }
}

template <typename TensorDataType,
          data_layout T_layout, El::Device Dev>
const input_distconv_adapter<TensorDataType, T_layout, Dev>&
input_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() const {
  return dynamic_cast<const input_distconv_adapter<
    TensorDataType, T_layout, Dev>&>(
        data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType,
          data_layout T_layout, El::Device Dev>
input_distconv_adapter<TensorDataType, T_layout, Dev>&
input_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() {
  return const_cast<input_distconv_adapter<
    TensorDataType, T_layout, Dev>&>(
        static_cast<const input_layer<
        TensorDataType, T_layout, Dev>&>(*this).get_distconv_adapter());
}

template <typename TensorDataType,
          data_layout T_layout,
          El::Device Dev>
bool input_layer<TensorDataType, T_layout, Dev>::
keep_original_outputs(int index) const {
  // The original output matrices are always needed as we copy them
  // into distconv tensors.
  return true;
}
#endif // LBANN_HAS_DISTCONV

#define PROTO_DEVICE(T, Device) \
  template class input_layer<T, data_layout::DATA_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"

}// namespace lbann
