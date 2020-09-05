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
#include "lbann/layers/io/input/input_layer.hpp"
#include "lbann/utils/profiling.hpp"
#ifdef LBANN_HAS_DISTCONV
#include "lbann/data_readers/data_reader_hdf5.hpp"
#endif // LBANN_HAS_DISTCONV

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, typename T_io_buffer,
          data_layout T_layout, El::Device Dev>
input_distconv_adapter<TensorDataType, T_io_buffer, T_layout, Dev>::
input_distconv_adapter(Layer& layer): data_type_distconv_adapter<TensorDataType>(layer),
                                      m_shuffle_required(true) {
  // Input data is only processed when its consumer layer is also
  // enabled for distconv
  for (int i = 0; i < layer.get_num_children(); ++i) {
    m_is_input_processed.push_back(layer.get_child_layers()[i]->distconv_enabled());
  }
  auto &l = dynamic_cast<input_layer<
    TensorDataType, T_io_buffer, T_layout, Dev>&>(this->layer());
  // TODO: hdf5_reader is assumed to return a sub-sample partitioned
  // in the same way as specified by the parallel strategy of this input
  // layer. Other data readers are assumed to return a complete
  // sample, thus shuffling is required (unless sample-parallel
  // strategy is given). Conceptually, it seems to make sense if a
  // data reader is annotated with a parallel strategy. Note that,
  // when the HDF5 data reader is used, it is assumed that it is used
  // in all execution modes.
  auto training_dr = l.get_data_reader(execution_mode::training);
  m_shuffle_required = dynamic_cast<hdf5_reader*>(training_dr) == nullptr;
  if (m_shuffle_required) {
    m_shufflers.resize(layer.get_num_children());
  }
}

template <typename TensorDataType, typename T_io_buffer,
          data_layout T_layout, El::Device Dev>
bool input_distconv_adapter<TensorDataType, T_io_buffer, T_layout, Dev>::
is_input_processed(size_t index) const {
  if (index >= m_is_input_processed.size()) {
    LBANN_ERROR("Invalid index: ", index);
  }
  return m_is_input_processed[index];
}

template <typename TensorDataType, typename T_io_buffer,
          data_layout T_layout, El::Device Dev>
typename input_distconv_adapter<TensorDataType, T_io_buffer, T_layout, Dev>::TensorHostShuffler&
input_distconv_adapter<TensorDataType, T_io_buffer, T_layout, Dev>::get_shuffler(
    const TensorHost &src, const TensorHost &dst, int mat_idx) {
  size_t cur_mb_size = src.get_shape()[dc::get_sample_dim()];
  auto src_buf = m_shuffler_src_buf.get();
  auto dst_buf = m_shuffler_dst_buf.get();
  int shfl_idx = -1;
  const auto& context = this->layer().get_model()->get_execution_context();
  if (cur_mb_size == context.get_trainer().get_max_mini_batch_size()) {
    shfl_idx = 0;
  } else {
    // The last remaining mini-batches for the train, validation, and
    // testing modes
    auto mode = context.get_execution_mode();
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

template <typename TensorDataType, typename T_io_buffer,
          data_layout T_layout, El::Device Dev>
void input_distconv_adapter<TensorDataType, T_io_buffer, T_layout, Dev>::setup_fp_tensors() {
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

template <typename TensorDataType, typename T_io_buffer,
          data_layout T_layout, El::Device Dev>
std::unique_ptr<typename input_distconv_adapter<TensorDataType, T_io_buffer, T_layout, Dev>::TensorDevType>
input_distconv_adapter<TensorDataType, T_io_buffer, T_layout, Dev>::
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

template <typename TensorDataType, typename T_io_buffer,
          data_layout T_layout, El::Device Dev>
dc::Shape input_distconv_adapter<TensorDataType, T_io_buffer, T_layout, Dev>::
get_activations_local_shape(int index) const {
  // No enforced local shape as the activations tensor is always
  // copied from the El matrix.
  return dc::Shape(dc::get_num_dims(this->layer()), 0);
}

template <typename TensorDataType, typename T_io_buffer,
          data_layout T_layout, El::Device Dev>
dc::Shape input_distconv_adapter<TensorDataType, T_io_buffer, T_layout, Dev>::
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
    auto num_channels = label_size / shape.reduce_prod();
    shape[-2] = num_channels;
    return shape;
  }
}

template <typename TensorDataType, typename T_io_buffer,
          data_layout T_layout, El::Device Dev>
void input_distconv_adapter<TensorDataType, T_io_buffer, T_layout, Dev>::
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

template <typename TensorDataType, typename T_io_buffer,
          data_layout T_layout, El::Device Dev>
bool input_distconv_adapter<TensorDataType, T_io_buffer, T_layout, Dev>::
child_copy_required(size_t output_index) const {
  // Not required when label is not handled.
  if (output_index == 1 && !is_input_processed(1)) {
    return false;
  } else {
    return data_type_distconv_adapter<TensorDataType>::
        child_copy_required(output_index);
  }
}

template <typename TensorDataType, typename T_io_buffer,
          data_layout T_layout, El::Device Dev>
bool input_distconv_adapter<TensorDataType, T_io_buffer, T_layout, Dev>::
child_shuffle_required(size_t output_index) const {
  // Not required when label is not handled.
  if (output_index == 1 && !is_input_processed(1)) {
    return false;
  } else {
    return data_type_distconv_adapter<TensorDataType>::
        child_shuffle_required(output_index);
  }
}

template <typename TensorDataType, typename T_io_buffer,
          data_layout T_layout, El::Device Dev>
void input_distconv_adapter<TensorDataType, T_io_buffer, T_layout, Dev>::fp_compute() {
  auto &l = dynamic_cast<input_layer<
    TensorDataType, T_io_buffer, T_layout, Dev>&>(this->layer());
  auto stream = hydrogen::cuda::GetDefaultStream();
  // Note that the mini-batch size of the data reader is not
  // actually the one for the current mini-batch as the mini-batch
  // index is already updated by fp_compute.
  const int mb_size = static_cast<sgd_execution_context&>(
      l.get_model()->get_execution_context()).get_current_mini_batch_size();

  for (int mat_idx = 0; mat_idx < l.get_num_children(); ++mat_idx) {
    if (!is_input_processed(mat_idx)) continue;

    assert_eq(mb_size * dc::get_number_of_io_partitions(),
              l.get_activations(mat_idx).Width());

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

template <typename TensorDataType, typename T_io_buffer,
          data_layout T_layout, El::Device Dev>
const input_distconv_adapter<TensorDataType, T_io_buffer, T_layout, Dev>&
input_layer<TensorDataType, T_io_buffer, T_layout, Dev>::get_distconv_adapter() const {
  return dynamic_cast<const input_distconv_adapter<
    TensorDataType, T_io_buffer, T_layout, Dev>&>(
        data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType, typename T_io_buffer,
          data_layout T_layout, El::Device Dev>
input_distconv_adapter<TensorDataType, T_io_buffer, T_layout, Dev>&
input_layer<TensorDataType, T_io_buffer, T_layout, Dev>::get_distconv_adapter() {
  return const_cast<input_distconv_adapter<
    TensorDataType, T_io_buffer, T_layout, Dev>&>(
        static_cast<const input_layer<
        TensorDataType, T_io_buffer, T_layout, Dev>&>(*this).get_distconv_adapter());
}

template <typename TensorDataType,
          typename T_io_buffer,
          data_layout T_layout,
          El::Device Dev>
bool input_layer<TensorDataType, T_io_buffer, T_layout, Dev>::
keep_original_outputs(int index) const {
  // The original output matrices are always needed as we copy them
  // into distconv tensors.
  return true;
}

template <typename TensorDataType,
          typename T_io_buffer,
          data_layout T_layout,
          El::Device Dev>
void input_layer<TensorDataType, T_io_buffer, T_layout, Dev>::
fp_compute() {
  generic_input_layer<TensorDataType>::fp_compute();
  if (this->distconv_enabled()) {
    get_distconv_adapter().fp_compute();
  }
}
#endif // LBANN_HAS_DISTCONV

#define PROTO_DEVICE(T, Device) \
  template class input_layer<T, partitioned_io_buffer<T>, data_layout::DATA_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"

}// namespace lbann
