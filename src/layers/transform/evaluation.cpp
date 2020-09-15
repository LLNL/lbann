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

#define LBANN_EVALUATION_LAYER_INSTANTIATE
#include "lbann/layers/transform/evaluation.hpp"
#include "lbann/models/model.hpp"
#include "lbann/execution_contexts/sgd_execution_context.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/hydrogen_utils.hpp"
#ifdef LBANN_HAS_GPU
#include "lbann/utils/gpu/helpers.hpp"
#endif // LBANN_HAS_GPU

namespace lbann {

namespace {

/** CPU implementation of evaluation layer forward prop. */
template <typename TensorDataType, typename EvalDataType>
void fp_cpu(lbann_comm& comm,
            const El::AbstractDistMatrix<TensorDataType>& input,
            EvalDataType& value,
            Al::request& req) {
  const auto& local_input = input.LockedMatrix();
  const auto& local_height = local_input.Height();
  const auto& local_width = local_input.Width();
  const auto& mini_batch_size = input.Width();
  value = El::TypeTraits<EvalDataType>::Zero();
  LBANN_OMP_PARALLEL_FOR_ARGS(reduction(+:value) collapse(2))
  for (El::Int col = 0; col < local_width; ++col) {
    for (El::Int row = 0; row < local_height; ++row) {
      value += local_input(row, col);
    }
  }
  value = value / mini_batch_size;
  comm.nb_allreduce(&value, 1, input.DistComm(), req);
}

#ifdef LBANN_HAS_HALF
template <typename EvalDataType>
void fp_cpu(lbann_comm& comm,
            const El::AbstractDistMatrix<cpu_fp16>& input,
            EvalDataType& value,
            Al::request& req) {
    LBANN_ERROR("This function is not supported in FP16 on CPUs");
}
#endif // LBANN_HAS_HALF

#ifdef LBANN_HAS_GPU_FP16
template <typename EvalDataType>
void fp_cpu(lbann_comm& comm,
            const El::AbstractDistMatrix<fp16>& input,
            EvalDataType& value,
            Al::request& req) {
    LBANN_ERROR("This function is not supported in FP16 on CPUs");
}
#endif // LBANN_HAS_GPU_HALF

#ifdef LBANN_HAS_GPU
/** GPU implementation of evaluation layer forward prop. */
template <typename TensorDataType, typename EvalDataType>
void fp_gpu(lbann_comm& comm,
            const El::AbstractDistMatrix<TensorDataType>& input,
            EvalDataType& value,
            cuda::event_wrapper& copy_event) {
  const EvalDataType zero = El::TypeTraits<EvalDataType>::Zero();
  const EvalDataType one = El::TypeTraits<EvalDataType>::One();

  // Local matrix
  const auto& local_tdf_input = input.LockedMatrix();
  const auto local_input = ViewIfPossibleOrCopy<TensorDataType, EvalDataType>::get(local_tdf_input);
  const auto& local_height = local_input->Height();
  const auto& local_width = local_input->Width();
  const auto& mini_batch_size = input.Width();

  // GPU objects
  El::Matrix<EvalDataType, El::Device::GPU> sum_d, ones_d;
#ifdef HYDROGEN_HAVE_CUB
  sum_d.SetMemoryMode(1);  // Use CUB GPU memory pool
  ones_d.SetMemoryMode(1); // Use CUB GPU memory pool
#endif // HYDROGEN_HAVE_CUB
  sum_d.Resize(1, 1);

  // Sync object
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(sum_d),
                                     gpu::get_sync_info(ones_d),
                                     gpu::get_sync_info(*local_input));
  El::SyncInfo<El::Device::GPU> const& sync_info = multisync;

  // Setup GPU_BLAS to be in Device pointer mode
  hydrogen::gpu_blas::SetPointerMode(hydrogen::PointerMode::DEVICE);

  // Compute sum of local input matrix entries
  static constexpr El::Int int_one = 1;
  if (local_input->IsEmpty()) {
    El::Zero(sum_d);
  } else if (local_input->Contiguous()) {
    ones_d.Resize(local_height * local_width, 1);
    El::Fill(ones_d, one);
    hydrogen::gpu_blas::Dot(local_height * local_width,
                            local_input->LockedBuffer(), int_one,
                            ones_d.LockedBuffer(), int_one,
                            sum_d.Buffer(),
                            sync_info);
  } else if (local_height == 1) {
    ones_d.Resize(local_width, 1);
    El::Fill(ones_d, one);
    hydrogen::gpu_blas::Dot(local_width,
                            local_input->LockedBuffer(),
                            local_input->LDim(),
                            ones_d.LockedBuffer(), int_one,
                            sum_d.Buffer(),
                            sync_info);
  } else {
    El::Matrix<TensorDataType, El::Device::GPU> col_sums_d;
    El::SetSyncInfo(col_sums_d, sync_info);
#ifdef HYDROGEN_HAVE_CUB
    col_sums_d.SetMemoryMode(1);  // Use CUB GPU memory pool
#endif // HYDROGEN_HAVE_CUB
    col_sums_d.Resize(local_width, 1);
    ones_d.Resize(local_height, 1);
    El::Fill(ones_d, one);
    El::Gemv(El::TRANSPOSE, one, *local_input, ones_d, zero, col_sums_d);
    if (local_width > local_height) {
      ones_d.Resize(local_width, 1);
      El::Fill(ones_d, one);
    }
    hydrogen::gpu_blas::Dot(local_width,
                            col_sums_d.LockedBuffer(), int_one,
                            ones_d.LockedBuffer(), int_one,
                            sum_d.Buffer(),
                            sync_info);
  }
  // Restore the host pointer mode
  hydrogen::gpu_blas::SetPointerMode(hydrogen::PointerMode::HOST);

  // Compute average value across mini-batch
  El::Scale(one / El::To<TensorDataType>(mini_batch_size), sum_d);
  comm.allreduce(
    static_cast<El::AbstractMatrix<TensorDataType>&>(sum_d), input.DistComm());
  hydrogen::gpu::Copy1DToHost(sum_d.LockedBuffer(), &value, 1, sync_info);
  copy_event.record(sync_info.Stream());
}

#ifdef LBANN_HAS_GPU_FP16
template <typename EvalDataType>
void fp_gpu(lbann_comm& comm,
            const El::AbstractDistMatrix<cpu_fp16>& input,
            EvalDataType& value,
            cuda::event_wrapper& copy_event) {
  LBANN_ERROR("This function is not supported with "
              "the CPU FP16 type on GPUs. "
              "A severe logic error has occured; please "
              "report this bug to LBANN developers (or just Tim).");
}
#endif // LBANN_HAS_GPU_HALF

#endif // LBANN_HAS_GPU

} // namespace

template <typename TensorDataType>
EvalType abstract_evaluation_layer<TensorDataType>::get_value(bool scaled) {
  switch (this->get_device_allocation()) {
  case El::Device::CPU: this->get_comm()->wait(m_allreduce_req); break;
#ifdef LBANN_HAS_GPU
  case El::Device::GPU: this->m_copy_event.synchronize(); break;
#endif // LBANN_HAS_GPU
  default: LBANN_ERROR("invalid device");
  }
  if (scaled) { return El::To<EvalDataType>(m_scale) * El::To<EvalDataType>(m_value(0,0)); }
  else        { return m_value(0,0); }
}

template <typename TensorDataType>
abstract_evaluation_layer<TensorDataType>::abstract_evaluation_layer(lbann_comm *comm)
  : transform_layer<TensorDataType>(comm) {
  this->m_expected_num_child_layers = 0;
}

template <typename TensorDataType>
void abstract_evaluation_layer<TensorDataType>::setup_dims(DataReaderMetaData& dr_metadata) {
  transform_layer<TensorDataType>::setup_dims(dr_metadata);
  if (this->get_input_size() != 1) {
    std::stringstream err;
    const auto& dims = this->get_input_dims();
    err << this->get_type() << " layer \"" << this->get_name() << "\" "
        << "expects a scalar input, but "
        << "parent layer \"" << this->get_parent_layers()[0]->get_name() << "\" "
        << "has dimensions of ";
    for (size_t i = 0; i < dims.size(); ++i) {
      err << (i > 0 ? " x " : "") << dims[i];
    }
    LBANN_ERROR(err.str());
  }
}

template <typename TensorDataType>
void abstract_evaluation_layer<TensorDataType>::setup_data(size_t max_mini_batch_size) {
  transform_layer<TensorDataType>::setup_data(max_mini_batch_size);
#ifdef LBANN_HAS_GPU
  m_value.SetMemoryMode(1); // Use pinned memory on host
#endif // LBANN_HAS_GPU
  El::Zeros(m_value, 1, 1);
}

template <typename TensorDataType>
void abstract_evaluation_layer<TensorDataType>::fp_compute() {
  switch (this->get_device_allocation()) {
  case El::Device::CPU:
    fp_cpu(*this->get_comm(),
           this->get_prev_activations(),
           m_value(0, 0),
           m_allreduce_req);
    break;
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    fp_gpu(*this->get_comm(),
           this->get_prev_activations(),
           m_value(0, 0),
           m_copy_event);
    break;
#endif // LBANN_HAS_GPU
  default: LBANN_ERROR("invalid device");
  }
}

template <typename TensorDataType>
void abstract_evaluation_layer<TensorDataType>::bp_compute() {
  const auto& context = static_cast<sgd_execution_context&>(this->m_model->get_execution_context());
  const auto mini_batch_size = context.get_effective_mini_batch_size();
  El::Fill(this->get_error_signals(), TensorDataType(m_scale / mini_batch_size));
}

template <typename TensorDataType>
abstract_evaluation_layer<TensorDataType>*
abstract_evaluation_layer<TensorDataType>::construct(lbann_comm *comm,
                                                     data_layout layout,
                                                     El::Device device) {
#define EVAL_LAYER_CONSTRUCT(T_layout, T_device)                \
  do {                                                          \
    if (layout == T_layout && device == T_device) {             \
      return new evaluation_layer<TensorDataType, T_layout, T_device>(comm); \
    }                                                           \
  } while (false)
  EVAL_LAYER_CONSTRUCT(data_layout::DATA_PARALLEL, El::Device::CPU);
  EVAL_LAYER_CONSTRUCT(data_layout::MODEL_PARALLEL, El::Device::CPU);
#ifdef LBANN_HAS_GPU
  EVAL_LAYER_CONSTRUCT(data_layout::DATA_PARALLEL, El::Device::GPU);
  EVAL_LAYER_CONSTRUCT(data_layout::MODEL_PARALLEL, El::Device::GPU);
#endif // LBANN_HAS_GPU
#undef EVAL_LAYER_CONSTRUCT

  // Could not construct evaluation layer
  LBANN_ERROR("Attempted to construct evaluation layer "
              "with invalid parameters "
              "(data layout type: ", to_string(layout), ", device type: ",
              to_string(device), ")");
  return nullptr;
}

LBANN_LAYER_DEFAULT_BUILDER(evaluation)

#define PROTO(T)                              \
  template class abstract_evaluation_layer<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#undef LBANN_INSTANTIATE_CPU_HALF
#undef LBANN_INSTANTIATE_GPU_HALF

#define PROTO_DEVICE(T, Device)                                           \
  template class evaluation_layer<T, data_layout::DATA_PARALLEL, Device>; \
  template class evaluation_layer<T, data_layout::MODEL_PARALLEL, Device>; \
  LBANN_LAYER_BUILDER_ETI(evaluation, T, Device)

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
