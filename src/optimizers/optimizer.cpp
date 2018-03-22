////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#include "lbann/optimizers/optimizer.hpp"
#include "lbann/utils/cublas_wrapper.hpp"
#include "lbann/utils/timer.hpp"

namespace lbann {

optimizer::optimizer(lbann_comm *comm, DataType learning_rate)
  : m_comm(comm),
    m_cudnn(nullptr),
    m_weights(nullptr),
    m_learning_rate(learning_rate),
    m_gradient(nullptr),
    m_gradient_staging(nullptr),
    m_gradient_allreduce_needed(false),
    m_gradient_allreduce_started(false),
    m_gradient_allreduce_finished(false) {}

optimizer::optimizer(const optimizer& other)
  : m_comm(other.m_comm),
    m_cudnn(other.m_cudnn),
    m_weights(other.m_weights),
    m_learning_rate(other.m_learning_rate),
    m_gradient(other.m_gradient),
    #ifdef LBANN_HAS_CUDNN
    m_gradient_d(other.m_gradient_d),
    #endif // LBANN_HAS_CUDNN
    m_gradient_staging(other.m_gradient_staging),
    #ifdef LBANN_HAS_CUDNN
    m_gradient_staging_d(other.m_gradient_staging_d),
    #endif // LBANN_HAS_CUDNN
    m_gradient_allreduce_needed(other.m_gradient_allreduce_needed),
    m_gradient_allreduce_started(other.m_gradient_allreduce_started),
    m_gradient_allreduce_finished(other.m_gradient_allreduce_finished),
    m_step_time(other.m_step_time)
{
  if (m_gradient != nullptr) {
    m_gradient = m_gradient->Copy();
  }
  if (m_gradient_staging != nullptr) {
    m_gradient_staging = m_gradient_staging->Copy();
  }
}

optimizer& optimizer::operator=(const optimizer& other) {
  m_comm = other.m_comm;
  m_cudnn = other.m_cudnn;
  m_weights = other.m_weights;
  m_learning_rate = other.m_learning_rate;
  m_step_time = other.m_step_time;
  m_gradient_allreduce_needed = other.m_gradient_allreduce_needed;
  m_gradient_allreduce_started = other.m_gradient_allreduce_started;
  m_gradient_allreduce_finished = other.m_gradient_allreduce_finished;
  m_gradient_allreduce_started = other.m_gradient_allreduce_started;

  // Deep copy matrices
  if (m_gradient != nullptr) { delete m_gradient; }
  if (m_gradient_staging != nullptr) { delete m_gradient_staging; }
  m_gradient = other.m_gradient;
  m_gradient_staging = other.m_gradient_staging;
  if (m_gradient != nullptr) {
    m_gradient = m_gradient->Copy();
  }
  if (m_gradient_staging != nullptr) {
    m_gradient_staging = m_gradient_staging->Copy();
  }

  // Copy GPU data
  #ifdef LBANN_HAS_CUDNN
  m_gradient_d = other.m_gradient_d;
  m_gradient_staging_d = other.m_gradient_staging_d;
  #endif // LBANN_HAS_CUDNN

  return *this;
}

optimizer::~optimizer() {
  if (m_gradient != nullptr) { delete m_gradient; }
  if (m_gradient_staging != nullptr)  { delete m_gradient_staging; }
}

std::string optimizer::get_description() const {
  std::stringstream ss;
  ss << get_type();
  if (m_weights != nullptr) {
    ss << " (optimizing " << m_weights->get_name() << ")";
  }
  ss << "; learning_rate=" << m_learning_rate;
  return ss.str();
}

weights& optimizer::get_weights() {
  if (!is_initialized()) {
    LBANN_ERROR("attempted to access the weights being optimized before they are set");
  }
  return *m_weights;
}

const AbsDistMat& optimizer::get_gradient() {

  // Check if gradient is initialized
  if (!is_initialized()) {
    LBANN_ERROR("attempted to access gradients before they are set up");
  }

  // Perform allreduce on staging matrix if needed
  if (m_gradient_allreduce_needed && !m_gradient_allreduce_started) {
    start_gradient_staging_allreduce();
  }
  if (m_gradient_allreduce_started && !m_gradient_allreduce_finished) {
    m_comm->wait(m_gradient_allreduce_req);
    m_gradient_allreduce_finished = true;
  }
  if (m_gradient_allreduce_needed) {
    if (m_cudnn == nullptr) {
      add_to_gradient(*m_gradient_staging);
    } else {
      #ifdef LBANN_HAS_CUDNN
      add_to_gradient(m_gradient_staging_d);
      #endif // LBANN_HAS_CUDNN
    }
  }
  m_gradient_allreduce_needed = false;
  m_gradient_allreduce_started = false;
  m_gradient_allreduce_finished = false;

  // Return CPU gradient matrix
  if (m_cudnn != nullptr) {
    #ifdef LBANN_HAS_CUDNN
    m_cudnn->copy_from_gpu(0,
                           m_gradient->Matrix(),
                           m_gradient_d.get_locked_data(0),
                           m_gradient_d.get_leading_dim());
    m_cudnn->synchronize();
    #endif // LBANN_HAS_CUDNN
  }
  return *m_gradient;

}

#ifdef LBANN_HAS_CUDNN
const cudnn::matrix& optimizer::get_gradient_gpu() {

  // Check if gradient is initialized
  if (!is_initialized()) {
    LBANN_ERROR("attempted to access gradients before they are set up");
  }
  if (m_cudnn == nullptr) {
    LBANN_ERROR("attempted to get GPU gradient, but GPU is not set up");
  }

  // Check if all gradient sources have made contributions
  m_gradient_sources.erase(nullptr);
  if (!m_gradient_sources.empty()) {
    std::stringstream err;
    err << "attempted to access gradient before all gradient sources "
        << "have made contributions "
        << "(missing " << m_gradient_sources.size() << " sources)";
    LBANN_ERROR(err.str());
  }

  // Perform allreduce on staging matrix if needed
  if (m_gradient_allreduce_needed && !m_gradient_allreduce_started) {
    start_gradient_staging_allreduce();
  }
  if (m_gradient_allreduce_started && !m_gradient_allreduce_finished) {
    m_comm->wait(m_gradient_allreduce_req);
    m_gradient_allreduce_finished = true;
  }
  if (m_gradient_allreduce_needed) {
    add_to_gradient(m_gradient_staging_d);
  }
  m_gradient_allreduce_needed = false;
  m_gradient_allreduce_started = false;
  m_gradient_allreduce_finished = false;

  // Return gradient
  return m_gradient_d;

}
#endif // LBANN_HAS_CUDNN

void optimizer::start_gradient_staging_allreduce() {
  if (!m_gradient_allreduce_needed || m_gradient_allreduce_started) {
    return;
  }

  m_gradient_allreduce_started = true;
  if (m_cudnn == nullptr) {
    m_comm->nb_allreduce(*m_gradient_staging,
                         m_gradient_staging->RedundantComm(),
                         m_gradient_allreduce_req,
                         El::mpi::SUM,
                         std::type_index(typeid(Al::mpi_backend)));
    m_gradient_allreduce_finished = false;
  } else {
    #ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
    #else
    #if defined(LBANN_HAS_ALUMINUM) && defined(LBANN_HAS_NCCL2)
    // Non-blocking GPU allreduce with NCCL
    // Note: We assume each process has one GPU and that the gradient
    // is in STAR,STAR distribution.
    if (m_cudnn->get_num_gpus() != 1) {
      LBANN_ERROR("non-blocking GPU allreduce with NCCL assumes one GPU per process");
    }
    StarMat gradient_staging_d;
    gradient_staging_d.Attach(m_gradient_staging_d.get_height(),
                              m_gradient_staging_d.get_width_per_gpu(),
                              m_gradient_staging->Grid(),
                              m_gradient_staging->ColAlign(),
                              m_gradient_staging->RowAlign(),
                              m_gradient_staging_d.get_data(0),
                              m_gradient_staging_d.get_leading_dim(),
                              m_gradient_staging->Root());
    m_cudnn->synchronize();
    m_comm->nb_allreduce(gradient_staging_d,
                         gradient_staging_d.RedundantComm(),
                         m_gradient_allreduce_req,
                         El::mpi::SUM,
                         std::type_index(typeid(Al::nccl_backend)));
    m_gradient_allreduce_finished = false;
    #else
    // Naive GPU allreduce
    m_cudnn->global_allreduce_on_gpus(m_gradient_staging_d.get_data(),
                                      m_gradient_staging_d.get_height(),
                                      m_gradient_staging_d.get_width_per_gpu(),
                                      m_gradient->RedundantComm());
    m_gradient_allreduce_finished = true;
    #endif // defined(LBANN_HAS_ALUMINUM) && defined(LBANN_HAS_NCCL2)
    #endif // LBANN_HAS_CUDNN
  }

}

void optimizer::clear_gradient() {

  // Clear matrices
  if (m_cudnn == nullptr) {
    El::Zero(*m_gradient);
  } else {
    #ifdef LBANN_HAS_CUDNN
    m_gradient_d.zero();
    #endif // LBANN_HAS_CUDNN
  }

  // Reset gradient allreduce flags
  m_gradient_allreduce_needed = false;
  m_gradient_allreduce_started = false;
  m_gradient_allreduce_finished = false;

}

void optimizer::add_to_gradient(const AbsDistMat& gradient,
                                DataType scale) {
  if (!is_initialized()) {
    LBANN_ERROR("attempted to access gradients before they are set up");
  }
  if (scale != DataType(0)) {
    if (m_cudnn == nullptr) {
      El::Axpy(scale, gradient, *m_gradient);
    } else {
      #ifndef LBANN_HAS_CUDNN
      LBANN_ERROR("cuDNN not detected");
      #else
      cudnn::matrix gradient_d(m_cudnn);
      gradient_d.attach_to_work_spaces(gradient.LocalHeight(),
                                       gradient.LocalWidth());
      m_cudnn->broadcast_to_gpus(gradient_d.get_data(),
                                 gradient.LockedMatrix());
      add_to_gradient(gradient_d, scale);
      #endif // LBANN_HAS_CUDNN
    }
  }
}
#ifdef LBANN_HAS_CUDNN
void optimizer::add_to_gradient(const cudnn::matrix& gradient_d,
                                DataType scale) {
  if (!is_initialized()) {
    LBANN_ERROR("attempted to access gradients before they are set up");
  }
  if (m_cudnn == nullptr) {
    LBANN_ERROR("attempted to add to GPU gradient, but GPU is not set up");
  }
  if (scale != DataType(0)) {
    for(int i = 0; i < m_cudnn->get_num_gpus(); ++i) {
      CHECK_CUDA(cudaSetDevice(m_cudnn->get_gpu(i)));
      cublas::axpy(m_cudnn->get_cublas_handle(i),
                   m_weights->get_size(),
                   scale, gradient_d.get_locked_data(i), 1,
                   m_gradient_d.get_data(i), 1);
    }
  }
}
#endif // LBANN_HAS_CUDNN

void optimizer::add_to_gradient_staging(const AbsDistMat& gradient,
                                        DataType scale) {
  if (!is_initialized()) {
    LBANN_ERROR("attempted to access gradients before they are set up");
  }
  if (m_gradient_allreduce_started) {
    LBANN_ERROR("attempted to add to staging matrix after gradient accumulation has started");
  }
  if (scale != DataType(0)) {

    // Clear staging matrix if needed
    if (!m_gradient_allreduce_needed) {
      if (m_cudnn == nullptr) {
        El::Zero(*m_gradient_staging);
      } else {
        #ifndef LBANN_HAS_CUDNN
        LBANN_ERROR("cuDNN not detected");
        #else
        m_gradient_staging_d.zero();
        #endif // LBANN_HAS_CUDNN
      }
    }
    m_gradient_allreduce_needed = true;

    // Add to staging matrix
    if (m_cudnn == nullptr) {
      El::Axpy(scale, gradient, *m_gradient_staging);
    } else {
      #ifndef LBANN_HAS_CUDNN
      LBANN_ERROR("cuDNN not detected");
      #else
      cudnn::matrix gradient_d(m_cudnn,
                               gradient.LocalHeight(),
                               gradient.LocalWidth());
      gradient_d.zero();
      m_cudnn->copy_to_gpu(0,
                           gradient_d.get_data(0),
                           gradient.LockedMatrix(),
                           gradient_d.get_leading_dim());
      add_to_gradient_staging(gradient_d, scale);
      #endif // LBANN_HAS_CUDNN
    }

  }
}
#ifdef LBANN_HAS_CUDNN
void optimizer::add_to_gradient_staging(const cudnn::matrix& gradient_d,
                                        DataType scale) {
  if (!is_initialized()) {
    LBANN_ERROR("attempted to access gradients before they are set up");
  }
  if (m_gradient_allreduce_started) {
    LBANN_ERROR("attempted to add to staging matrix after gradient accumulation has started");
  }
  if (m_cudnn == nullptr) {
    LBANN_ERROR("attempted to add to GPU gradient, but GPU is not set up");
  }
  if (scale != DataType(0)) {

    // Clear staging matrix if needed
    if (!m_gradient_allreduce_needed) {
      m_gradient_staging_d.zero();
    }
    m_gradient_allreduce_needed = true;

    // Add to staging matrix
    for(int i = 0; i < m_cudnn->get_num_gpus(); ++i) {
      CHECK_CUDA(cudaSetDevice(m_cudnn->get_gpu(i)));
      cublas::axpy(m_cudnn->get_cublas_handle(i),
                   m_weights->get_size(),
                   scale, gradient_d.get_locked_data(i), 1,
                   m_gradient_staging_d.get_data(i), 1);
    }

  }
}
#endif // LBANN_HAS_CUDNN

void optimizer::add_gradient_source(const void* source) {
  if (source != nullptr) {
    m_gradient_sources.insert(source);
  }
}

void optimizer::remove_gradient_source(const void* source) {
  m_gradient_sources.erase(nullptr);
  m_gradient_sources.erase(source);
  if (m_gradient_sources.empty()) {
    start_gradient_staging_allreduce();
  }
}

void optimizer::setup(weights& w) {
  if (is_initialized()) {
    LBANN_ERROR("attempted to setup an optimizer that is already set up");
  }
  set_weights(w);

  // Initialize matrices
  const int height = m_weights->get_matrix_height();
  const int width = m_weights->get_matrix_width();
  const AbsDistMat& values = m_weights->get_values();

  m_gradient = values.Construct(values.Grid(), values.Root());
  m_gradient_staging = values.Construct(values.Grid(), values.Root());
  m_gradient->Resize(height, width);
  m_gradient_staging->Resize(height, width);

  // Initialize GPU
  m_cudnn = m_weights->m_cudnn;
  if (m_cudnn != nullptr) {
#ifdef LBANN_HAS_CUDNN
    m_gradient_d = cudnn::matrix(m_cudnn, height, width);
    m_gradient_staging_d = cudnn::matrix(m_cudnn, height, width);
#endif // LBANN_HAS_CUDNN
  }

  // Initialize with zero gradient
  clear_gradient();

}

void optimizer::step() {
  if (!is_initialized()) {
    LBANN_ERROR("optimizer must be set up before performing optimization step");
  }

  double step_start = get_time();
  // Apply optimization step
  if (m_cudnn != nullptr) {
  #ifdef LBANN_HAS_CUDNN
    cudnn::matrix values_d(m_cudnn);
    values_d.attach(m_weights->m_values_d, m_weights->get_size());
    const auto& gradient_d = get_gradient_gpu();
    step_compute_gpu(values_d, gradient_d);
  #endif // LBANN_HAS_CUDNN
  } else {
    m_weights->get_values(); // Move data to CPU
    auto& values = *m_weights->m_values;
    const auto& gradient = get_gradient();
    step_compute(values, gradient);
  }

  // Clear gradients
  clear_gradient();

  m_step_time += get_time() - step_start;

}

#ifdef LBANN_HAS_CUDNN
void optimizer::step_compute_gpu(cudnn::matrix& values_d,
                                 const cudnn::matrix& gradient_d) {
  m_cudnn->copy_from_gpu(0, m_weights->m_values->Matrix(), values_d.get_locked_data(0));
  m_cudnn->copy_from_gpu(0, m_gradient->Matrix(), gradient_d.get_locked_data(0));
  m_cudnn->synchronize();
  step_compute(*m_weights->m_values, *m_gradient);
  m_cudnn->broadcast_to_gpus(values_d.get_data(), m_weights->m_values->LockedMatrix());
}
#endif // LBANN_HAS_CUDNN

//************************************************************************
// Checkpointing
//************************************************************************

/**
 * Copies state from GPU to host only if the data is on GPU, which is done
 * asynchronously. Thus, needs synchronization before accessing the states.
 * state is the AbsDistMat type pointer on host, and state_d is the DataType
 * pointer on device
 */
void optimizer::set_mat_state_on_host(AbsDistMat* state, const std::vector<DataType*>& state_d) {
  // Check if states have been setup
  if (state == nullptr) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to set states before they are setup";
    throw lbann_exception(err.str());
  }

  #ifdef LBANN_HAS_CUDNN
  if (m_cudnn != nullptr) {
    if (state_d.empty() || state_d[0] == nullptr) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "attempted to access state on device before they are setup";
      throw lbann_exception(err.str());
    }
    m_cudnn->copy_from_gpu(0, state->Matrix(), state_d[0]);
  }
  #endif // LBANN_HAS_CUDNN
}

/**
 * Copies state from host to GPU if the data has to be on GPU. This is done
 * asynchronously. Thus, needs synchronization before accessing the states.
 * state is the AbsDistMat type pointer on host, and state_d is the DataType
 * pointer on device
 */
void optimizer::set_mat_state_on_device(AbsDistMat* state, std::vector<DataType*>& state_d) {
  // Check if states have been setup
  if (state == nullptr) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to access states before they are setup";
    throw lbann_exception(err.str());
  }

  #ifdef LBANN_HAS_CUDNN
  if (m_cudnn != nullptr) {
    if (state_d.empty() || state_d[0] == nullptr) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "attempted to set state on device before they are setup";
      throw lbann_exception(err.str());
    }
    m_cudnn->broadcast_to_gpus(state_d, state->Matrix());
  }
  #endif // LBANN_HAS_CUDNN
}

bool optimizer::save_to_checkpoint_shared(persist& p, std::string m_name) {
  //  m_learning_rate;
  p.write_datatype(persist_type::train, "learning_rate", m_learning_rate);
  return true;
}

bool optimizer::load_from_checkpoint_shared(persist& p, std::string m_name) {
  p.read_datatype(persist_type::train, "learning_rate", &m_learning_rate);
  MPI_Bcast(&m_learning_rate, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  return true;
}

bool optimizer::save_to_checkpoint_distributed(persist& p, std::string m_name) {
  p.write_datatype(persist_type::train, "learning_rate", m_learning_rate);
  return true;
}

bool optimizer::load_from_checkpoint_distributed(persist& p, std::string m_name) {
  p.read_datatype(persist_type::train, "learning_rate", &m_learning_rate);
  return true;
}
}  // namespace lbann
