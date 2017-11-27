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
//
// lbann_optimizer .hpp .cpp - Abstract optimizer class
////////////////////////////////////////////////////////////////////////////////

#include "lbann/optimizers/optimizer.hpp"
#include "lbann/utils/cublas_wrapper.hpp"

namespace lbann {

optimizer::optimizer(lbann_comm *comm, DataType learning_rate)
  : m_comm(comm),
    m_cudnn(nullptr),
    m_weights(nullptr),
    m_learning_rate(learning_rate),
    m_gradient(nullptr),
    m_cpu_gradient_is_nonzero(false),
    m_cpu_staging_is_nonzero(false),
    m_staging(nullptr)
#ifdef __LIB_CUDNN
  ,
    m_gpu_gradient_is_nonzero(false),
    m_gpu_staging_is_nonzero(false)
#endif // __LIB_CUDNN
{}

optimizer::optimizer(const optimizer& other)
  : m_comm(other.m_comm),
    m_cudnn(other.m_cudnn),
    m_weights(other.m_weights),
    m_learning_rate(other.m_learning_rate),
    m_gradient(other.m_gradient),
    m_cpu_gradient_is_nonzero(other.m_cpu_gradient_is_nonzero),
    m_cpu_staging_is_nonzero(other.m_cpu_staging_is_nonzero),
    m_staging(other.m_staging)
    #ifdef __LIB_CUDNN
    ,
    m_gpu_gradient_is_nonzero(other.m_gpu_gradient_is_nonzero),
    m_gpu_staging_is_nonzero(other.m_gpu_staging_is_nonzero)
    #endif // __LIB_CUDNN
{
  if (m_gradient != nullptr) { m_gradient = m_gradient->Copy(); }
  if (m_staging != nullptr)  { m_staging = m_staging->Copy(); }
  #ifdef __LIB_CUDNN
  if (m_cudnn != nullptr) {
    const int height = other.m_weights->get_height();
    const int width = other.m_weights->get_width();
    m_gradient_d = m_cudnn->copy(other.m_gradient_d, height, width);
    m_staging_d = m_cudnn->copy(other.m_staging_d, height, width);
  }
  #endif // __LIB_CUDNN
}

optimizer& optimizer::operator=(const optimizer& other) {
  m_comm = other.m_comm;
  m_cudnn = other.m_cudnn;
  m_weights = other.m_weights;
  m_learning_rate = other.m_learning_rate;

  // Copy matrices
  #define COPY_MATRIX(src, dst)                 \
    do {                                        \
      if(src != nullptr && dst != nullptr) {    \
        El::Copy(*src, *dst);                   \
      }                                         \
      if(src != nullptr && dst == nullptr) {    \
        dst = src->Copy();                      \
      }                                         \
      if(src == nullptr && dst != nullptr) {    \
        delete dst;                             \
        dst = nullptr;                          \
      }                                         \
    } while(false)
    COPY_MATRIX(other.m_gradient, m_gradient);
    COPY_MATRIX(other.m_staging, m_staging);
  #undef COPY_MATRIX

  // Copy GPU data
  #ifdef __LIB_CUDNN
  if (m_cudnn != nullptr) {
    const int height = other.m_weights->get_height();
    const int width = other.m_weights->get_width();
    m_cudnn->deallocate_on_gpus(m_gradient_d);
    m_cudnn->deallocate_on_gpus(m_staging_d);
    m_gradient_d = m_cudnn->copy(other.m_gradient_d, height, width);
    m_staging_d = m_cudnn->copy(other.m_staging_d, height, width);
  }
  #endif // __LIB_CUDNN

  return *this;
}

optimizer::~optimizer() {
  if (m_gradient != nullptr) { delete m_gradient; }
  if (m_staging != nullptr)  { delete m_staging; }
  #ifdef __LIB_CUDNN
  if (m_cudnn != nullptr) {
    m_cudnn->deallocate_on_gpus(m_gradient_d);
    m_cudnn->deallocate_on_gpus(m_staging_d);
  }
  #endif // __LIB_CUDNN
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
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to access the weights being optimized before they are set";
    throw lbann_exception(err.str());
  }
  return *m_weights;
}

AbsDistMat& optimizer::get_gradient() {

  // Check if gradient is initialized
  if (!is_initialized()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to access gradients before they are set up";
    throw lbann_exception(err.str());
  }

  // Accumulate CPU allreduce staging matrix
  if (m_cpu_staging_is_nonzero) {
    m_comm->allreduce(*m_staging,
                      m_staging->RedundantComm());
    add_to_gradient(*m_staging);
    El::Zero(*m_staging);
    m_cpu_staging_is_nonzero = false;
  }

#if __LIB_CUDNN

  // Get matrix dimensions
  const int height = m_weights->get_height();
  const int width = m_weights->get_width();

  // Accumulate GPU allreduce staging matrix
  if (m_gpu_staging_is_nonzero) {
    m_cudnn->global_allreduce_on_gpus(m_staging_d,
                                      height,
                                      width,
                                      m_gradient->RedundantComm());
    add_to_gradient_gpu(m_staging_d);
    m_cudnn->clear_on_gpus(m_staging_d, height, width);
    m_gpu_staging_is_nonzero = false;
  }

  // Accumulate gradient in CPU
  if (m_gpu_gradient_is_nonzero) {
    m_cudnn->copy_from_gpu(0,
                           m_staging->Matrix(),
                           m_gradient_d[0]);
    m_cudnn->clear_on_gpus(m_gradient_d, height, width);
    m_cudnn->synchronize();
    add_to_gradient(*m_staging);
    El::Zero(*m_staging);
    m_gpu_gradient_is_nonzero = false;
  }
  
#endif // __LIB_CUDNN

  // Return full gradient
  return *m_gradient;
  
}

#if __LIB_CUDNN
std::vector<DataType*> optimizer::get_gradient_gpu() {

  // Check if gradient is initialized
  if (!is_initialized()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to access gradients before they are set up";
    throw lbann_exception(err.str());
  }
  if (m_cudnn == nullptr) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to get GPU gradient, but GPU is not set up";
    throw lbann_exception(err.str());
  }

  // Get matrix dimensions
  const int height = m_weights->get_height();
  const int width = m_weights->get_width();

  // Accumulate GPU allreduce staging matrix
  if (m_gpu_staging_is_nonzero) {
    m_cudnn->global_allreduce_on_gpus(m_staging_d,
                                      height,
                                      width,
                                      m_gradient->RedundantComm());
    add_to_gradient_gpu(m_staging_d);
    m_cudnn->clear_on_gpus(m_staging_d, height, width);
    m_gpu_staging_is_nonzero = false;
  }

  // Accumulate CPU allreduce staging matrix
  if (m_cpu_staging_is_nonzero) {
    m_comm->allreduce(*m_staging,
                      m_staging->RedundantComm());
    add_to_gradient(*m_staging);
    El::Zero(*m_staging);
    m_cpu_staging_is_nonzero = false;
  }

  // Accumulate gradient in GPU
  if (m_cpu_gradient_is_nonzero) {
    m_cudnn->broadcast_to_gpus(m_staging_d,
                               m_gradient->LockedMatrix());
    m_cudnn->synchronize();
    add_to_gradient_gpu(m_staging_d);
    El::Zero(*m_gradient);
    m_cudnn->clear_on_gpus(m_staging_d, height, width);
    m_cpu_gradient_is_nonzero = false;
  }
  
  // Return full gradient
  return m_gradient_d;

}
#endif // __LIB_CUDNN

void optimizer::clear_gradient() {

  // Check if gradient is initialized
  if (!is_initialized()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to access gradients before they are set up";
    throw lbann_exception(err.str());
  }

  // Zero out matrices
  El::Zero(*m_gradient);
  El::Zero(*m_staging);
#if __LIB_CUDNN
  if (m_cudnn != nullptr) {
    m_cudnn->clear_on_gpus(m_gradient_d,
                           m_weights->get_height(),
                           m_weights->get_width());
    m_cudnn->clear_on_gpus(m_staging_d,
                           m_weights->get_height(),
                           m_weights->get_width());
  }
#endif // __LIB_CUDNN

  // Reset flags
  m_cpu_gradient_is_nonzero = false;
  m_cpu_staging_is_nonzero = false;
#if __LIB_CUDNN
  m_gpu_gradient_is_nonzero = false;
  m_gpu_staging_is_nonzero = false;
#endif // __LIB_CUDNN

}

void optimizer::add_to_gradient(const AbsDistMat& gradient,
                                DataType scale) {
  if (!is_initialized()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to access gradients before they are set up";
    throw lbann_exception(err.str());
  }
  if (scale != DataType(0)) {
    El::Axpy(scale, gradient, *m_gradient);
    m_cpu_gradient_is_nonzero = true;
  }
}

void optimizer::allreduce_and_add_to_gradient(const AbsDistMat& gradient,
                                              DataType scale) {
  if (!is_initialized()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to access gradients before they are set up";
    throw lbann_exception(err.str());
  }
  if (scale != DataType(0)) {
    El::Axpy(scale, gradient, *m_staging);
    m_cpu_staging_is_nonzero = true;
  }
}

#if __LIB_CUDNN

void optimizer::add_to_gradient_gpu(std::vector<DataType*>& gradient,
                                    DataType scale) {
  if (!is_initialized()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to access gradients before they are set up";
    throw lbann_exception(err.str());
  }
  if (m_cudnn == nullptr) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to add to GPU gradient, but GPU is not set up";
    throw lbann_exception(err.str());
  }
  if (scale != DataType(0)) {
    const int num_gpus = m_cudnn->get_num_gpus();
    for(int i=0; i<num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(m_cudnn->get_gpu(i)));
      CHECK_CUBLAS(cublas::axpy(m_cudnn->get_cublas_handle(i),
                                m_weights->get_height() * m_weights->get_width(),
                                scale, gradient[i], 1,
                                m_gradient_d[i], 1));
    }
    m_gpu_gradient_is_nonzero = true;
  }
}

void optimizer::allreduce_and_add_to_gradient_gpu(std::vector<DataType*>& gradient,
                                                  DataType scale) {
  if (!is_initialized()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to access gradients before they are set up";
    throw lbann_exception(err.str());
  }
  if (m_cudnn == nullptr) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to add to GPU gradient, but GPU is not set up";
    throw lbann_exception(err.str());
  }
  if (scale != DataType(0)) {
    const int num_gpus = m_cudnn->get_num_gpus();
    for(int i=0; i<num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(m_cudnn->get_gpu(i)));
      CHECK_CUBLAS(cublas::axpy(m_cudnn->get_cublas_handle(i),
                                m_weights->get_height() * m_weights->get_width(),
                                scale, gradient[i], 1,
                                m_staging_d[i], 1));
    }
    m_gpu_staging_is_nonzero = true;
  }
}

#endif // __LIB_CUDNN

void optimizer::setup(weights& w) {
  if (is_initialized()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to setup an optimizer that is already set up";
    throw lbann_exception(err.str());
  }
  set_weights(w);

  // Initialize matrices
  const int height = m_weights->get_height();
  const int width = m_weights->get_width();
  const AbsDistMat& values = m_weights->get_values();
  m_gradient = values.Construct(values.Grid(), values.Root());
  m_staging = values.Construct(values.Grid(), values.Root());
  El::Zeros(*m_gradient, height, width);
  El::Zeros(*m_staging, height, width);

  // Initialize GPU
  m_cudnn = m_weights->m_cudnn;
  if (m_cudnn != nullptr) {
#ifdef __LIB_CUDNN
    m_cudnn->allocate_on_gpus(m_gradient_d, height, width);
    m_cudnn->allocate_on_gpus(m_staging_d, height, width);
#endif // __LIB_CUDNN
  }

  // Initialize with zero gradient
  clear_gradient();

}

void optimizer::step() {
  if (!is_initialized()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "optimizer must be set up before performing optimization step";
    throw lbann_exception(err.str());
  }
  
  // Apply optimization step
  if (m_cudnn != nullptr) {
  #if __LIB_CUDNN
    std::vector<DataType*> values_d = m_weights->m_values_d;
    std::vector<DataType*> gradient_d = get_gradient_gpu();
    step_compute_gpu(values_d, gradient_d);
  #endif // __LIB_CUDNN
  } else {
    m_weights->get_values(); // Move data to CPU
    AbsDistMat& values = *m_weights->m_values;
    const AbsDistMat& gradient = get_gradient();
    step_compute(values, gradient);
  }

  // Clear gradients
  clear_gradient();

}

#ifdef __LIB_CUDNN
void optimizer::step_compute_gpu(std::vector<DataType*> values_d,
                                 std::vector<DataType*> gradient_d) {
  m_cudnn->copy_from_gpu(0, m_weights->m_values->Matrix(), values_d[0]);
  m_cudnn->copy_from_gpu(0, m_gradient->Matrix(), gradient_d[0]);
  m_cudnn->synchronize();
  step_compute(*m_weights->m_values, *m_gradient);
  m_cudnn->broadcast_to_gpus(values_d, m_weights->m_values->LockedMatrix());
}
#endif // __LIB_CUDNN

}  // namespace lbann
