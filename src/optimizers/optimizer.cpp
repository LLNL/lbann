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

optimizer::optimizer(DataType learning_rate)
  : m_cudnn(nullptr),
    m_weights(nullptr),
    m_learning_rate(learning_rate),
    m_gradient(nullptr),
    m_gradient_allreduce_staging(nullptr)
    #ifdef __LIB_CUDNN
    ,
    m_gradient_gpu_staging(nullptr)
    #endif // __LIB_CUDNN
{}

optimizer::optimizer(const optimizer& other)
  : m_cudnn(other.m_cudnn),
    m_weights(other.m_weights),
    m_learning_rate(other.m_learning_rate),
    m_gradient(other.m_gradient),
    m_gradient_allreduce_staging(other.m_gradient_allreduce_staging)
    #ifdef __LIB_CUDNN
    ,
    m_gradient_gpu_staging(other.m_gradient_gpu_staging)
    #endif // __LIB_CUDNN
{
  if (m_gradient != nullptr)  {
    m_gradient = m_gradient->Copy();
  }
  if (m_gradient_allreduce_staging != nullptr) {
    m_gradient_allreduce_staging = m_gradient_allreduce_staging->Copy();
  }
  #ifdef __LIB_CUDNN
  if (!other.m_gradient_allreduce_staging_d.empty()) {
    m_cudnn->allocate_on_gpus(m_gradient_allreduce_staging_d,
                              other.m_weights->get_height(),
                              other.m_weights->get_width());
    m_cudnn->copy_on_gpus(m_gradient_allreduce_staging_d,
                          other.m_gradient_allreduce_staging_d,
                          other.m_weights->get_height(),
                          other.m_weights->get_width());
  }
  if (m_gradient_gpu_staging != nullptr) {
    m_gradient_gpu_staging = m_gradient_gpu_staging->Copy();
    m_cudnn->pin_matrix(*m_gradient_gpu_staging);
  }
  #endif // __LIB_CUDNN
}

optimizer& optimizer::operator=(const optimizer& other) {
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
    COPY_MATRIX(other.m_gradient_allreduce_staging,
                m_gradient_allreduce_staging);
  #ifdef __LIB_CUDNN
    COPY_MATRIX(other.m_gradient_gpu_staging, m_gradient_gpu_staging);
  #endif // __LIB_CUDNN
  #undef COPY_MATRIX

  // Copy GPU data
  #ifdef __LIB_CUDNN
  if (m_cudnn != nullptr) {
    m_cudnn->deallocate_on_gpus(m_gradient_allreduce_staging_d);
    m_gradient_allreduce_staging_d = m_cudnn->copy(other.m_gradient_allreduce_staging_d,
                                                   other.m_weights->get_height(),
                                                   other.m_weights->get_width());

    // Pin staging matrix
    if (m_gradient_gpu_staging) {
      m_cudnn->pin_matrix(*m_gradient_gpu_staging);
    }

  }
  #endif // __LIB_CUDNN

  return *this;
}

optimizer::~optimizer() {
  if (m_gradient != nullptr) {
    delete m_gradient;
  }
  if (m_gradient_allreduce_staging != nullptr) {
    delete m_gradient_allreduce_staging;
  }
  #ifdef __LIB_CUDNN
  if (m_cudnn != nullptr) {
    m_cudnn->deallocate_on_gpus(m_gradient_allreduce_staging_d);
  }
  if (m_gradient_gpu_staging != nullptr) {
    delete m_gradient_gpu_staging;
  }
  #endif // __LIB_CUDNN
}

std::string optimizer::get_description() const {
  std::stringstream ss;
  ss << get_type();
  if (m_weights != nullptr) {
    ss << " is optimizing " << m_weights->get_name();
  } else {
    ss << " is not optimizing anything";
  }
  ss << "; learning_rate=" << m_learning_rate;
  return ss.str();
}

weights& optimizer::get_weights() {
  if (m_weights == nullptr) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to access the weights being optimized before they are set";
    throw lbann_exception(err.str());
  }
  return *m_weights;
}

AbsDistMat& optimizer::get_gradient() {
  if (m_gradient == nullptr) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to access gradients before they are set";
    throw lbann_exception(err.str());
  }
  return *m_gradient;
}

const AbsDistMat& optimizer::get_gradient() const {
  if (m_gradient == nullptr) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to access gradients before they are set";
    throw lbann_exception(err.str());
  }
  return *m_gradient;
}

void optimizer::clear_gradient() {
  El::Zero(get_gradient());
  if (m_gradient_allreduce_staging != nullptr) {
    El::Zero(*m_gradient_allreduce_staging);
  }
#if __LIB_CUDNN
  if (!m_gradient_allreduce_staging_d.empty()) {
    m_cudnn->clear_on_gpus(m_gradient_allreduce_staging_d,
                           m_weights->get_height(),
                           m_weights->get_width());
  }
#endif // __LIB_CUDNN
}

void optimizer::allreduce_and_add_to_gradient(const AbsDistMat& gradient) {
  if (m_gradient_allreduce_staging == nullptr) {
    AbsDistMat& full_gradient = get_gradient();
    m_gradient_allreduce_staging = full_gradient.Construct(full_gradient.Grid(),
                                                           full_gradient.Root());
    El::Zeros(*m_gradient_allreduce_staging,
              full_gradient.Height(),
              full_gradient.Width());
  }
  El::Axpy(DataType(1), gradient, *m_gradient_allreduce_staging);
}

#if __LIB_CUDNN
void optimizer::gpu_allreduce_and_add_to_gradient(std::vector<DataType*>& gradient) {
  
  // Initialize staging matrices if needed
  if (m_gradient_allreduce_staging_d.empty()) {

    // Initialize GPU memory
    m_cudnn->allocate_on_gpus(m_gradient_allreduce_staging_d,
                              m_weights->get_height(),
                              m_weights->get_width());
    m_cudnn->clear_on_gpus(m_gradient_allreduce_staging_d,
                           m_weights->get_height(),
                           m_weights->get_width());

    // Initialize staging matrix for GPU memory transfers
    AbsDistMat& full_gradient = get_gradient();
    m_gradient_gpu_staging = full_gradient.Construct(full_gradient.Grid(),
                                                     full_gradient.Root());
    El::Zeros(*m_gradient_gpu_staging,
              full_gradient.Height(),
              full_gradient.Width());
    m_cudnn->pin_matrix(*m_gradient_gpu_staging);
  }

  // Add input to staging matrix
  const DataType one = DataType(1);
  const int num_gpus = m_cudnn->get_num_gpus();
  for(int i=0; i<num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(m_cudnn->get_gpu(i)));
    CHECK_CUBLAS(cublas::axpy(m_cudnn->get_cublas_handle(i),
                              m_weights->get_height() * m_weights->get_width(),
                              &one, gradient[i], 1,
                              m_gradient_allreduce_staging_d[i], 1));
  }

}
#endif // __LIB_CUDNN

void optimizer::setup(weights& w) {
  if (m_weights != nullptr) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to setup an optimizer that is already set up";
    throw lbann_exception(err.str());
  }
  set_weights(w);

  // Initialize gradient matrix
  const int height = m_weights->get_height();
  const int width = m_weights->get_width();
  const AbsDistMat& values = m_weights->get_values();
  m_gradient = values.Construct(values.Grid(), values.Root());
  El::Zeros(*m_gradient, height, width);

  // Initialize GPU
  m_cudnn = m_weights->m_cudnn;

}

void optimizer::step() {

  // Accumulate gradient with allreduce
  AbsDistMat& gradient = get_gradient();
  #ifdef __LIB_CUDNN
  if (m_cudnn != nullptr && !m_gradient_allreduce_staging_d.empty()) {
    const int height = m_weights->get_height();
    const int width = m_weights->get_width();
    m_cudnn->allreduce(m_gradient_allreduce_staging_d, height, width);
    m_cudnn->copy_from_gpu(0,
                           m_gradient_gpu_staging->Matrix(),
                           m_gradient_allreduce_staging_d[0]);
    allreduce_and_add_to_gradient(*m_gradient_gpu_staging);
  }
  #endif // __LIB_CUDNN
  if (m_gradient_allreduce_staging != nullptr) {
    El::AllReduce(*m_gradient_allreduce_staging, m_gradient_allreduce_staging->RedundantComm());
    El::Axpy(DataType(1), *m_gradient_allreduce_staging, gradient);
  }

  // Get weights matrix
  // Note: need to make sure data is copied from GPU to CPU
  m_weights->get_values();
  AbsDistMat& values = *m_weights->m_values;

  // Apply optimization step
  step_compute(values, gradient);
  
  #if __LIB_CUDNN
  // Copy result to GPU if needed
  if (m_cudnn != nullptr) {
    m_cudnn->broadcast_to_gpus(m_weights->m_values_d, values.LockedMatrix());
  }
  #endif // __LIB_CUDNN

  // Clear gradients
  clear_gradient();

}

}  // namespace lbann
