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
// weights .hpp .cpp - Layer weights class
////////////////////////////////////////////////////////////////////////////////

#include "lbann/weights/weights.hpp"
#include "lbann/optimizers/optimizer.hpp"

namespace lbann {

weights::weights(lbann_comm* comm,
                 cudnn::cudnn_manager* cudnn)
  : m_comm(comm),
    m_cudnn(cudnn),
    m_height(0),
    m_width(0),
    m_values(nullptr),
    m_initializer(nullptr),
    m_optimizer(nullptr) {

  // Initialize weights name
  static int num_weights = 0;
  m_name = "weights" + std::to_string(num_weights);
  num_weights++;

  // Zero initialization is default
  if (m_initializer == nullptr) {
    m_initializer = new constant_initializer(m_comm, DataType(0));
  }

}

weights::weights(const weights& other) 
  : m_name(other.m_name),
    m_comm(other.m_comm),
    m_cudnn(other.m_cudnn),
    m_height(other.m_height),
    m_width(other.m_width),
    m_values(other.m_values),
    m_initializer(other.m_initializer),
    m_optimizer(other.m_optimizer) {

  // Create deep copy of pointers
  if (m_values != nullptr)      { m_values = m_values->Copy(); }
  if (m_initializer != nullptr) { m_initializer = m_initializer->copy(); }
  if (m_optimizer != nullptr) {
    m_optimizer = m_optimizer->copy();
    m_optimizer->set_weights(*this);
  }

  #ifdef __LIB_CUDNN
  // Copy GPU data
  if (m_cudnn != nullptr) {
    m_values_d = m_cudnn->copy(other.m_values_d, m_height, m_width);
  }
  #endif // __LIB_CUDNN

}

weights& weights::operator=(const weights& other) {
  m_name = other.m_name;
  m_comm = other.m_comm;
  m_cudnn = other.m_cudnn;
  m_height = other.m_height;
  m_width = other.m_width;

  // Copy weights matrix
  if (m_values != nullptr && other.m_values != nullptr
      && m_values->DistData() == other.m_values->DistData()) {
    El::Copy(*other.m_values, *m_values);
  }
  if (m_values != nullptr) {
    delete m_values;
    m_values = nullptr;
  }
  if (other.m_values != nullptr) {
    m_values = other.m_values->Copy();
  }

  // Copy initializer
  if (m_initializer != nullptr) {
    delete m_initializer;
    m_initializer = nullptr;
  }
  if (other.m_initializer != nullptr) {
    m_initializer = other.m_initializer->copy();
  }

  // Copy optimizer
  if (m_optimizer != nullptr) {
    delete m_optimizer;
    m_optimizer = nullptr;
  }
  if (other.m_optimizer != nullptr) {
    m_optimizer = other.m_optimizer->copy();
    m_optimizer->set_weights(*this);
  }

  #ifdef __LIB_CUDNN
  // Copy GPU data
  if (m_cudnn != nullptr) {
    m_cudnn->deallocate_on_gpus(m_values_d);
    m_values_d = m_cudnn->copy(other.m_values_d, m_height, m_width);
  }
  #endif // __LIB_CUDNN

  return *this;
}

weights::~weights() {
  if (m_values != nullptr)      { delete m_values; }
  if (m_initializer != nullptr) { delete m_initializer; }
  if (m_optimizer != nullptr)   { delete m_optimizer; }
}

void weights::setup(int height,
                    int width,
                    El::Distribution col_dist,
                    El::Distribution row_dist) {

  // Check if weights has already been set up
  if (m_values != nullptr) {
    const El::DistData dist_data(*m_values);
    if (m_height == height
        && m_width == width
        && dist_data.colDist == col_dist
        && dist_data.rowDist == row_dist) {
      return;
    } else {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "attempted to setup " << m_name << " with "
          << "height=" << height << ","
          << "width=" << width << ","
          << "col_dist=" << col_dist << ","
          << "row_dist=" << row_dist << ", "
          << "but the it is already setup with "
          << "height=" << m_height << ","
          << "width=" << m_width << ","
          << "col_dist=" << dist_data.colDist << ","
          << "row_dist=" << dist_data.rowDist;
      throw lbann_exception(err.str());
    }
  }
  
  // Initialize weights matrix
  m_height = height;
  m_width = width;
  m_values = m_initializer->construct_matrix(m_height,
                                             m_width,
                                             col_dist,
                                             row_dist);

  // Setup GPU objects
  if (m_cudnn != nullptr) {
    setup_gpu();
  }

  // Setup optimizer
  if (m_optimizer != nullptr) {
    m_optimizer->setup(*this);
  }

}

void weights::setup_gpu() {
  #ifndef __LIB_CUDNN
  std::stringstream err;
  err << __FILE__ << " " << __LINE__ << " :: " << "cuDNN not detected";
  throw lbann_exception(err.str());
  #else

  // Check that distributed matrix is in STAR,STAR format
  const El::DistData dist_data(*m_values);
  if (dist_data.colDist != El::STAR || dist_data.rowDist != El::STAR) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to setup weights with "
        << "col_dist=" << dist_data.colDist << ","
        << "row_dist=" << dist_data.rowDist << ", "
        << "but weights with GPU support must have STAR,STAR format";
    throw lbann_exception(err.str());
  }

  // Copy weights matrix to GPU
  m_cudnn->allocate_on_gpus(m_values_d, m_height, m_width);
  m_cudnn->broadcast_to_gpus(m_values_d, m_values->LockedMatrix());

  #endif // __LIB_CUDNN
}

void weights::set_initializer(weights_initializer* initializer) {
  if (m_initializer != nullptr) { delete m_initializer; }
  m_initializer = initializer;
}


void weights::set_optimizer(optimizer* opt) {
  if (m_optimizer != nullptr) { delete m_optimizer; }
  m_optimizer = opt;
}

const AbsDistMat& weights::get_values() {

  // Check if weights matrix has been setup
  if (m_values == nullptr) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to access values of weights before they are setup";
    throw lbann_exception(err.str());
  }

  #if __LIB_CUDNN
  // Copy weights matrix from GPU if needed
  if (m_cudnn != nullptr) {
    m_cudnn->copy_from_gpu(0, m_values->Matrix(), m_values_d[0]);
  }
  #endif // __LIB_CUDNN

  return *m_values;
}

void weights::set_values(const AbsDistMat& values) {
  
  // Check if weights matrix has been setup
  if (m_values == nullptr) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to set values of weights before they are setup";
    throw lbann_exception(err.str());
  }

  // Copy input to weights matrix
  El::Copy(values, *m_values);

  #if __LIB_CUDNN
  // Copy weights matrix to GPU if needed
  if (m_cudnn != nullptr) {
    m_cudnn->broadcast_to_gpus(m_values_d, m_values->Matrix());
  }
  #endif // __LIB_CUDNN

}

void weights::set_value(int row, int col, DataType value) {
  if (m_cudnn == nullptr) {
    if (m_values->IsLocal(row, col)) {
      const El::Int local_row = m_values->LocalRow(row);
      const El::Int local_col = m_values->LocalCol(col);
      m_values->SetLocal(local_row, local_col, value);
    }
  } else {
    #if __LIB_CUDNN
    Mat cpu_value(1, 1);
    cpu_value(0, 0) = value;
    std::vector<DataType*> gpu_value = m_values_d;
    for (DataType*& pointer : gpu_value) {
      pointer += row + col * m_height;
    }
    m_cudnn->broadcast_to_gpus(gpu_value, cpu_value);
    #endif // __LIB_CUDNN
  }
}

void weights::get_values_view(AbsDistMat& values_v) {
  const AbsDistMat& values = get_values();
  if (values.DistData() == values_v.DistData()
      && m_cudnn == nullptr) {
    El::LockedView(values_v, values);
  }
  else {
    #if __LIB_CUDNN
    if (m_cudnn != nullptr) {
      m_cudnn->copy_from_gpu(0, m_values->Matrix(), m_values_d[0]);
    }
    #endif // __LIB_CUDNN
    El::Copy(values, values_v);
  }
}

#ifdef __LIB_CUDNN
std::vector<DataType*> weights::get_values_gpu() {
  if (m_cudnn == nullptr) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "attempted to access weights on GPU when GPU is not setup";
    throw lbann_exception(err.str());
  }
  return m_values_d;
}
#endif // __LIB_CUDN

bool weights::saveToCheckpointShared(lbann::persist& p)
{
  // define name to store our parameters
  char l_name[512];
  sprintf(l_name, "weights_%s_%lldx%lld", m_name.c_str(), m_values->Height(), m_values->Width());
  
  // write out our weights to the model file
  p.write_distmat(persist_type::model, l_name, (DistMat*)m_values);
  //
  // if saving training state, also write out state of optimizer
  m_optimizer->saveToCheckpointShared(p, m_name);
  
  return true;
}

bool weights::loadFromCheckpointShared(lbann::persist& p)
{
  // define name to store our parameters
  char l_name[512];
  sprintf(l_name, "weights_%s_%lldx%lld.bin", m_name.c_str(), m_values->Height(), m_values->Width());
  
  // read our weights from model file
  p.read_distmat(persist_type::model, l_name, (DistMat*)m_values);
  
  // if loading training state, read in state of optimizer
  m_optimizer->loadFromCheckpointShared(p, m_name);
  
  return true;
}

}  // namespace lbann
