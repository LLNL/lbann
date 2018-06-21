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
#include "lbann/utils/timer.hpp"

namespace lbann {

optimizer::optimizer(lbann_comm *comm, DataType learning_rate)
  : m_comm(comm),
    m_weights(nullptr),
    m_learning_rate(learning_rate),
    m_gradient(nullptr),
    m_gradient_staging(nullptr),
    m_gradient_allreduce_needed(false),
    m_gradient_allreduce_started(false),
    m_gradient_allreduce_finished(false) {}

optimizer::optimizer(const optimizer& other)
  : m_comm(other.m_comm),
    m_weights(other.m_weights),
    m_learning_rate(other.m_learning_rate),
    m_gradient(other.m_gradient),
    m_gradient_staging(other.m_gradient_staging),
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
    add_to_gradient(*m_gradient_staging);
  }
  m_gradient_allreduce_needed = false;
  m_gradient_allreduce_started = false;
  m_gradient_allreduce_finished = false;

  return *m_gradient;

}

void optimizer::start_gradient_staging_allreduce() {
  if (!m_gradient_allreduce_needed || m_gradient_allreduce_started) {
    return;
  }

  m_gradient_allreduce_started = true;
  m_comm->nb_allreduce(*m_gradient_staging,
                       m_gradient_staging->RedundantComm(),
                       m_gradient_allreduce_req,
                       El::mpi::SUM);
  m_gradient_allreduce_finished = false;
}

void optimizer::clear_gradient() {

  // Clear matrices
  El::Zero(*m_gradient);

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
  if (scale == DataType(0)) { return; } 

  // Add to gradient
  const auto dist_data = m_gradient->DistData();
  if (gradient.DistData() == dist_data) {
    El::Axpy(scale, gradient, *m_gradient);
  } else {
    std::unique_ptr<AbsDistMat> workspace(m_gradient->Construct(*dist_data.grid,
                                                                dist_data.root));
#ifdef HYDROGEN_HAVE_CUB
    if (workspace->GetLocalDevice() == El::Device::GPU) {
      workspace->Matrix().SetMemoryMode(1); // CUB GPU memory pool
    }
#endif // HYDROGEN_HAVE_CUB
    El::Copy(gradient, *workspace);
    El::Axpy(scale, *workspace, *m_gradient);
  }

}

void optimizer::add_to_gradient_staging(const AbsDistMat& gradient,
                                        DataType scale) {
  if (!is_initialized()) {
    LBANN_ERROR("attempted to access gradients before they are set up");
  }
  if (m_gradient_allreduce_started) {
    LBANN_ERROR("attempted to add to staging matrix after gradient accumulation has started");
  }
  if (scale == DataType(0)) { return; } 

  // Clear staging matrix if needed
  if (!m_gradient_allreduce_needed) {
    El::Zero(*m_gradient_staging);
  }
  m_gradient_allreduce_needed = true;

  // Add to staging matrix
  const auto dist_data = m_gradient_staging->DistData();
  if (gradient.DistData() == dist_data) {
    El::Axpy(scale, gradient, *m_gradient_staging);
  } else {
    std::unique_ptr<AbsDistMat> workspace(m_gradient_staging->Construct(*dist_data.grid,
                                                                        dist_data.root));
#ifdef HYDROGEN_HAVE_CUB
    if (workspace->GetLocalDevice() == El::Device::GPU) {
      workspace->Matrix().SetMemoryMode(1); // CUB GPU memory pool
    }
#endif // HYDROGEN_HAVE_CUB
    El::Copy(gradient, *workspace);
    El::Axpy(scale, *workspace, *m_gradient_staging);
  }

}

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

  // Initialize with zero gradient
  clear_gradient();

}

void optimizer::step() {
  if (!is_initialized()) {
    LBANN_ERROR("optimizer must be set up before performing optimization step");
  }

  double step_start = get_time();

  // Apply optimization step
  auto& values = m_weights->get_values();
  const auto& gradient = get_gradient();
  switch (values.GetLocalDevice()) {
  case El::Device::CPU:
    step_compute(values, gradient);
    break;
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    step_compute_gpu(values, gradient);
    break;
#endif // LBANN_HAS_GPU
  default:
    std::stringstream err;
    err << "invalid device (" << (int) values.GetLocalDevice() << ")";
    LBANN_ERROR(err.str());
  }

  // Clear gradients
  clear_gradient();

  m_step_time += get_time() - step_start;

}

#ifdef LBANN_HAS_GPU
void optimizer::step_compute_gpu(AbsDistMat& values, const AbsDistMat& gradient) {
  /// @todo Automatically use CPU implementation
  LBANN_ERROR("no GPU implementation detected");
}
#endif // LBANN_HAS_GPU

//************************************************************************
// Checkpointing
//************************************************************************

bool optimizer::save_to_checkpoint_shared(persist& p, std::string m_name) {
  //  m_learning_rate;
  p.write_datatype(persist_type::train, "learning_rate", m_learning_rate);
  return true;
}

bool optimizer::load_from_checkpoint_shared(persist& p, std::string m_name) {
  p.read_datatype(persist_type::train, "learning_rate", &m_learning_rate);
  m_comm->model_broadcast(0, m_learning_rate);
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
