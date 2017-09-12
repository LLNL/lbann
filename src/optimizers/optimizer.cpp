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

namespace lbann {

optimizer::optimizer(lbann_comm *comm, DataType learning_rate,
                     cudnn::cudnn_manager *cudnn)
  : m_comm(comm), m_cudnn(cudnn), m_parameters(nullptr),
    m_learning_rate(learning_rate) {}

optimizer::~optimizer() {}

void optimizer::setup(AbsDistMat *parameters) {
  m_parameters = parameters;
  m_height = m_parameters->Height();
  m_width = m_parameters->Width();
  El::DistData dist(*m_parameters);
  if(dist.colDist == El::MC && dist.rowDist == El::MR) {
    m_matrix_format = matrix_format::MC_MR;
  } else if(dist.colDist == El::CIRC && dist.rowDist == El::CIRC) {
    m_matrix_format = matrix_format::CIRC_CIRC;
  } else if(dist.colDist == El::STAR && dist.rowDist == El::STAR) {
    m_matrix_format = matrix_format::STAR_STAR;
  } else if(dist.colDist == El::STAR && dist.rowDist == El::VC) {
    m_matrix_format = matrix_format::STAR_VC;
  } else if(dist.colDist == El::MC && dist.rowDist == El::STAR) {
    m_matrix_format = matrix_format::MC_STAR;
  } else {
    m_matrix_format = matrix_format::invalid;
  }
}

void optimizer::setup_gpu(AbsDistMat *parameters,
                          const std::vector<DataType *> &parameters_d) {
  setup(parameters);
  m_parameters_d = parameters_d;
}

optimizer_factory::optimizer_factory(lbann_comm *comm,
                                     const std::string _name)
  : m_comm(comm), m_name(_name) {}

optimizer_factory::~optimizer_factory() {}

}  // namespace lbann
