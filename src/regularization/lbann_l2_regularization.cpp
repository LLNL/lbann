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
// lbann_l2_regularization .cpp .hpp - L2 regularization
////////////////////////////////////////////////////////////////////////////////

#include "lbann/lbann_base.hpp"
#include "lbann/regularization/lbann_l2_regularization.hpp"

using namespace El;

namespace lbann {

l2_regularization::l2_regularization(DataType lambda)
  : m_lambda(lambda) {}

l2_regularization::~l2_regularization() {}

void l2_regularization::update_gradients() {
  Axpy(m_lambda, *(m_layer->m_weights), *(m_layer->m_weights_gradient));
}

}  // namespace lbann
