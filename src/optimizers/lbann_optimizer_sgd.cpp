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
// lbann_optimizer_sgd .hpp .cpp - Stochastic gradient descent optimizer
//
// Inspired by Kera.io implementation
// Stochastic gradient descent, with support for momentum, decay, and Nesterov momentum.
//  lr: float >= 0. Learning rate.
//  momentum: float >= 0. Parameter updates momentum.
//  decay: float >= 0. Learning rate decay over each update.
//  nesterov: boolean. Whether to apply Nesterov momentum.
////////////////////////////////////////////////////////////////////////////////

#include "lbann/optimizers/lbann_optimizer_sgd.hpp"

using namespace std;
using namespace El;

lbann::SGD_factory::SGD_factory(lbann_comm* comm, float lr, float momentum, float decay, bool nesterov)
  : comm(comm), lr(lr), momentum(momentum), decay(decay), nesterov(nesterov)
{
}

lbann::SGD_factory::~SGD_factory()
{

}

lbann::Optimizer *lbann::SGD_factory::create_optimizer(matrix_format format) {
  switch(format) {
  case matrix_format::MC_MR:
    return new SGD<DistMat>(this->comm, this->lr, this->momentum, this->decay, this->nesterov);
  case matrix_format::CIRC_CIRC:
    return new SGD<CircMat>(this->comm, this->lr, this->momentum, this->decay, this->nesterov);
  case matrix_format::STAR_STAR:
    return new SGD<StarMat>(this->comm, this->lr, this->momentum, this->decay, this->nesterov);
  case matrix_format::STAR_VC:
    return new SGD<StarVCMat>(this->comm, this->lr, this->momentum, this->decay, this->nesterov);
  default:
    // TODO: throw an exception
    printf("LBANN Error: unknown matrix distribution for SGD optimizer\n");
    exit(-1);
  }
}
