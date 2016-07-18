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
// lbann_optimizer_rmsprop .hpp .cpp - Stochastic gradient descent optimizer with RMSprop
//
// Inspired by Kera.io implementation
// Stochastic gradient descent with RMSprop.
//  lr: float >= 0. Learning rate.
//  rho: float >= 0.
//  epsilon: float >= 0. Fuzz factor.
////////////////////////////////////////////////////////////////////////////////

#include "lbann/optimizers/lbann_optimizer_rmsprop.hpp"
#include <sys/types.h>
#include <unistd.h>

using namespace std;
using namespace El;

lbann::RMSprop_factory::RMSprop_factory(lbann_comm* comm, float lr, float rho, float epsilon)
  : comm(comm), lr(lr), rho(rho), epsilon(epsilon)
{
}

lbann::RMSprop_factory::~RMSprop_factory()
{
}

lbann::Optimizer *lbann::RMSprop_factory::create_optimizer(matrix_distribution m_matrix_distribution) {
  switch(m_matrix_distribution) {
  case McMr:
    return new RMSprop<DistMat>(this->comm, this->lr, this->rho, this->epsilon);
  case CircCirc:
    return new RMSprop<CircMat>(this->comm, this->lr, this->rho, this->epsilon);
  case StarStar:
    return new RMSprop<StarMat>(this->comm, this->lr, this->rho, this->epsilon);
  case MrStar:
    return new RMSprop<ColSumMat>(this->comm, this->lr, this->rho, this->epsilon);
  case StarVc:
    return new RMSprop<StarVCMat>(this->comm, this->lr, this->rho, this->epsilon);
  default:
    // TODO: throw an exception
    printf("LBANN Error: unknown matrix distribution for Adagrad optimizer\n");
    exit(-1);
  }
}
