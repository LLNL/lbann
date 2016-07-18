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
// lbann_optimizer_adagrad .hpp .cpp - Stochastic gradient descent optimizer with Adagrad
//
// Inspired by Kera.io implementation
// Stochastic gradient descent with Adagrad.
//  lr: float >= 0. Learning rate.
//  epsilon: float >= 0.
////////////////////////////////////////////////////////////////////////////////

#include "lbann/optimizers/lbann_optimizer_adagrad.hpp"
#include <sys/types.h>
#include <unistd.h>
#include <cmath>

using namespace std;
using namespace El;

lbann::Adagrad_factory::Adagrad_factory(lbann_comm* comm, float lr, float epsilon)
  : comm(comm), lr(lr), epsilon(epsilon)
{
}

lbann::Adagrad_factory::~Adagrad_factory()
{
}

lbann::Optimizer *lbann::Adagrad_factory::create_optimizer(matrix_distribution m_matrix_distribution) {
  switch(m_matrix_distribution) {
  case McMr:
    return new Adagrad<DistMat>(this->comm, this->lr, this->epsilon);
  case CircCirc:
    return new Adagrad<CircMat>(this->comm, this->lr, this->epsilon);
  case StarStar:
    return new Adagrad<StarMat>(this->comm, this->lr, this->epsilon);
  case MrStar:
    return new Adagrad<ColSumMat>(this->comm, this->lr, this->epsilon);
  case StarVc:
    return new Adagrad<StarVCMat>(this->comm, this->lr, this->epsilon);
  default:
    // TODO: throw an exception
    printf("LBANN Error: unknown matrix distribution for Adagrad optimizer\n");
    exit(-1);
  }
}
