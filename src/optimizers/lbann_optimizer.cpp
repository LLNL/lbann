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
// lbann_optimizer .hpp .cpp - Abstract class for neural network optimizers
////////////////////////////////////////////////////////////////////////////////

#include "lbann/optimizers/lbann_optimizer.hpp"
#include "lbann/optimizers/lbann_optimizer_sgd.hpp"
#include "lbann/optimizers/lbann_optimizer_adagrad.hpp"
#include "lbann/optimizers/lbann_optimizer_rmsprop.hpp"

using namespace std;
using namespace El;

void lbann::Optimizer::update_weight_bias_matrix(ElMat &WB_D, ElMat& WB) {
  // TODO: throw exception
  printf("LBANN Error: optimizer update not defined\n");
  exit(-1);
}

#if 0
lbann::Optimizer::Optimizer() {

}

lbann::Optimizer::~Optimizer() {

}

static Optimizer *lbann::Optimizer::create_optimizer(int method) {
  Optimizer *optimizer = NULL

  if (LearnRateMethod == 1) { // Adagrad
    optimizer = new Adagrad((float) 0.1 /*0.01*/, (float) 1e-6, grid);
  }else if (LearnRateMethod == 2) { // RMSprop
    optimizer = new RMSprop((float) 0.001, (float) 0.9, (float) 1e-6, grid);
  }else {
    optimizer = new SGD();
  }
  return optimizer;
}

bool lbann::Optimizer::saveToCheckpoint(int fd, const char* filename, uint64_t* bytes) {
    return true;
}

bool lbann::Optimizer::loadFromCheckpoint(int fd, const char* filename, uint64_t* bytes) {
  return true;
}

bool lbann::Optimizer::saveToCheckpointShared(const char* dir, int Index, uint64_t* bytes) {
  return true;
}

bool lbann::Optimizer::loadFromCheckpointShared(const char* dir, int Index, uint64_t* bytes) {
  return true;
}

lbann::Optimizer_factory::Optimizer_factory() {

}

lbann::Optimizer_factory::~Optimizer_factory() {

}
#endif
