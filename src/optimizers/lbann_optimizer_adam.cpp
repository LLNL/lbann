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
// lbann_optimizer_adam .hpp .cpp - SGD with Adam
////////////////////////////////////////////////////////////////////////////////

#include "lbann/optimizers/lbann_optimizer_adam.hpp"

namespace lbann {

Adam_factory::Adam_factory(lbann_comm* comm, float lr, float rho1, float rho2,
                           float eps) :
  comm(comm), lr(lr), rho1(rho1), rho2(rho2), eps(eps) {}

Adam_factory::~Adam_factory() {}

Optimizer* Adam_factory::create_optimizer(matrix_format format) {
  switch(format) {
  case matrix_format::MC_MR:
    return new Adam<DistMat>(comm, lr, rho1, rho2, eps);
  case matrix_format::CIRC_CIRC:
    return new Adam<CircMat>(comm, lr, rho1, rho2, eps);
  case matrix_format::STAR_STAR:
    return new Adam<StarMat>(comm, lr, rho1, rho2, eps);
  case matrix_format::STAR_VC:
    return new Adam<StarVCMat>(comm, lr, rho1, rho2, eps);
  default:
    // TODO: throw an exception
    printf("LBANN Error: unknown matrix distribution for Adam optimizer\n");
    exit(-1);
  }
}

}  // namespace lbann
