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

#include "lbann/utils/distconv.hpp"
#include "lbann/utils/cudnn.hpp"
#include <cstdlib>

#ifdef LBANN_HAS_DISTCONV

using namespace distconv;

namespace lbann {
namespace dc {

////////////////////////////////////////////////////////////
// Global Distconv objects
////////////////////////////////////////////////////////////

namespace {
p2p::P2P *p2p_instance = nullptr;
Backend *backend_instance = nullptr;
} // namespace

void initialize(MPI_Comm comm) {
  if (!p2p_instance) {
    p2p_instance = new p2p::P2P(comm);
  }
  if (!backend_instance) {
    auto &cudnn_h = lbann::cudnn::get_handle();
    cudaStream_t s;
    CHECK_CUDNN(cudnnGetStream(cudnn_h, &s));
    backend_instance = new Backend(comm, cudnn_h, s);
  }
}

void finalize() {
  if (p2p_instance) {
    delete p2p_instance;
    p2p_instance = nullptr;
  }
  if (backend_instance) {
    delete backend_instance;
    backend_instance = nullptr;
  }
}

p2p::P2P &get_p2p() {
  return *p2p_instance;
}

Backend &get_backend() {
  return *backend_instance;
}

HaloExchangeMethod get_halo_exchange_method() {
  char *env = std::getenv("DISTCONV_HALO_EXCHANGE");
  if (!env) {
    // not specified
    return HaloExchangeMethod::MPI_DERIVED_TYPE;
  }
  std::string s(env);
  if (s == "P2P") {
    return HaloExchangeMethod::P2P;
  } else if (s == "MPI") {
    return HaloExchangeMethod::MPI;
  } else if (s == "MPI_DERIVED_TYPE") {
    return HaloExchangeMethod::MPI_DERIVED_TYPE;
  } else {
    LBANN_ERROR("Unknown value of environment variable DISTCONV_HALO_EXCHANGE");
  }
}

TensorShuffler *get_tensor_shuffler(const TensorDev &src,
                                    const TensorDev &dst) {
  char *env = std::getenv("DISTCONV_TENSOR_SHUFFLER");
  if (env && std::string(env) == "P2P") {
    return new TensorShufflerP2P(src, dst, get_p2p());
  } else {
    return new TensorShuffler(src, dst);
  }
}

} // namespace dc
} // namespace lbann

#endif // LBANN_HAS_DISTCONV
