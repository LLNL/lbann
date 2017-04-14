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
// lbann_optimizer_rmsprop .hpp .cpp - SGD with RMSprop
////////////////////////////////////////////////////////////////////////////////

#include "lbann/optimizers/lbann_optimizer_rmsprop.hpp"
#include "lbann/utils/lbann_exception.hpp"

using namespace std;
using namespace El;

lbann::rmsprop::rmsprop
(lbann_comm* comm,
 DataType learning_rate,
 DataType decay_rate,
 DataType eps)
  : optimizer(comm, "rmsprop", learning_rate),
    m_decay_rate(decay_rate),
    m_eps(eps) {}

lbann::rmsprop::~rmsprop()
{
  if(m_cache)
    delete m_cache;
}

void lbann::rmsprop::setup(AbsDistMat* parameters)
{
  optimizer::setup(parameters);

  // Initialize RMSprop cache
  switch(m_matrix_format) {
  case matrix_format::MC_MR:
    m_cache = new DistMat(comm->get_model_grid()); break;
  case matrix_format::STAR_STAR:
    m_cache = new StarMat(comm->get_model_grid()); break;
  default:
    throw lbann_exception("lbann_optimizer_rmsprop: invalid data layout");
  }
  Zeros(*m_cache, m_height, m_width);

}

void lbann::rmsprop::update(const AbsDistMat* gradient)
{
  
  // Get local matrix data
  const Int local_height = m_parameters->LocalHeight();
  const Int local_width = m_parameters->LocalWidth();
  DataType* parameter_buffer = m_parameters->Buffer();
  const DataType* gradient_buffer = gradient->LockedBuffer();
  DataType* cache_buffer = m_cache->Buffer();

  // Update parameters
  // Note: we assume data is contiguous
#pragma omp parallel for
  for(Int i=0; i<local_height*local_width; ++i) {
    const DataType g = gradient_buffer[i];
    cache_buffer[i] = ( m_decay_rate * cache_buffer[i] 
                        + (DataType(1) - m_decay_rate) * g * g );
    parameter_buffer[i] -= ( m_learning_rate * g
                             / (Sqrt(cache_buffer[i]) + m_eps) );
  }

}

#if 0
    bool saveToCheckpoint(int fd, const char* filename, uint64_t* bytes) {
      //    writeDist(fd, filename, WB_D_Cache, bytes);
      return true;
    }

    bool loadFromCheckpoint(int fd, const char* filename, uint64_t* bytes) {
      //    readDist(fd, filename, WB_D_Cache, bytes);
      return true;
    }

    bool saveToCheckpointShared(persist& p, int Index) {
      char name[512];

      // current learning rate value
      if (p.m_rank == 0) {
        sprintf(name, "L%d_learning_rate", Index);
        p.write_float(persist_type::train, name, LearnRate);
      }

      // build name of the checkpoint file
      sprintf(name, "L%d_rmsprop_%dx%d", Index, WB_D_Cache.Height(), WB_D_Cache.Width());
      p.write_distmat(persist_type::train, name, (DistMat*)&WB_D_Cache);

      return true;
    }

    bool loadFromCheckpointShared(persist& p, int Index) {
      char name[512];

      // current learning rate value
      if (p.m_rank == 0) {
        sprintf(name, "L%d_learning_rate", Index);
        p.read_float(persist_type::train, name, &LearnRate);
      }
      MPI_Bcast(&LearnRate, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

      // read in the cache of gradients for WB
      sprintf(name, "L%d_rmsprop_%dx%d.bin", Index, WB_D_Cache.Height(), WB_D_Cache.Width());
      p.read_distmat(persist_type::train, name, (DistMat*)&WB_D_Cache);

      return true;
    }
#endif

lbann::rmsprop_factory::rmsprop_factory
(lbann_comm* comm,
 DataType learning_rate,
 DataType decay_rate,
 DataType eps)
  : optimizer_factory(comm, "rmsprop"),
    m_learning_rate(learning_rate),
    m_decay_rate(decay_rate),
    m_eps(eps) {}

lbann::rmsprop_factory::~rmsprop_factory() {}

lbann::optimizer* lbann::rmsprop_factory::create_optimizer()
{
  return new rmsprop(comm, m_learning_rate, m_decay_rate, m_eps);
}
