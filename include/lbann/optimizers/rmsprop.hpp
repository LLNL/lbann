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

#ifndef LBANN_OPTIMIZER_RMSPROP_HPP
#define LBANN_OPTIMIZER_RMSPROP_HPP

#include "lbann/optimizers/optimizer.hpp"
#include <sys/stat.h>

namespace lbann {

/** RMSprop optimizer. */
class rmsprop : public optimizer {
 public:

  /** Constructor. */
  rmsprop(lbann_comm *comm,
          DataType learning_rate,
          DataType decay_rate,
          DataType eps = DataType(1e-8));

  /** Copy constructor. */
  rmsprop(const rmsprop& other);
  /** Copy assignment operator. */
  rmsprop& operator=(const rmsprop& other);
  /** Destructor. */
  ~rmsprop() override;
  /** Create a copy. */
  rmsprop* copy() const override { return new rmsprop(*this); }
  
  /** Get the optimizer name. */
  std::string get_type() const override { return "rmsprop"; }
  /** Get a human-readable description of the optimizer. */
  std::string get_description() const override;

  /** Setup optimizer. */
  void setup(weights& w) override;

  /** Perform the computation in an optimization step. */
  void step_compute(AbsDistMat& values, const AbsDistMat& gradient) override;

 private:

  /** Decay rate. */
  DataType m_decay_rate;
  /** Small factor to avoid division by zero. */
  DataType m_eps;
  /** RMSprop cache. */
  AbsDistMat *m_cache;


//************************************************************************
// Checkpointing
//************************************************************************

  struct packing_header {
    DataType decay_rate;
  };

  bool pack_scalars(persist& p) {
    p.write_datatype(persist_type::train, "decay_rate", m_decay_rate);
    return true;
  }

  bool unpack_scalars(persist& p, struct packing_header *header){
    p.read_datatype(persist_type::train, "momentum",  &m_decay_rate);
    
    if(header != nullptr){
      header->decay_rate = m_decay_rate;
    }
   
  return true;
  }
  
  void unpack_header(struct packing_header& header){
    m_decay_rate = header.decay_rate;
  }
  
  bool save_to_checkpoint_shared(persist& p, std::string m_name) override;
  bool load_from_checkpoint_shared(persist& p, std::string m_name) override;


};

} // namespace lbann

#endif // LBANN_OPTIMIZER_RMSPROP_HPP
