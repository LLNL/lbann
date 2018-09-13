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

#ifndef LBANN_OPTIMIZER_ADAM_HPP
#define LBANN_OPTIMIZER_ADAM_HPP

#include "lbann/optimizers/optimizer.hpp"

namespace lbann {

/** Adam optimizer.
 *  Reference:
 *  Kingma, D. and Ba, J. 2014. Adam: A Method for Stochastic Optimization.
 */
class adam : public optimizer {
 public:

  /** Constructor. */
  adam(lbann_comm *comm,
       DataType learning_rate,
       DataType beta1 = DataType(0.9),
       DataType beta2 = DataType(0.99),
       DataType eps = DataType(1e-8));


  /** Copy constructor. */
  adam(const adam& other);
  /** Copy assignment operator. */
  adam& operator=(const adam& other);
  /** Destructor. */
  ~adam() override;
  /** Create a copy. */
  adam* copy() const override { return new adam(*this); }

  /** Returns the optimizer name. */
  std::string get_type() const override { return "Adam"; }
  /** Get a human-readable description of the optimizer. */
  std::string get_description() const override;

  /** Setup optimizer. */
  void setup(weights& w) override;

  /** Perform the computation in an optimization step. */
  void step_compute(AbsDistMat& values, const AbsDistMat& gradient) override;
#ifdef LBANN_HAS_CUDNN
  /** Perform the computation in an optimization step on GPU. */
  void step_compute_gpu(AbsDistMat& values, const AbsDistMat& gradient) override;
#endif // LBANN_HAS_CUDNN

 private:

  /** Update factor for first moment estimate. */
  DataType m_beta1;
  /** Update factor for second moment estimate. */
  DataType m_beta2;
  /** Small factor to avoid division by zero. */
  DataType m_eps;
  /** beta1 ^ iteration. */
  DataType m_current_beta1;
  /** beta2 ^ iteration. */
  DataType m_current_beta2;
  /** First moment estimates. */
  AbsDistMat *m_moment1;
  /** Second moment estimates. */
  AbsDistMat *m_moment2;

//************************************************************************
// Checkpointing
//************************************************************************
 private:
  /* struct used to serialize mode fields in file and MPI transfer */
  struct packing_header {
    DataType beta1;
    DataType beta2;
    DataType eps;
    DataType current_beta1;
    DataType current_beta2;
  };

  bool pack_scalars(persist& p) {
    p.write_datatype(persist_type::train, "beta1", m_beta1);
    p.write_datatype(persist_type::train, "beta2", m_beta2);
    p.write_datatype(persist_type::train, "eps",   m_eps);
    p.write_datatype(persist_type::train, "current_beta1", m_current_beta1);
    p.write_datatype(persist_type::train, "current_beta2", m_current_beta2);
    return true;
  }

  bool unpack_scalars(persist& p, struct packing_header *header) {
    p.read_datatype(persist_type::train, "beta1", &m_beta1);
    p.read_datatype(persist_type::train, "beta2", &m_beta2);
    p.read_datatype(persist_type::train, "eps",   &m_eps);
    p.read_datatype(persist_type::train, "current_beta1", &m_current_beta1);
    p.read_datatype(persist_type::train, "current_beta2", &m_current_beta2);

    if(header != nullptr) {
      header->beta1 = m_beta1;
      header->beta2 = m_beta2;
      header->eps = m_eps;
      header->current_beta1 = m_current_beta1;
      header->current_beta2 = m_current_beta2;
    }
    return true;
  }

  void unpack_header(struct packing_header& header) {
    m_beta1 = header.beta1;
    m_beta2 = header.beta2;
    m_eps = header.eps;
    m_current_beta1 = header.current_beta1;
    m_current_beta2 = header.current_beta2;
  }

  bool save_to_checkpoint_shared(persist& p, std::string m_name) override;
  bool load_from_checkpoint_shared(persist& p, std::string m_name) override;
  bool save_to_checkpoint_distributed(persist& p, std::string m_name) override;
  bool load_from_checkpoint_distributed(persist& p, std::string m_name) override;
};

} // namespace lbann

#endif  // LBANN_OPTIMIZER_ADAM_HPP
