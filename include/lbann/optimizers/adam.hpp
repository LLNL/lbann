////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_OPTIMIZERS_ADAM_HPP_INCLUDED
#define LBANN_OPTIMIZERS_ADAM_HPP_INCLUDED

#include "lbann/optimizers/optimizer.hpp"

namespace lbann {

/** @brief Adam optimizer.
 *
 *  Reference:
 *
 *  Diederik P. Kingma and Jimmy Ba. "Adam: A method for stochastic
 *  optimization." arXiv preprint arXiv:1412.6980 (2014).
 */
class adam : public optimizer {
public:

  /** @name Life cycle functions */
  ///@{

  adam(lbann_comm* comm,
       DataType learning_rate,
       DataType beta1 = 0.9,
       DataType beta2 = 0.99,
       DataType eps = 1e-8);
  adam(const adam& other);
  adam& operator=(const adam& other);
  ~adam() = default;
  adam* copy() const override { return new adam(*this); }

  ///@}

  /** @name Descriptions */
  ///@{

  /** Human-readable type name. */
  std::string get_type() const override { return "Adam"; }
  /** Human-readable description. */
  description get_description() const override;

  ///@}

  /** @name Access functions */
  ///@{

  /** Update factor for first moment estimate. */
  DataType get_beta1() const noexcept { return m_beta1; }
  /** Update factor for first moment estimate. */
  void set_beta1(DataType beta1) { m_beta1 = beta1; }
  /** Update factor for second moment estimate. */
  DataType get_beta2() const noexcept { return m_beta2; }
  /** Update factor for second moment estimate. */
  void set_beta2(DataType beta2) { m_beta2 = beta2; }
  /** Small factor to avoid division by zero. */
  DataType get_eps() const noexcept { return m_eps; }
  /** Small factor to avoid division by zero. */
  void set_eps(DataType eps) { m_eps = eps; }

  /** First moment estimates. */
  const AbsDistMat& get_moment1() const;
  /** First moment estimates. */
  AbsDistMat& get_moment1();
  /** Second moment estimates. */
  const AbsDistMat& get_moment2() const;
  /** Second moment estimates. */
  AbsDistMat& get_moment2();

  /** beta1 ^ iteration.
   *  @todo This probably shouldn't be exposed.
   */
  DataType get_current_beta1() const noexcept { return m_current_beta1; }
  /** beta1 ^ iteration.
   *  @todo This probably shouldn't be exposed.
   */
  void set_current_beta1(DataType current_beta1) { m_current_beta1 = current_beta1; }
  /** beta2 ^ iteration.
   *  @todo This probably shouldn't be exposed.
   */
  DataType get_current_beta2() const noexcept { return m_current_beta2; }
  /** beta2 ^ iteration.
   *  @todo This probably shouldn't be exposed.
   */
  void set_current_beta2(DataType current_beta2) { m_current_beta2 = current_beta2; }

  ///@}

  /** @name Setup */
  ///@{

  void setup(weights* w = nullptr) override;

  ///@}

protected:

  /** Computation for an optimization step. */
  void step_compute(AbsDistMat& values,
                    const AbsDistMat& gradient) override;

private:

  /** Update factor for first moment estimate. */
  DataType m_beta1;
  /** Update factor for second moment estimate. */
  DataType m_beta2;
  /** Small factor to avoid division by zero. */
  DataType m_eps;
  /** beta1 ^ iteration. */
  DataType m_current_beta1 = 1;
  /** beta2 ^ iteration. */
  DataType m_current_beta2 = 1;
  /** First moment estimates. */
  std::unique_ptr<AbsDistMat> m_moment1;
  /** Second moment estimates. */
  std::unique_ptr<AbsDistMat> m_moment2;

  /** Hyperparameter exploration. */
  friend class lbann_callback_perturb_adam;

  /** CPU implementation of optimization step. */
  void step_compute_cpu(AbsDistMat& values, const AbsDistMat& gradient);
#ifdef LBANN_HAS_CUDA
  /** GPU implementation of optimization step. */
  void step_compute_gpu(AbsDistMat& values, const AbsDistMat& gradient);
#endif // LBANN_HAS_CUDA

  /** @name Checkpointing */
  ///@{

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

  ///@}

};

} // namespace lbann

#endif // LBANN_OPTIMIZERS_ADAM_HPP_INCLUDED
