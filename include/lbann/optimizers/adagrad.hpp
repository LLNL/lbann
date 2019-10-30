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

#ifndef LBANN_OPTIMIZERS_ADAGRAD_HPP_INCLUDED
#define LBANN_OPTIMIZERS_ADAGRAD_HPP_INCLUDED

#include "lbann/optimizers/optimizer.hpp"

namespace lbann {

/** AdaGrad optimizer.
 *
 *  Reference:
 *
 *  John Duchi, Elad Hazan, and Yoram Singer. "Adaptive subgradient
 *  methods for online learning and stochastic optimization." Journal
 *  of Machine Learning Research 12, no. Jul (2011): 2121-2159.
 */
template <typename TensorDataType>
class adagrad : public optimizer<TensorDataType> {
public:

  adagrad(TensorDataType learning_rate, TensorDataType eps = 1e-8);
  adagrad(const adagrad& other);
  adagrad& operator=(const adagrad& other);
  ~adagrad() override = default;
  adagrad* copy() const override { return new adagrad(*this); }

  /** Human-readable type name. */
  std::string get_type() const override { return "AdaGrad"; }
  /** Human-readable description. */
  description get_description() const override;

  void setup(weights<TensorDataType>* w = nullptr) override;

protected:

  /** Computation for an optimization step. */
  void step_compute(El::AbstractDistMatrix<TensorDataType>& values, const El::AbstractDistMatrix<TensorDataType>& gradient) override;

private:

  /** Small factor to avoid division by zero. */
  TensorDataType m_eps;
  /** AdaGrad cache. */
  std::unique_ptr<El::AbstractDistMatrix<TensorDataType>> m_cache;

  /** CPU implementation of optimization step. */
  void step_compute_cpu(El::AbstractDistMatrix<TensorDataType>& values, const El::AbstractDistMatrix<TensorDataType>& gradient);
#ifdef LBANN_HAS_CUDNN
  /** GPU implementation of optimization step. */
  void step_compute_gpu(El::AbstractDistMatrix<TensorDataType>& values, const El::AbstractDistMatrix<TensorDataType>& gradient);
#endif // LBANN_HAS_CUDNN

  // ===========================================
  // Checkpointing
  // ===========================================

  bool save_to_checkpoint_shared(persist& p, std::string m_name) override;
  bool load_from_checkpoint_shared(persist& p, std::string m_name) override;
  bool save_to_checkpoint_distributed(persist& p, std::string m_name) override;
  bool load_from_checkpoint_distributed(persist& p, std::string m_name) override;

};

std::unique_ptr<optimizer<DataType>>
build_adagrad_optimizer_from_pbuf(
  google::protobuf::Message const&);

} // namespace lbann

#endif // LBANN_OPTIMIZERS_ADAGRAD_HPP_INCLUDED
