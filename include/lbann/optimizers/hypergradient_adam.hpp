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

#ifndef LBANN_OPTIMIZERS_HYPERGRADIENT_ADAM_HPP_INCLUDED
#define LBANN_OPTIMIZERS_HYPERGRADIENT_ADAM_HPP_INCLUDED

#include "lbann/optimizers/data_type_optimizer.hpp"
#include "lbann/io/persist.hpp"
#include <optimizers.pb.h>

namespace lbann {

/** @class hypergradient_adam
 *  @brief Hypergradient Adam optimizer.
 *
 *  Reference:
 *
 *  Baydin et al. "Online Learning Rate Adaptation with Hypergradient
 *  Descent", 2017.
 */
template <typename TensorDataType>
class hypergradient_adam
  : public Cloneable<hypergradient_adam<TensorDataType>,
                     data_type_optimizer<TensorDataType>> {
  using BaseType = Cloneable<hypergradient_adam<TensorDataType>,
                             data_type_optimizer<TensorDataType>>;
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  /** @brief The concrete weights type used by this object. */
  using WeightsType = data_type_weights<TensorDataType>;

  /** @brief The base optimizer type for this class. */
  using OptimizerType = data_type_optimizer<TensorDataType>;

  ///@}

public:

  /** @brief Construct a Hypergradient Adam optimizer object
   *
   *  @param comm                   Communication context for this object
   *  @param init_learning_rate     Initial Adam learning rate (0.001 is
   *                                reasonable).
   *  @param hyper_learning_rate    Hypergradient learning rate.
   *  @param beta1                  Decay rate for the first moment
   *                                moving average.
   *  @param beta2                  Decay rate for the second moment
   *                                moving average.
   *  @param eps                    Small factor to avoid division by
   *                                zero.
   */
  hypergradient_adam(TensorDataType init_learning_rate = 1e-3,
                     TensorDataType hyper_learning_rate = 1e-7,
                     TensorDataType beta1 = 0.9,
                     TensorDataType beta2 = 0.99,
                     TensorDataType eps = 1e-8);
  hypergradient_adam(const hypergradient_adam& other);
  hypergradient_adam& operator=(const hypergradient_adam& other);
  ~hypergradient_adam() override = default;

    /** Archive for checkpoint and restart */
  template <class Archive> void serialize(Archive & ar) {
    ar(cereal::base_class<data_type_optimizer<TensorDataType>>(this),
       CEREAL_NVP(m_hyper_learning_rate),
       CEREAL_NVP(m_beta1),
       CEREAL_NVP(m_beta2),
       CEREAL_NVP(m_eps),
       CEREAL_NVP(m_current_beta1),
       CEREAL_NVP(m_current_beta2));
  }

  /** @brief Human-readable type name. */
  std::string get_type() const override { return "hypergradient Adam"; }
  /** @brief Human-readable description. */
  description get_description() const override;

  void setup(WeightsType* w = nullptr) override;

protected:

  /** @brief Computation for an optimization step. */
  void step_compute(AbsDistMatrixType& values, const AbsDistMatrixType& gradient) override;

private:

  /** @brief Hypergradient learning rate. */
  TensorDataType m_hyper_learning_rate;
  /** @brief Update factor for first moment estimate. */
  TensorDataType m_beta1;
  /** @brief Update factor for second moment estimate. */
  TensorDataType m_beta2;
  /** @brief Small factor to avoid division by zero. */
  TensorDataType m_eps;
  /** @brief beta1 ^ iteration. */
  TensorDataType m_current_beta1;
  /** @brief beta2 ^ iteration. */
  TensorDataType m_current_beta2;
  /** @brief First moment estimates. */
  std::unique_ptr<AbsDistMatrixType> m_moment1;
  /** @brief Second moment estimates. */
  std::unique_ptr<AbsDistMatrixType> m_moment2;
  /** @brief Gradient estimate from the prior step (for hypergradient). */
  std::unique_ptr<AbsDistMatrixType> m_old_gradient;

  // ===========================================
  // Checkpointing
  // ===========================================

  bool save_to_checkpoint_shared(persist& p, std::string m_name) override;
  bool load_from_checkpoint_shared(persist& p, std::string m_name) override;
  bool save_to_checkpoint_distributed(persist& p, std::string m_name) override;
  bool load_from_checkpoint_distributed(persist& p, std::string m_name) override;

};

template <typename TensorDataType>
std::unique_ptr<optimizer>
build_hypergradient_adam_optimizer_from_pbuf(
  google::protobuf::Message const&);

} // namespace lbann

#endif // LBANN_OPTIMIZER_HYPERGRADIENT_ADAM_HPP_INCLUDED
