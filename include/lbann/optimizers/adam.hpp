////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#include "lbann/io/persist.hpp"
#include "lbann/optimizers/data_type_optimizer.hpp"
#include "lbann/proto/optimizers.pb.h"

namespace lbann {
namespace callback {
class perturb_adam;
} // namespace callback

/** @brief Adam optimizer.
 *
 *  Reference:
 *
 *  Diederik P. Kingma and Jimmy Ba. "Adam: A method for stochastic
 *  optimization." arXiv preprint arXiv:1412.6980 (2014).
 */
template <typename TensorDataType>
class adam
  : public Cloneable<adam<TensorDataType>, data_type_optimizer<TensorDataType>>
{
  using BaseType =
    Cloneable<adam<TensorDataType>, data_type_optimizer<TensorDataType>>;

public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  /** @brief The optimizer base type of this object. */
  using OptimizerType = data_type_optimizer<TensorDataType>;

  /** @brief The concrete weights type used by this object. */
  using WeightsType = data_type_weights<TensorDataType>;

  ///@}

public:
  /** @name Life cycle functions */
  ///@{

  adam(TensorDataType learning_rate,
       TensorDataType beta1 = 0.9,
       TensorDataType beta2 = 0.99,
       TensorDataType eps = 1e-8,
       TensorDataType adamw_weight_decay = 0.0);
  adam(const adam& other);
  adam& operator=(const adam& other);
  ~adam() = default;

  /** Archive for checkpoint and restart */
  template <class Archive>
  void serialize(Archive& ar);

  ///@}

  /** @name Descriptions */
  ///@{

  /** Human-readable type name. */
  std::string get_type() const override { return "Adam"; }
  /** Human-readable description. */
  description get_description() const override;
  /** @brief Returns the optimizer state size in bytes. */
  size_t get_state_size() const override;
  ///@}

  /** @name Access functions */
  ///@{

  /** Update factor for first moment estimate. */
  TensorDataType get_beta1() const noexcept { return m_beta1; }
  /** Update factor for first moment estimate. */
  void set_beta1(TensorDataType beta1) { m_beta1 = beta1; }
  /** Update factor for second moment estimate. */
  TensorDataType get_beta2() const noexcept { return m_beta2; }
  /** Update factor for second moment estimate. */
  void set_beta2(TensorDataType beta2) { m_beta2 = beta2; }
  /** Small factor to avoid division by zero. */
  TensorDataType get_eps() const noexcept { return m_eps; }
  /** Small factor to avoid division by zero. */
  void set_eps(TensorDataType eps) { m_eps = eps; }
  /** Regularizer coefficient for AdamW weight decay. */
  TensorDataType get_adamw_weight_decay() const noexcept
  {
    return m_adamw_weight_decay;
  }
  /** Regularizer coefficient for AdamW weight decay. */
  void set_adamw_weight_decay(TensorDataType adamw_weight_decay)
  {
    m_adamw_weight_decay = adamw_weight_decay;
  }

  /** First moment estimates. */
  const AbsDistMatrixType& get_moment1() const;
  /** First moment estimates. */
  AbsDistMatrixType& get_moment1();
  /** Second moment estimates. */
  const AbsDistMatrixType& get_moment2() const;
  /** Second moment estimates. */
  AbsDistMatrixType& get_moment2();

  /** beta1 ^ iteration.
   *  @todo This probably shouldn't be exposed.
   */
  TensorDataType get_current_beta1() const noexcept { return m_current_beta1; }
  /** beta1 ^ iteration.
   *  @todo This probably shouldn't be exposed.
   */
  void set_current_beta1(TensorDataType current_beta1)
  {
    m_current_beta1 = current_beta1;
  }
  /** beta2 ^ iteration.
   *  @todo This probably shouldn't be exposed.
   */
  TensorDataType get_current_beta2() const noexcept { return m_current_beta2; }
  /** beta2 ^ iteration.
   *  @todo This probably shouldn't be exposed.
   */
  void set_current_beta2(TensorDataType current_beta2)
  {
    m_current_beta2 = current_beta2;
  }

  ///@}

  /** @name Setup */
  ///@{

  using OptimizerType::setup;
  void setup(WeightsType* w = nullptr) override;

  ///@}

  /** Add optimizer data to prototext */
  void write_proto(lbann_data::Optimizer& opt) const final;

protected:
  friend cereal::access;

  /** @brief Default constructor.
   *  @details This constructor exists as an implementation detail of
   *  the serialization code. It is not for general use.
   */
  adam()
    : adam(El::To<TensorDataType>(1.f),
           El::To<TensorDataType>(0.9),
           El::To<TensorDataType>(0.99),
           El::To<TensorDataType>(1e-8),
           El::To<TensorDataType>(0))
  {}

  /** Computation for an optimization step. */
  void step_compute(AbsDistMatrixType& values,
                    const AbsDistMatrixType& gradient) override;

private:
  /** Update factor for first moment estimate. */
  TensorDataType m_beta1;
  /** Update factor for second moment estimate. */
  TensorDataType m_beta2;
  /** Small factor to avoid division by zero. */
  TensorDataType m_eps;
  /** Regularizer coefficient for AdamW weight decay. */
  TensorDataType m_adamw_weight_decay;
  /** beta1 ^ iteration. */
  TensorDataType m_current_beta1 = TensorDataType(1.);
  /** beta2 ^ iteration. */
  TensorDataType m_current_beta2 = TensorDataType(1.);
  /** First moment estimates. */
  std::unique_ptr<AbsDistMatrixType> m_moment1;
  /** Second moment estimates. */
  std::unique_ptr<AbsDistMatrixType> m_moment2;

  /** Hyperparameter exploration. */
  friend class callback::perturb_adam;

  /** CPU implementation of optimization step. */
  void step_compute_cpu(AbsDistMatrixType& values,
                        const AbsDistMatrixType& gradient,
                        const TensorDataType& correction);
#ifdef LBANN_HAS_GPU
  /** GPU implementation of optimization step. */
  void step_compute_gpu(AbsDistMatrixType& values,
                        const AbsDistMatrixType& gradient,
                        const TensorDataType& correction);
#endif // LBANN_HAS_GPU
};

template <typename TensorDataType>
std::unique_ptr<optimizer>
build_adam_optimizer_from_pbuf(google::protobuf::Message const&);

} // namespace lbann

#endif // LBANN_OPTIMIZERS_ADAM_HPP_INCLUDED
