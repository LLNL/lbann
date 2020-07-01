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

#ifndef LBANN_OPTIMIZERS_RMSPROP_HPP_INCLUDED
#define LBANN_OPTIMIZERS_RMSPROP_HPP_INCLUDED

#include "lbann/optimizers/data_type_optimizer.hpp"
#include <sys/stat.h>
#include "lbann/io/persist.hpp"
#include <optimizers.pb.h>

namespace lbann {

/** RMSprop optimizer.
 *
 *  See
 *  https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf.
 */
template <typename TensorDataType>
class rmsprop : public Cloneable<rmsprop<TensorDataType>,
                                 data_type_optimizer<TensorDataType>> {
  using BaseType = Cloneable<rmsprop<TensorDataType>,
                             data_type_optimizer<TensorDataType>>;
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

  rmsprop(TensorDataType learning_rate,
          TensorDataType decay_rate,
          TensorDataType eps = 1e-8);
  rmsprop(const rmsprop& other);
  rmsprop& operator=(const rmsprop& other);
  ~rmsprop() override = default;

  /** Archive for checkpoint and restart */
  template <class Archive> void serialize(Archive & ar) {
    ar(cereal::base_class<data_type_optimizer<TensorDataType>>(this),
       CEREAL_NVP(m_decay_rate));
  }

  /** Human-readable type name. */
  std::string get_type() const override { return "RMSprop"; }
  /** Human-readable description. */
  description get_description() const override;

  void setup(WeightsType* w = nullptr) override;

protected:

  /** Computation for an optimization step. */
  void step_compute(AbsDistMatrixType& values,
                    const AbsDistMatrixType& gradient) override;

private:

  /** Decay rate. */
  TensorDataType m_decay_rate;
  /** Small factor to avoid division by zero. */
  TensorDataType m_eps;
  /** RMSprop cache. */
  std::unique_ptr<AbsDistMatrixType> m_cache;

  /** CPU implementation of optimization step. */
  void step_compute_cpu(AbsDistMatrixType& values, const AbsDistMatrixType& gradient);
#ifdef LBANN_HAS_CUDA
  /** GPU implementation of optimization step. */
  void step_compute_gpu(AbsDistMatrixType& values, const AbsDistMatrixType& gradient);
#endif // LBANN_HAS_CUDA

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
build_rmsprop_optimizer_from_pbuf(
  google::protobuf::Message const&);

} // namespace lbann

#endif // LBANN_OPTIMIZERS_RMSPROP_HPP_INCLUDED
