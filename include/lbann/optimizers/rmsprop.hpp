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

#include "lbann/optimizers/optimizer.hpp"
#include <sys/stat.h>

namespace lbann {

/** RMSprop optimizer.
 *
 *  See
 *  https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf.
 */
template <typename TensorDataType>
class rmsprop : public data_type_optimizer<TensorDataType> {
public:

  rmsprop(TensorDataType learning_rate,
          TensorDataType decay_rate,
          TensorDataType eps = 1e-8);
  rmsprop(const rmsprop& other);
  rmsprop& operator=(const rmsprop& other);
  ~rmsprop() override = default;
  rmsprop* copy() const override { return new rmsprop(*this); }

  /** Human-readable type name. */
  std::string get_type() const override { return "RMSprop"; }
  /** Human-readable description. */
  description get_description() const override;

  void setup(data_type_weights<TensorDataType>* w = nullptr) override;

protected:

  /** Computation for an optimization step. */
  void step_compute(El::AbstractDistMatrix<TensorDataType>& values,
                    const El::AbstractDistMatrix<TensorDataType>& gradient) override;

private:

  /** Decay rate. */
  TensorDataType m_decay_rate;
  /** Small factor to avoid division by zero. */
  TensorDataType m_eps;
  /** RMSprop cache. */
  std::unique_ptr<El::AbstractDistMatrix<TensorDataType>> m_cache;

  /** CPU implementation of optimization step. */
  void step_compute_cpu(El::AbstractDistMatrix<TensorDataType>& values, const El::AbstractDistMatrix<TensorDataType>& gradient);
#ifdef LBANN_HAS_CUDA
  /** GPU implementation of optimization step. */
  void step_compute_gpu(El::AbstractDistMatrix<TensorDataType>& values, const El::AbstractDistMatrix<TensorDataType>& gradient);
#endif // LBANN_HAS_CUDA

  // ===========================================
  // Checkpointing
  // ===========================================

  struct packing_header {
    TensorDataType decay_rate;
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
  bool save_to_checkpoint_distributed(persist& p, std::string m_name) override;
  bool load_from_checkpoint_distributed(persist& p, std::string m_name) override;

};

std::unique_ptr<data_type_optimizer<DataType>>
build_rmsprop_optimizer_from_pbuf(
  google::protobuf::Message const&);

} // namespace lbann

#endif // LBANN_OPTIMIZERS_RMSPROP_HPP_INCLUDED
