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

#ifndef LBANN_LAYER_EVALUATION_HPP_INCLUDED
#define LBANN_LAYER_EVALUATION_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"

namespace lbann {

/** Evaluation layer.
 *  Computes the average value across a mini-batch. If the input
 *  tensor has multiple neurons, their values are added together.
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class evaluation_layer : public transform_layer {

 public:

  evaluation_layer(lbann_comm *comm)
    : transform_layer(comm), m_scale(0), m_value(0) {
    static_assert(Dev == El::Device::CPU,
                  "evaluation layer currently only supports CPU");

    // Evaluation layer has no children
    m_expected_num_child_layers = 0;

  }

  evaluation_layer* copy() const override { return new evaluation_layer(*this); }
  std::string get_type() const override { return "evaluation"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  /** Returns description. */
  std::string get_description() const override {
    std::stringstream s;
     s << "evaluation_layer  dataLayout: " << this->get_data_layout_string(get_data_layout());
     return s.str();
  }

  /** Get scaling factor. */
  EvalType get_scale() const { return m_scale; }
  /** Set scaling factor. */
  void set_scale(EvalType scale) { m_scale = scale; }

  /** Get evaluated value. */
  EvalType get_value(bool unscaled = false) {
    this->m_comm->wait(m_allreduce_req);
    if (unscaled) {
      return m_value;
    } else {
      return m_scale * m_value;
    }
  }

 protected:

  virtual void fp_compute() override {
    const auto& input = get_prev_activations();
    const auto& local_input = input.LockedMatrix();
    const El::Int local_height = local_input.Height();
    const El::Int local_width = local_input.Width();
    const auto& mini_batch_size = input.Width();

    // Compute average value
    EvalType sum = EvalType(0);
    int nthreads = omp_get_num_threads();
    std::vector<EvalType> local_sum(nthreads, EvalType(0));
#pragma omp taskloop collapse(2) default(shared)
    for (El::Int col = 0; col < local_width; ++col) {
      for (El::Int row = 0; row < local_height; ++row) {
        const int tid = omp_get_thread_num();
        local_sum[tid] += local_input(row, col);
      }
    }
    for (int i = 0; i < nthreads; ++i) {
      sum += local_sum[i];
    }
    m_value = sum / mini_batch_size;
    this->m_comm->nb_allreduce(&m_value, 1, input.DistComm(), m_allreduce_req);

  }

  virtual void bp_compute() override {
    auto& error_signal = get_error_signals();
    if (m_scale == EvalType(0)) {
      El::Zero(error_signal);
    } else {
      El::Fill(error_signal, DataType(m_scale));
    }
  }

 private:
  /** Scaling factor to apply to evaluated value. */
  EvalType m_scale;
  /** Evaluated value. */
  EvalType m_value;
  /** Non-blocking allreduce request. */
  Al::request m_allreduce_req;

};

} // namespace lbann

#endif // LBANN_LAYER_EVALUATION_HPP_INCLUDED
