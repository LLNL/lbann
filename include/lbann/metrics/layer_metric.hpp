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

#ifndef LBANN_METRIC_LAYER_METRIC_HPP
#define LBANN_METRIC_LAYER_METRIC_HPP

#include "lbann/metrics/metric.hpp"
#include "lbann/layers/transform/evaluation.hpp"

namespace lbann {

class layer_metric : public metric {

 public:
  layer_metric(lbann_comm *comm,
               std::string name = "",
               std::string unit = "");
  layer_metric(const layer_metric& other) = default;
  layer_metric& operator=(const layer_metric& other) = default;
  virtual ~layer_metric() = default;
  layer_metric* copy() const override { return new layer_metric(*this); }

  /** Return a string name for this metric. */
  std::string name() const override;
  std::string get_unit() const override { return m_unit; }

  /** Set corresponding layer. */
  void set_layer(Layer& l);
  /** Get corresponding layer. */
  Layer& get_layer();
  /** Get corresponding layer (const). */
  const Layer& get_layer() const;

  /** Get list of pointers to layers. */
  std::vector<Layer*> get_layer_pointers() const override;
  /** Set list of pointers to layers. */
  void set_layer_pointers(std::vector<Layer*> layers) override;

 protected:

  void setup(model& m) override;
  EvalType evaluate(execution_mode mode, int mini_batch_size) override;

  /** Computation to evaluate the metric function (deprecated).
   *  This function is not called since the 'evaluate' function is
   *  overridden.
   */
  EvalType evaluate_compute(const AbsDistMat& prediction,
                            const AbsDistMat& ground_truth) override {
    LBANN_ERROR("This function should not be called");
    return EvalType(0);
  }

 private:

  /** Descriptive name for metric. */
  std::string m_name;
  /** Metric unit.
   *  If the unit is "%", the reported value is multiplied by 100.
   */
  std::string m_unit;
  /** Corresponding layer. */
  Layer* m_layer;

  /** Get corresponding evaluation layer. */
  abstract_evaluation_layer& get_evaluation_layer();

};

}  // namespace lbann

#endif  // LBANN_METRIC_LAYER_METRIC_HPP
