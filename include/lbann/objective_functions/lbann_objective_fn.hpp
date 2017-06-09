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

#ifndef LBANN_OBJECTIVE_FN_HPP_INCLUDED
#define LBANN_OBJECTIVE_FN_HPP_INCLUDED

#include "lbann/lbann_base.hpp"
#include "lbann/lbann_comm.hpp"
#include "lbann/utils/lbann_exception.hpp"
#include "lbann/layers/lbann_layer.hpp"

namespace lbann
{
  namespace objective_functions
  {
    enum class obj_fn_type {categorical_cross_entropy, mean_squared_error, INVALID};

    static const char* __attribute__((used)) _to_string(obj_fn_type o) {
      switch(o) {
      case obj_fn_type::categorical_cross_entropy:
        return "categorical cross entropy";
      case obj_fn_type::mean_squared_error:
        return "mean squared error";
      default:
        throw lbann_exception("Invalid obj_fn type specified");
      }
      return NULL;
    }

    class statistics
    {
    public:
      statistics() {
        init_stats();
      }

      ~statistics() {}

      void init_stats() {
        m_last_mini_batch_avg_cost = 0.0;
        m_aggregate_avg_cost_per_epoch = 0.0;
        m_num_mini_batch_per_epoch = 0;
      }

      void reset_stats() {
        m_last_mini_batch_avg_cost = 0.0;
        m_aggregate_avg_cost_per_epoch = 0.0;
        m_num_mini_batch_per_epoch = 0;
      }

      /// Error is accumulated as a double -- this works for both sum of
      /// squared errors and categorical errors
      double m_last_mini_batch_avg_cost;
      double m_aggregate_avg_cost_per_epoch;
      long m_num_mini_batch_per_epoch;
    };

    /**
     * Objective functions / loss functions are computed and averaged on a batch by batch basis.
     * Additionally, the average loss per batch is averaged over the epoch.
     *

     *eturns
     A `History` object. Its `History.history` attribute is
     a record of training loss values and metrics values
     at successive epochs, as well as validation loss values
     and validation metrics values (if applicable).
     *
     */
    class objective_fn {
    public:
      objective_fn() {
        m_training_stats.init_stats();
        m_validation_stats.init_stats();
        m_testing_stats.init_stats();
        this->type = obj_fn_type::INVALID;
      }
      objective_fn(std::string name): m_name(name) {
        objective_fn();
      }
      virtual ~objective_fn() {}
      virtual void setup(int num_neurons, int mini_batch_size) {}
      virtual void fp_set_std_matrix_view(int64_t cur_mini_batch_size) {}
      /// Compute the object function -- Note that it is averaged across a mini-batch
      virtual double compute_obj_fn(ElMat &predictions_v, ElMat &groundtruth_v) {return 0.0;}
      virtual void compute_obj_fn_derivative(layer_type prev_layer_type,
                                             ElMat &predictions_v,
                                             ElMat &groundtruth_v,
                                             ElMat &error_signal_v) {}

      statistics* get_statistics(execution_mode mode);
      double report_obj_fn(execution_mode mode);
      double report_aggregate_avg_obj_fn(execution_mode mode);
      void record_obj_fn(execution_mode mode, double avg_cost);
      void reset_obj_fn();
      const std::string & name() { return m_name; }

      statistics m_training_stats;
      statistics m_validation_stats;
      statistics m_testing_stats;

      obj_fn_type type;

    private:
      std::string m_name;
    };
  }
}

#endif // LBANN_OBJECTIVE_FN_INCLUDED
