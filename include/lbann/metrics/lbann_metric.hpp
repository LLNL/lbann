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

#ifndef LBANN_METRIC_HPP
#define LBANN_METRIC_HPP

#include "lbann/lbann_base.hpp"
#include "lbann/lbann_comm.hpp"
#include "lbann/utils/lbann_exception.hpp"

namespace lbann
{
  /**
   * Usage of metrics - NOTE FROM KERAS DOCUMENTATION
   * A metric is a function that is used to judge the performance of your
   * model. Metric functions are to be supplied in the metrics parameter
   * when a model is compiled.

   * A metric function is similar to an objective function, except that the
   * results from evaluating a metric are not used when training the model.
   */

  // Forward-declare this.
  class model;

  enum class metric_type {binary_accuracy, categorical_accuracy, sparse_categorical_accuracy, top_k_categorical_accuracy,
      mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_logarithmic_error,
      hinge, squared_hinge, categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy, kullback_leibler_divergence,
      poisson, cosine_proximity, matthews_correlation, precision, recall, fbeta_score, fmeasure, INVALID};

  static const char* __attribute__((used)) _to_string(metric_type m) {
    switch(m) {
    case metric_type::categorical_accuracy:
      return "categorical_accuracy";
    case metric_type::mean_squared_error:
      return "mean_squared_error";
    default:
      throw lbann_exception("Invalid metric type specified");
    }
    return NULL;
  }


  class statistics
  {
  public:
    statistics() {}
    ~statistics() {}

    void init_stats() {
      m_error_per_epoch = 0;
      m_samples_per_epoch = 0;

      m_total_error = 0;
      m_total_num_samples = 0;
    }

    void reset_stats() {
      m_total_error += m_error_per_epoch;
      m_total_num_samples += m_samples_per_epoch;
      m_error_per_epoch = 0;
      m_samples_per_epoch = 0;
    }

    /// Error is accumulated as a double -- this works for both sum of
    /// squared errors and categorical errors
    double m_error_per_epoch;
    long m_samples_per_epoch;

    double m_total_error;
    long m_total_num_samples;
  };

  //template <class T, class V> 
  class metric
  {
  public:
    /// Constructor
    metric(lbann_comm *comm) {
      m_training_stats.init_stats();
      m_validation_stats.init_stats();
      m_testing_stats.init_stats();
      this->comm = comm;
      this->type = metric_type::INVALID;
    }

    /// Destructor
    ~metric(){};

    virtual void setup(int num_neurons, int mini_batch_size) {}
    virtual void fp_set_std_matrix_view(int64_t cur_mini_batch_size) {}
    virtual double compute_metric(ElMat& predictions_v, ElMat& groundtruth_v) {}
    virtual double report_metric(execution_mode mode) {}

    statistics* get_statistics(execution_mode mode);

    void record_error(double error, long num_samples);
    void reset_error();

  public:
    statistics m_training_stats;
    statistics m_validation_stats;
    statistics m_testing_stats;

    lbann_comm* comm;
    model* neural_network_model;
    metric_type type;
  };
}


#endif // LBANN_METRIC_HPP
