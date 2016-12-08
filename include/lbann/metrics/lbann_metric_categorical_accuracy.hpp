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

#ifndef LBANN_CATEGORICAL_ACCURACY_HPP
#define LBANN_CATEGORICAL_ACCURACY_HPP

#include "lbann/metrics/lbann_metric.hpp"
#include "lbann/lbann_Elemental_extensions.h"

namespace lbann
{

  class categorical_accuracy : public metric
  {
  public:
    /// Constructor
    categorical_accuracy(lbann_comm* comm);
    
    /// Destructor
    ~categorical_accuracy();

    void setup(int num_neurons, int mini_batch_size);
    void fp_set_std_matrix_view(int64_t cur_mini_batch_size);
    double compute_metric(ElMat& predictions_v, ElMat& groundtruth_v);

#if 0
    V report_average_error();
    void display_average_error();

    void record_error(StarMat y_true, StarMat y_pred, long num_samples);

    void record_error(T error, long num_samples) {
      m_error_per_epoch += error;
      m_samples_per_epoch += num_samples;
    }
    
    void reset_error() {
      m_total_error += m_error_per_epoch;
      m_total_num_samples += m_samples_per_epoch;
      m_error_per_epoch = (T) 0;
      m_samples_per_epoch = 0;
    }
#endif

  protected:
    // T m_error_per_epoch;
    // long m_samples_per_epoch;

    // T m_total_error;
    // long m_total_num_samples;

    ColSumMat YsColMax; /// Note that the column max matrix has the number of mini-batches on the rows instead of columns
    StarMat YsColMaxStar;
    Mat m_max_index;    /// Local array to hold max indicies
    Mat m_reduced_max_indicies;  /// Local array to build global view of maximum indicies
    
    Mat Y_local;
    Mat Y_local_v;

    int64_t m_max_mini_batch_size;
  };
}


#endif // LBANN_CATEGORICAL_ACCURACY_HPP
