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

#include "lbann/objective_functions/lbann_objective_fn_categorical_cross_entropy.hpp"
#include <sys/types.h>
#include <unistd.h>

using namespace std;
using namespace El;

lbann::objective_functions::categorical_cross_entropy::categorical_cross_entropy(lbann_comm* comm)
  : objective_fn("categorical_cross_entropy")
{
  this->type = obj_fn_type::categorical_cross_entropy;
}

lbann::objective_functions::categorical_cross_entropy::~categorical_cross_entropy() {}

void lbann::objective_functions::categorical_cross_entropy::setup(int num_neurons, int mini_batch_size) {}

void lbann::objective_functions::categorical_cross_entropy::fp_set_std_matrix_view(int64_t cur_mini_batch_size) {}

/// Compute the cross-entropy cost function - comparing the activations from the previous layer and the ground truth (activations of this layer)
/// cost=-1/m*(sum(sum(groundTruth.*log(a3))))
/// predictions_v - a.k.a. coding_dist - coding distribution (e.g. prev_activations)
/// groundtruth_v - a.k.a. true_dist - true distribution (e.g. activations)
double lbann::objective_functions::categorical_cross_entropy::compute_categorical_cross_entropy(ElMat &predictions_v,
                                                                                                ElMat &groundtruth_v) {

    // Compute categorical cross entropy on current process
    //   Note: robust against zero predictions; assumes predictions_v
    //   and groundtruth_v are aligned
    DataType total_error = 0;
    for(Int c = 0; c < groundtruth_v.LocalWidth(); c++) {
      for(Int r = 0; r < groundtruth_v.LocalHeight(); r++) {
        const DataType true_val = groundtruth_v.GetLocal(r,c);
        if(true_val != DataType(0)) {
          const DataType pred_val = predictions_v.GetLocal(r,c);
          total_error += - true_val * Log(pred_val);
        }
      }
    }

    // Get categorical cross entropy by summing results from all processes
    total_error = mpi::AllReduce(total_error, groundtruth_v.DistComm());
    return total_error;

}

/// Compute the average categorical cross entropy over the mini-batch
double lbann::objective_functions::categorical_cross_entropy::compute_obj_fn(ElMat &predictions_v, ElMat &groundtruth_v) {
    double avg_error = 0.0, total_error = 0.0;
    Int cur_mini_batch_size = groundtruth_v.Width();

    total_error = compute_categorical_cross_entropy(predictions_v, groundtruth_v);

    avg_error = total_error / cur_mini_batch_size;

    return avg_error;
}

void lbann::objective_functions::categorical_cross_entropy::compute_obj_fn_derivative(ElMat &predictions_v, ElMat &groundtruth_v, ElMat &error_signal_v) {

  /// @todo Handle case with softmax output and cross entropy objective function
  /// Compute the error between the target values and the previous layer's activations
  /// Copy the results to the m_error_signal variable for access by the next lower layer
  // Copy(predictions_v, error_signal_v); // delta = (activation - y)
  // Axpy(-1., groundtruth_v, error_signal_v); // Per-neuron error

  // Compute derivative of categorical cross entropy
  //   Note: robust against zero predictions; assumes predictions_v,
  //   groundtruth_v, and error_signal_v are aligned
  IndexDependentFill(error_signal_v.Matrix(),
                     (std::function<DataType(Int,Int)>)
                     ([&predictions_v, &groundtruth_v](Int r, Int c)->DataType {
                       const DataType true_val = groundtruth_v.GetLocal(r,c);
                       if(true_val != DataType(0))
                         return - true_val / predictions_v.GetLocal(r,c);
                       else
                         return DataType(0);
                     }));

}
