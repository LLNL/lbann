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
//
// lbann_layer_softmax .hpp .cpp - Softmax layer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_SOFTMAX_HPP_INCLUDED
#define LBANN_LAYER_SOFTMAX_HPP_INCLUDED

#include "lbann/layers/lbann_layer.hpp"
#include "lbann/lbann_Elemental_extensions.h"
#include "lbann/io/lbann_file_io.hpp"
#include "lbann/utils/lbann_random.hpp"
#include "lbann/models/lbann_model.hpp"
#include <unistd.h>
#include <string>

namespace lbann {
template <data_layout DATA_DIST>
class SoftmaxLayer: public Layer {
 protected:
  DataType   WBL2NormSum;

 private:
  weight_initialization m_weight_initialization;
  AbsDistMat *m_workspace;
  AbsDistMat *m_workspace_v;

 public:
  SoftmaxLayer(data_layout data_dist,
               const uint index,
               const int numPrevNeurons,
               const uint numNeurons,
               const uint minim_batch_size,
               const weight_initialization init,
               lbann_comm *comm,
               optimizer *opt)
    :  Layer(data_dist, index, comm, opt, minim_batch_size),
       m_weight_initialization(init) {
    m_type = layer_type::softmax;
    m_index = index;
    m_num_neurons = numNeurons;
    WBL2NormSum = 0.0;

    // Setup the data distribution
    switch(data_dist) {
    case data_layout::MODEL_PARALLEL:
      initialize_model_parallel_distribution();
      break;
    case data_layout::DATA_PARALLEL:
      initialize_data_parallel_distribution();
      break;
    default:
      throw lbann_exception(std::string{} + __FILE__ + " " +
                            std::to_string(__LINE__) +
                            "Invalid data layout selected");
    }
  }

  ~SoftmaxLayer() {
    delete m_workspace;
    delete m_workspace_v;
  }

  /// Matrices should be in MC,MR distributions
  void initialize_model_parallel_distribution() {
    m_workspace = new StarMRMat(comm->get_model_grid());
    m_workspace_v = new StarMRMat(comm->get_model_grid());
  }

  /// Weight matrices should be in Star,Star and data matrices Star,VC distributions
  void initialize_data_parallel_distribution() {
    m_workspace = new StarVCMat(comm->get_model_grid());
    m_workspace_v = new StarVCMat(comm->get_model_grid());
  }

  void setup(int numPrevNeurons) {
    Layer::setup(numPrevNeurons);

    // Zero the weight-bias matrix
    Zeros(*m_weights, m_num_neurons, numPrevNeurons);

    /// Initialize the activations part of the weight matrix -- leave the bias term weights zero
    initialize_matrix(*m_weights, m_weight_initialization, numPrevNeurons, m_num_neurons);

    // Initialize other matrices
    Zeros(*m_weights_gradient, m_num_neurons, numPrevNeurons);
    Zeros(*m_prev_error_signal, m_num_neurons, m_mini_batch_size);
    Zeros(*m_error_signal, numPrevNeurons, m_mini_batch_size); // m_error_signal holds the product of m_weights^T * m_prev_error_signal
    Zeros(*m_weighted_sum, m_num_neurons, m_mini_batch_size);
    Zeros(*m_activations, m_num_neurons, m_mini_batch_size);
    Zeros(*m_prev_activations, numPrevNeurons, m_mini_batch_size);
    Zeros(*m_workspace, 1, m_mini_batch_size);

    // Initialize optimizer
    if(m_optimizer != NULL) {
      m_optimizer->setup(m_weights);
    }

  }

  void fp_set_std_matrix_view() {
    int64_t cur_mini_batch_size = neural_network_model->get_current_mini_batch_size();
    Layer::fp_set_std_matrix_view();
    View(*m_workspace_v, *m_workspace, ALL, IR(0, cur_mini_batch_size));
  }

  void fp_linearity() {

    // Apply weight matrix
    switch(m_data_layout) {
    case data_layout::MODEL_PARALLEL:
      Gemm(NORMAL, NORMAL, DataType(1),
           *m_weights,
           *m_prev_activations_v,
           DataType(0),
           *m_weighted_sum_v);
      break;
    case data_layout::DATA_PARALLEL:
      Gemm(NORMAL, NORMAL, DataType(1),
           m_weights->LockedMatrix(),
           m_prev_activations_v->LockedMatrix(),
           DataType(0),
           m_weighted_sum_v->Matrix());
      break;
    }

    // Copy result to output matrix
    Copy(*m_weighted_sum_v, *m_activations_v);

  }

  void fp_nonlinearity() {

    // Get local matrices and parameters
    Mat& workspace_local = m_workspace_v->Matrix();
    Mat& activations_local = m_activations_v->Matrix();
    const Int local_height = activations_local.Height();
    const Int local_width = activations_local.Width();

    // Find maximum entry in each column
#pragma omp parallel for
    for(Int c=0; c<local_width; ++c) {
    DataType max_entry = -INFINITY;
    for(Int r=0; r<local_height; ++r) {
    max_entry = Max(max_entry, activations_local.Get(r,c));
  }
    workspace_local.Set(Int(0), c, max_entry);
  }
    AllReduce(*m_workspace_v, m_workspace_v->RedundantComm(), mpi::MAX);

    // Subtract column max and exponentiate activations
    // Note: Subtracting the column max prevents activations from blowing
    //   up. Large negative values underflow to 0.
    IndexDependentMap(activations_local,
      (std::function<DataType(Int,Int,const DataType&)>)
      ([this,&workspace_local](Int r, Int c, const DataType& z)->DataType {
    return Exp(z - workspace_local.Get(Int(0), c));
  }));

    // Compute column sums
#pragma omp parallel for
    for(Int c=0; c<local_width; ++c) {
      DataType sum = 0;
      for(Int r=0; r<local_height; ++r) {
        sum += activations_local.Get(r,c);
      }
      workspace_local.Set(Int(0), c, sum);
    }
    AllReduce(*m_workspace_v, m_workspace_v->RedundantComm(), mpi::SUM);

    // Divide activations by column sums
    // This truncates small values to 0 to avoid them becoming denormalized later
    // in the forward/backward stages. Denormalized values can significantly
    // impact floating point performance.
    IndexDependentMap(activations_local,
                      (std::function<DataType(Int,Int,const DataType&)>)
                      ([this,&workspace_local](Int r, Int c, const DataType& z)->DataType {
                        const DataType v = z / workspace_local.Get(Int(0), c);
                        return Abs(v) < 1e-8 ? DataType(1e-8) : v;
                      }));

  }

  void bp_linearity() {

    switch(m_data_layout) {
    case data_layout::MODEL_PARALLEL:
      // Compute the partial delta update for the next lower layer
      Gemm(TRANSPOSE, NORMAL, DataType(1),
           *m_weights,
           *m_prev_error_signal_v,
           DataType(0),
           *m_error_signal_v);
      // Compute update for activation weights
      Gemm(NORMAL, TRANSPOSE, DataType(1)/get_effective_minibatch_size(),
           *m_prev_error_signal_v,
           *m_prev_activations_v,
           DataType(0),
           *m_weights_gradient);
      break;
    case data_layout::DATA_PARALLEL:
      // Compute the partial delta update for the next lower layer
      Gemm(TRANSPOSE, NORMAL, DataType(1),
           m_weights->LockedMatrix(),
           m_prev_error_signal_v->LockedMatrix(),
           DataType(0),
           m_error_signal_v->Matrix());
      // Compute update for activation weights
      Gemm(NORMAL, TRANSPOSE, DataType(1)/get_effective_minibatch_size(),
           m_prev_error_signal_v->LockedMatrix(),
           m_prev_activations_v->LockedMatrix(),
           DataType(0),
           m_weights_gradient->Matrix());
      // Add gradients from all processes
      AllReduce(*m_weights_gradient,
                m_weights_gradient->RedundantComm());
      break;
    }

  }

  void bp_nonlinearity() {

    // Stop early if objective function is categorical cross entropy
    // Note: error signal is already computed in objective function object
    if(neural_network_model->obj_fn->type == objective_functions::obj_fn_type::categorical_cross_entropy
       && (m_next_layer_type == layer_type::target_distributed_minibatch
           || m_next_layer_type == layer_type::target_distributed_minibatch_parallel_io
           || m_next_layer_type == layer_type::target_partitioned_minibatch_parallel_io
           // || m_next_layer_type == layer_type::target_unsupervised
           )) {
      return;
    }

    // Get local matrices and parameters
    const Mat& activations_local = m_activations_v->LockedMatrix();
    Mat& workspace_local = m_workspace_v->Matrix();
    Mat& prev_error_signal_local = m_prev_error_signal_v->Matrix();
    const Int local_height = activations_local.Height();
    const Int local_width = activations_local.Width();

    // Compute dot products
    // Note: prev_error_signal^T activations
    for(Int c=0; c<local_width; ++c) {
      workspace_local.Set(Int(0), c,
                          Dot(prev_error_signal_local(ALL,IR(c)),
                              activations_local(ALL,IR(c))));
    }
    AllReduce(*m_workspace_v, m_workspace_v->RedundantComm(), mpi::SUM);

    // Update error signal
    // Note: prev_error_signal := activations * (prev_error_signal - prev_error_signal^T activations)
    IndexDependentMap(prev_error_signal_local,
                      (std::function<DataType(Int,Int,const DataType&)>)
                      ([this,&activations_local,&workspace_local]
                       (Int r, Int c, const DataType& z)->DataType {
                        const DataType activations_entry = activations_local.Get(r,c);
                        const DataType dot_product_entry = workspace_local.Get(Int(0),c);
                        return activations_entry * (z - dot_product_entry);
                      }));

  }

  DataType WBL2norm() {
    DataType nrm2 = Nrm2(*m_weights);
    return nrm2 * nrm2;
  }

  inline DataType _sq(DataType x) {
    return (x * x);
  }
  inline DataType _sqrt(DataType x) {
    return (1 / sqrt(x + 1e-8));
  }

  bool update() {
    double start = get_time();
    Layer::update();
    if(m_execution_mode == execution_mode::training) {
      m_optimizer->update(m_weights_gradient);
    }
    update_time += get_time() - start;
    return true;
  }

  DataType checkGradient(Layer& PrevLayer, const DataType Epsilon) {
    return 0.0;
  }

  bool saveToCheckpoint(int fd, const char *filename, uint64_t *bytes) {
    return Layer::saveToCheckpoint(fd, filename, bytes);
  }

  bool loadFromCheckpoint(int fd, const char *filename, uint64_t *bytes) {
    return Layer::loadFromCheckpoint(fd, filename, bytes);
  }

  bool saveToCheckpointShared(lbann::persist& p) {
    return Layer::saveToCheckpointShared(p);
  }

  bool loadFromCheckpointShared(lbann::persist& p) {
    return Layer::loadFromCheckpointShared(p);
  }
};
}


#endif // LBANN_LAYER_SOFTMAX_HPP_INCLUDED
