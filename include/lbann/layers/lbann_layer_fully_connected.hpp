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
// lbann_layer_fully_connected .hpp .cpp - Dense, fully connected, layer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_FULL_CONNECTED_HPP_INCLUDED
#define LBANN_LAYER_FULL_CONNECTED_HPP_INCLUDED

#include "lbann/layers/lbann_layer.hpp"
#include "lbann/layers/lbann_layer_activations.hpp"
#include "lbann/utils/lbann_random.hpp"
#include "lbann/models/lbann_model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lbann {
// FullyConnectedLayer : dense layer class
template <data_layout DATA_DIST>
class FullyConnectedLayer : public Layer {
 private:

  const weight_initialization m_weight_initialization;

  /// Views of the weight matrix that allow you to separate activation weights from bias weights
  ElMat *m_activation_weights_v;
  ElMat *m_bias_weights_v;
  ElMat *m_activation_weights_gradient_v;
  ElMat *m_bias_weights_gradient_v;

  /// Special matrices to allow backprop across the bias term
  ElMat *m_bias_bp_t;
  ElMat *m_bias_bp_t_v;
  ElMat *m_bias_weights_repl;
  DataType m_bias_term;

 protected:
  //Probability of dropping neuron/input used in dropout_layer
  //Range 0 to 1; default is -1 => no dropout
  DataType  WBL2NormSum;

 public:
  ////////////////////////////////////////////////////////////////////////////////
  // FullyConnectedLayer : single network layer class
  ////////////////////////////////////////////////////////////////////////////////
  // WB structure: (num units "neurons / filters" x (num features + 1))
  // Each row represents a neuron / filter
  // There is a column for each feature coming in from the previous layer plus 1 for the bias
  // [W00 ...   B0]
  // [|         |]
  // [Wn0       Bn]
  //
  // WB_D structure:
  // [dW     dB]
  // D structure:
  // [D        ]
  // Z, Zs, Act, Acts structure:
  // [Acts     ]

  FullyConnectedLayer(data_layout data_dist,
                      const uint index,
                      const int numPrevNeurons,
                      const uint numNeurons,
                      const uint minim_batch_size,
                      const activation_type activationType,
                      const weight_initialization init,
                      lbann_comm *comm,
                      optimizer *opt,
                      std::vector<regularizer *> regs = {})
    : Layer(data_dist,
            index, comm, opt, minim_batch_size, activationType, regs),
    m_weight_initialization(init) {

    m_type = layer_type::fully_connected;

    m_index = index;
    m_num_neurons = numNeurons;
    WBL2NormSum = 0.0;
    m_bias_term = 1.0;

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

  ~FullyConnectedLayer() {
    delete m_bias_bp_t;
    delete m_bias_weights_repl;
    delete m_activation_weights_v;
    delete m_bias_weights_v;
    delete m_activation_weights_gradient_v;
    delete m_bias_weights_gradient_v;
    delete m_bias_bp_t_v;
  }

  /// Matrices should be in MC,MR distributions
  void initialize_model_parallel_distribution() {
    m_bias_bp_t                      = new DistMat(comm->get_model_grid());
    m_bias_weights_repl              = new DistMatrix<DataType,MC,STAR>(comm->get_model_grid());

    /// Instantiate these view objects but do not allocate data for them
    m_activation_weights_v           = new DistMat(comm->get_model_grid());
    m_bias_weights_v                 = new DistMat(comm->get_model_grid());
    m_activation_weights_gradient_v  = new DistMat(comm->get_model_grid());
    m_bias_weights_gradient_v        = new DistMat(comm->get_model_grid());
    m_bias_bp_t_v                    = new DistMat(comm->get_model_grid());
  }

  /// Weight matrices should be in Star,Star and data matrices Star,VC distributions
  void initialize_data_parallel_distribution() {
    m_bias_bp_t                      = new StarVCMat(comm->get_model_grid());
    m_bias_weights_repl              = new StarMat(comm->get_model_grid());

    /// Instantiate these view objects but do not allocate data for them
    m_activation_weights_v           = new StarMat(comm->get_model_grid());
    m_bias_weights_v                 = new StarMat(comm->get_model_grid());
    m_activation_weights_gradient_v  = new StarMat(comm->get_model_grid());
    m_bias_weights_gradient_v        = new StarMat(comm->get_model_grid());
    m_bias_bp_t_v                    = new StarVCMat(comm->get_model_grid());
  }

  void setup(int numPrevNeurons) {
    Layer::setup(numPrevNeurons);

    // Initialize matrices
    // Note: the weights-bias matrix has an extra column so it includes bias term
    Zeros(*m_weights, m_num_neurons, numPrevNeurons+1);
    Zeros(*m_weights_gradient, m_num_neurons, numPrevNeurons + 1);
    Zeros(*m_prev_activations, numPrevNeurons, m_mini_batch_size);
    Zeros(*m_weighted_sum, m_num_neurons, m_mini_batch_size);
    Zeros(*m_activations, m_num_neurons, m_mini_batch_size);
    Zeros(*m_prev_error_signal, m_num_neurons, m_mini_batch_size);
    Zeros(*m_error_signal, numPrevNeurons, m_mini_batch_size); // m_error_signal holds the product of m_weights^T * m_prev_error_signal

    /// Setup independent views of the weight matrix for the activations and bias terms
    View(*m_activation_weights_v, *m_weights, ALL, IR(0, numPrevNeurons));
    View(*m_bias_weights_v, *m_weights, ALL, IR(numPrevNeurons));

    /// Setup independent views of the weights gradient matrix for the activations and bias terms
    View(*m_activation_weights_gradient_v, *m_weights_gradient, ALL, IR(0, numPrevNeurons));
    View(*m_bias_weights_gradient_v, *m_weights_gradient, ALL, IR(numPrevNeurons));

    /// Initialize the activations part of the weight matrix -- leave the bias term weights zero
    initialize_matrix(*m_activation_weights_v, m_weight_initialization, numPrevNeurons, m_num_neurons);

    /// Create a "transposed" vector of the bias term for use in backprop
    Ones(*m_bias_bp_t, 1, m_mini_batch_size);

    // Initialize optimizer
    if(m_optimizer != NULL) {
      m_optimizer->setup(m_weights);
    }

  }

  void fp_set_std_matrix_view() {
    int64_t cur_mini_batch_size = neural_network_model->get_current_mini_batch_size();

    Layer::fp_set_std_matrix_view();

    /// Note that the view of the bias backprop term is transposed, so the current mini-batch size is used to
    /// limit the height, not the width
    View(*m_bias_bp_t_v, *m_bias_bp_t, ALL, IR(0, cur_mini_batch_size));
  }

  void fp_linearity() {
    // Apply forward prop linearity

    // Apply bias
    Copy(*m_bias_weights_v, *m_bias_weights_repl);
    const Mat& local_bias_weights = m_bias_weights_repl->Matrix();
    IndexDependentFill(m_weighted_sum_v->Matrix(), (std::function<DataType(El::Int,El::Int)>)
                       ([&local_bias_weights](El::Int r, El::Int c)->DataType {
                         return local_bias_weights.Get(r);
                       }));
    Scale(m_bias_term, *m_weighted_sum_v);

    // Apply weight matrix
    switch(m_data_layout) {
    case data_layout::MODEL_PARALLEL:
      Gemm(NORMAL, NORMAL, DataType(1),
           *m_activation_weights_v,
           *m_prev_activations_v,
           DataType(1),
           *m_weighted_sum_v);
      break;
    case data_layout::DATA_PARALLEL:
      Gemm(NORMAL, NORMAL, DataType(1),
           m_activation_weights_v->LockedMatrix(),
           m_prev_activations_v->LockedMatrix(),
           DataType(1),
           m_weighted_sum_v->Matrix());
      break;
    }

    // Copy result to output matrix
    Copy(*m_weighted_sum_v, *m_activations_v);

  }

  void bp_linearity() {

    switch(m_data_layout) {
    case data_layout::MODEL_PARALLEL:
      // Compute the partial delta update for the next lower layer
      Gemm(TRANSPOSE, NORMAL, DataType(1),
           *m_activation_weights_v,
           *m_prev_error_signal_v,
           DataType(0),
           *m_error_signal_v);
      // Compute update for activation weights
      Gemm(NORMAL, TRANSPOSE, DataType(1)/get_effective_minibatch_size(),
           *m_prev_error_signal_v,
           *m_prev_activations_v,
           DataType(0),
           *m_activation_weights_gradient_v);
      // Compute update for bias terms
      Gemv(NORMAL, DataType(1)/get_effective_minibatch_size(),
           *m_prev_error_signal_v,
           *m_bias_bp_t_v,
           DataType(0),
           *m_bias_weights_gradient_v);
      break;
    case data_layout::DATA_PARALLEL:
      // Compute the partial delta update for the next lower layer
      Gemm(TRANSPOSE, NORMAL, DataType(1),
           m_activation_weights_v->LockedMatrix(),
           m_prev_error_signal_v->LockedMatrix(),
           DataType(0),
           m_error_signal_v->Matrix());
      // Compute update for activation weights
      Gemm(NORMAL, TRANSPOSE, DataType(1)/get_effective_minibatch_size(),
           m_prev_error_signal_v->LockedMatrix(),
           m_prev_activations_v->LockedMatrix(),
           DataType(0),
           m_activation_weights_gradient_v->Matrix());
      // Compute update for bias terms
      Gemv(NORMAL, DataType(1)/get_effective_minibatch_size(),
           m_prev_error_signal_v->LockedMatrix(),
           m_bias_bp_t_v->LockedMatrix(),
           DataType(0),
           m_bias_weights_gradient_v->Matrix());
      // Add gradients from all processes
      AllReduce(*m_weights_gradient,
                m_weights_gradient->RedundantComm());
      break;
    }

  }

  DataType computeCost(DistMat& deltas) {
    DataType avg_error = 0.0, total_error = 0.0;
    // Compute the L2 norm on the deltas (activation - y)
    ColSumMat norms;
    ColumnTwoNorms(deltas, norms);
    int c = 0;
    // Sum the local, total error
    for(int r = 0; r < norms.LocalHeight(); r++) {
      total_error += norms.GetLocal(r,c);
    }
    mpi::AllReduce(total_error, norms.DistComm());
    avg_error = total_error / norms.Height();
    return avg_error;
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
    DistMat WB_E1(m_weights->Grid());
    DistMat WB_E2(m_weights->Grid());
    DistMat Zs_E1(m_weighted_sum->Grid());
    DistMat Zs_E2(m_weighted_sum->Grid());
    DistMat Acts_E1(m_activations->Grid());
    DistMat Acts_E2(m_activations->Grid());
    DataType grad_diff = 0;
    DataType grad_sum = 0;

    Zeros(WB_E1, m_weights->Height(), m_weights->Width());
    Zeros(WB_E2, m_weights->Height(), m_weights->Width());
    Zeros(Zs_E1, m_weighted_sum->Height(), m_weighted_sum->Width());
    Zeros(Zs_E2, m_weighted_sum->Height(), m_weighted_sum->Width());
    Zeros(Acts_E1, m_activations->Height(), m_activations->Width());
    Zeros(Acts_E2, m_activations->Height(), m_activations->Width());

    Copy(*m_weights, WB_E1);
    Copy(*m_weights, WB_E2);

    DataType sum_error = 0;
    Int prow = 0;
    Int pcol = 0;

    if(WB_E1.IsLocal(prow, pcol)) {
      Int _prow = WB_E1.LocalRow(prow);
      Int _pcol = WB_E1.LocalCol(pcol);
      WB_E1.SetLocal(_prow, _pcol, WB_E1.GetLocal(_prow, _pcol) + Epsilon);
    }
    if(WB_E2.IsLocal(prow, pcol)) {
      Int _prow = WB_E2.LocalRow(prow);
      Int _pcol = WB_E2.LocalCol(pcol);
      WB_E2.SetLocal(_prow, _pcol, WB_E2.GetLocal(_prow, _pcol) - Epsilon);
    }
    for (Int row = 0; row < m_weights->Height(); row++) {
      for (Int col = 0; col < m_weights->Width(); col++) {
        //          printf("Updating %d: %d x %d\n", m_index, row, col);
        if(WB_E1.IsLocal(prow, pcol)) {
          Int _prow = WB_E1.LocalRow(prow);
          Int _pcol = WB_E1.LocalCol(pcol);
          WB_E1.SetLocal(_prow, _pcol, WB_E1.GetLocal(_prow, _pcol) - Epsilon);
        }
        if(WB_E2.IsLocal(prow, pcol)) {
          Int _prow = WB_E2.LocalRow(prow);
          Int _pcol = WB_E2.LocalCol(pcol);
          WB_E2.SetLocal(_prow, _pcol, WB_E2.GetLocal(_prow, _pcol) + Epsilon);
        }
        if(WB_E1.IsLocal(row, col)) {
          Int _row = WB_E1.LocalRow(row);
          Int _col = WB_E1.LocalCol(col);
          WB_E1.SetLocal(_row,  _col,  WB_E1.GetLocal(_row, _col)   + Epsilon);
        }
        if(WB_E2.IsLocal(row, col)) {
          Int _row = WB_E2.LocalRow(row);
          Int _col = WB_E2.LocalCol(col);
          WB_E2.SetLocal(_row,  _col,  WB_E2.GetLocal(_row, _col)   - Epsilon);
        }

        // J(theta)
        //            this->fp_linearity(*WB, *(PrevLayer.Acts), *Zs, *Acts);
        //this->fp_nonlinearity(*Acts);

        // J(thetaPlus(i))
        //            this->fp_linearity(WB_E1, *(PrevLayer.Acts), Zs_E1, Acts_E1);
        //this->fp_nonlinearity(Acts_E1);

        // J(thetaMinus(i))
        //            this->fp_linearity(WB_E2, *(PrevLayer.Acts), Zs_E2, Acts_E2);
        //this->fp_nonlinearity(Acts_E2);

        //            this->getCost();

        // gradApprox(i) = J(thetaPlus(i)) - J(thetaMinus(i)) / (2*Epsilon)
        Axpy(-1.0, *m_activations, Acts_E1);
        Axpy(-1.0, *m_activations, Acts_E2);

        bool bad_E1 = false, bad_E2 = false;
        for(Int r = 0; r < m_activations->Height(); r++) {
          for(Int c = 0; c < m_activations->Width(); c++) {
            if(Acts_E1.IsLocal(r,c)) {
              Int _r = Acts_E1.LocalRow(r);
              Int _c = Acts_E1.LocalCol(c);
              if((Acts_E1.GetLocal(_r,_c) > 1e-12 || Acts_E1.GetLocal(_r,_c) < -1e-12) && r != row) {
                bad_E1 = true;
                //                    cout << "Acts_E1=["<< r << "," << c << "]="<<Acts_E1.GetLocal(_r,_c)<<endl;
              }
            }
            if(Acts_E2.IsLocal(r,c)) {
              Int _r = Acts_E2.LocalRow(r);
              Int _c = Acts_E2.LocalCol(c);
              if((Acts_E2.GetLocal(_r,_c) > 1e-12 || Acts_E2.GetLocal(_r,_c) < -1e-12) && r != row) {
                bad_E2 = true;
                //                    cout << "Acts_E2=["<< r << "," << c << "]="<<Acts_E2.GetLocal(_r,_c)<<endl;
              }
            }
          }
        }

        if(bad_E1) {
          if(Acts_E1.Grid().Rank() == 0) {
            printf("BAD ENTRY Acts_E1 %lld x %lld\n", row, col);
          }
          cout.precision(20);
          Print(Acts_E1);
          if(Acts_E1.Grid().Rank() == 0) {
            printf("Acts\n");
          }
          Print(*m_activations);
        }
        if(bad_E2) {
          if(Acts_E2.Grid().Rank() == 0) {
            printf("BAD ENTRY Acts_E2 %lld x %lld\n", row, col);
          }
          Print(Acts_E2);
        }

        prow = row;
        pcol = col;
      }
    }
    DataType grad_error = sqrt(grad_diff / grad_sum);
    return grad_error;
  }
};

}


#endif // LBANN_LAYER_FULL_CONNECTED_HPP_INCLUDED
