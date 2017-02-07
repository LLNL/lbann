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
// lbann_layer_fully_connected .hpp .cpp - Dense, fully connected layer
////////////////////////////////////////////////////////////////////////////////

#include "lbann/layers/lbann_layer_fully_connected.hpp"
#include "lbann/utils/lbann_random.hpp"
#include "lbann/models/lbann_model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;
using namespace El;

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

lbann::FullyConnectedLayer::
FullyConnectedLayer(const uint index,
                    const int numPrevNeurons,
                    const uint numNeurons,
                    const uint miniBatchSize,
                    const activation_type activationType,
                    const weight_initialization init,
                    lbann_comm* comm,
                    Optimizer *optimizer,
                    std::vector<regularizer*> regs)
  : Layer(index, comm, optimizer, miniBatchSize, activationType, regs),
    m_weight_initialization(init),
    m_activation_weights_v(comm->get_model_grid()),
    m_bias_weights_v(comm->get_model_grid()),
    m_activation_weights_gradient_v(comm->get_model_grid()),
    m_bias_weights_gradient_v(comm->get_model_grid()),
    m_bias_bp_t(comm->get_model_grid()),
    m_bias_bp_t_v(comm->get_model_grid())
{

    m_type = layer_type::fully_connected;

    Index = index;
    NumNeurons = numNeurons;
    WBL2NormSum = 0.0;
    m_bias_term = 1.0;
}

lbann::FullyConnectedLayer::~FullyConnectedLayer() {}

void lbann::FullyConnectedLayer::setup(int numPrevNeurons) {
  Layer::setup(numPrevNeurons);
    if(optimizer != NULL) {
      optimizer->setup(numPrevNeurons+1, NumNeurons);
    }

    // Initialize weight-bias matrix
    // Note that the weight-bias matrix has an extra column so that it will include
    // the bias term from the previous layer's activations in the linear combination
    Zeros(*m_weights, NumNeurons, numPrevNeurons+1);

    /// Given that we don't include the bias term here does that mean that we are setting it to zero to start with
    // Initialize weights
    DistMat weights;
    View(weights, *m_weights, IR(0,NumNeurons), IR(0,numPrevNeurons));
    switch(m_weight_initialization) {
    case weight_initialization::uniform:
      uniform_fill(weights, weights.Height(), weights.Width(),
                   DataType(0), DataType(1));
      break;
    case weight_initialization::normal:
      gaussian_fill(weights, weights.Height(), weights.Width(),
                    DataType(0), DataType(1));
      break;
    case weight_initialization::glorot_normal: {
      const DataType var = 2.0 / (numPrevNeurons + NumNeurons);
      gaussian_fill(weights, weights.Height(), weights.Width(),
                    DataType(0), sqrt(var));
      break;
    }
    case weight_initialization::glorot_uniform: {
      const DataType var = 2.0 / (numPrevNeurons + NumNeurons);
      uniform_fill(weights, weights.Height(), weights.Width(),
                   DataType(0), sqrt(3*var));
      break;
    }
    case weight_initialization::he_normal: {
      const DataType var = 1.0 / numPrevNeurons;
      gaussian_fill(weights, weights.Height(), weights.Width(),
                    DataType(0), sqrt(var));
      break;
    }
    case weight_initialization::he_uniform: {
      const DataType var = 1.0 / numPrevNeurons;
      uniform_fill(weights, weights.Height(), weights.Width(),
                   DataType(0), sqrt(3*var));
      break;
    }
    case weight_initialization::zero: // Zero initialization is default
    default:
      Zero(weights);
      break;
    }

    // Initialize other matrices
    Zeros(*m_weights_gradient, NumNeurons, numPrevNeurons + 1);
    Zeros(*m_prev_error_signal, NumNeurons, m_mini_batch_size);
    Zeros(*m_error_signal, numPrevNeurons, m_mini_batch_size); // m_error_signal holds the product of m_weights^T * m_prev_error_signal
    Zeros(*m_weighted_sum, NumNeurons, m_mini_batch_size);
    Zeros(*m_activations, NumNeurons, m_mini_batch_size);
    Zeros(*m_prev_activations, numPrevNeurons, m_mini_batch_size);

    /// Setup independent views of the weight matrix for the activations and bias terms
    View(m_activation_weights_v, *m_weights, IR(0, m_weights->Height()), IR(0, m_weights->Width()-1));
    View(m_bias_weights_v, *m_weights, IR(0, m_weights->Height()), IR(m_weights->Width()-1, m_weights->Width()));

    /// Setup independent views of the weights gradient matrix for the activations and bias terms
    View(m_activation_weights_gradient_v, *m_weights_gradient, IR(0, m_weights_gradient->Height()), IR(0, m_weights_gradient->Width()-1));
    View(m_bias_weights_gradient_v, *m_weights_gradient, IR(0, m_weights_gradient->Height()), IR(m_weights_gradient->Width()-1, m_weights_gradient->Width()));

    /// Create a "transposed" vector of the bias term for use in backprop
    Ones(m_bias_bp_t, m_mini_batch_size, 1);
}

void lbann::FullyConnectedLayer::fp_set_std_matrix_view() {
  int64_t cur_mini_batch_size = neural_network_model->get_current_mini_batch_size();

  Layer::fp_set_std_matrix_view();

  /// Note that the view of the bias backprop term is transposed, so the current mini-batch size is used to
  /// limit the height, not the width
  View(m_bias_bp_t_v, m_bias_bp_t, IR(0, cur_mini_batch_size), IR(0, m_bias_bp_t.Width()));
}

void lbann::FullyConnectedLayer::fp_linearity()
{
  // Apply forward prop linearity
  // Note that this is done on the entire matrix, regardless of if there is a partial mini-batch
  // Given that only the last mini-batch in an epoch could be smaller, it is not necessary to operate only on the sub-matrix

  StarMat local_bias_weights(comm->get_model_grid());
  Copy(m_bias_weights_v, local_bias_weights);
  IndexDependentFill(*m_weighted_sum, (std::function<DataType(El::Int,El::Int)>)
                     ([this, local_bias_weights](El::Int r, El::Int c)->DataType { 
                       El::Int rL = local_bias_weights.LocalRow(r);
                       if(!local_bias_weights.IsLocal(r,0)) { throw lbann_exception("Bad fill");}
                       return local_bias_weights.GetLocal(rL,0) * m_bias_term;
                     }));
  Gemm(NORMAL, NORMAL, (DataType) 1., m_activation_weights_v, *m_prev_activations, (DataType) 1., *m_weighted_sum);
  Copy(*m_weighted_sum_v, *m_activations_v);
}

void lbann::FullyConnectedLayer::bp_linearity()
{
  // Compute the partial delta update for the next lower layer
  Gemm(TRANSPOSE, NORMAL, (DataType) 1., m_activation_weights_v, *m_prev_error_signal_v, (DataType) 0., *m_error_signal_v);
  // Compute update for activation weights
  Gemm(NORMAL, TRANSPOSE, (DataType) 1.0/get_effective_minibatch_size(), *m_prev_error_signal_v,
       *m_prev_activations_v, (DataType) 0., m_activation_weights_gradient_v);
  // Compute update for bias terms
  Gemv(NORMAL, (DataType) 1.0/get_effective_minibatch_size(), *m_prev_error_signal_v,
       m_bias_bp_t_v, (DataType) 0., m_bias_weights_gradient_v);
}

DataType lbann::FullyConnectedLayer::computeCost(DistMat &deltas) {
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

DataType lbann::FullyConnectedLayer::WBL2norm() {
  DataType nrm2 = Nrm2(*m_weights);
  return nrm2 * nrm2;
}

inline DataType _sq(DataType x) { return (x * x); }
inline DataType _sqrt(DataType x) { return (1 / sqrt(x + 1e-8)); }

bool lbann::FullyConnectedLayer::update()
{
  if(m_execution_mode == execution_mode::training) {
    optimizer->update_weight_bias_matrix(*m_weights_gradient, *m_weights);
  }
  return true;
}

DataType lbann::FullyConnectedLayer::checkGradient(Layer& PrevLayer, const DataType Epsilon)
{
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
    int prow = 0;
    int pcol = 0;

    if(WB_E1.IsLocal(prow, pcol)) {
      int _prow = WB_E1.LocalRow(prow);
      int _pcol = WB_E1.LocalCol(pcol);
      WB_E1.SetLocal(_prow, _pcol, WB_E1.GetLocal(_prow, _pcol) + Epsilon);
    }
    if(WB_E2.IsLocal(prow, pcol)) {
      int _prow = WB_E2.LocalRow(prow);
      int _pcol = WB_E2.LocalCol(pcol);
      WB_E2.SetLocal(_prow, _pcol, WB_E2.GetLocal(_prow, _pcol) - Epsilon);
    }
    for (int row = 0; row < m_weights->Height(); row++) {
        for (int col = 0; col < m_weights->Width(); col++) {
          //          printf("Updating %d: %d x %d\n", Index, row, col);
            if(WB_E1.IsLocal(prow, pcol)) {
              int _prow = WB_E1.LocalRow(prow);
              int _pcol = WB_E1.LocalCol(pcol);
              WB_E1.SetLocal(_prow, _pcol, WB_E1.GetLocal(_prow, _pcol) - Epsilon);
            }
            if(WB_E2.IsLocal(prow, pcol)) {
              int _prow = WB_E2.LocalRow(prow);
              int _pcol = WB_E2.LocalCol(pcol);
              WB_E2.SetLocal(_prow, _pcol, WB_E2.GetLocal(_prow, _pcol) + Epsilon);
            }
            if(WB_E1.IsLocal(row, col)) {
              int _row = WB_E1.LocalRow(row);
              int _col = WB_E1.LocalCol(col);
              WB_E1.SetLocal(_row,  _col,  WB_E1.GetLocal(_row, _col)   + Epsilon);
            }
            if(WB_E2.IsLocal(row, col)) {
              int _row = WB_E2.LocalRow(row);
              int _col = WB_E2.LocalCol(col);
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
            for(int r = 0; r < m_activations->Height(); r++) {
              for(int c = 0; c < m_activations->Width(); c++) {
                if(Acts_E1.IsLocal(r,c)) {
                  int _r = Acts_E1.LocalRow(r);
                  int _c = Acts_E1.LocalCol(c);
                  if((Acts_E1.GetLocal(_r,_c) > 1e-12 || Acts_E1.GetLocal(_r,_c) < -1e-12) && r != row) {
                    bad_E1 = true;
                    //                    cout << "Acts_E1=["<< r << "," << c << "]="<<Acts_E1.GetLocal(_r,_c)<<endl;
                  }
                }
                if(Acts_E2.IsLocal(r,c)) {
                  int _r = Acts_E2.LocalRow(r);
                  int _c = Acts_E2.LocalCol(c);
                  if((Acts_E2.GetLocal(_r,_c) > 1e-12 || Acts_E2.GetLocal(_r,_c) < -1e-12) && r != row) {
                    bad_E2 = true;
                    //                    cout << "Acts_E2=["<< r << "," << c << "]="<<Acts_E2.GetLocal(_r,_c)<<endl;
                  }
                }
              }
            }

            if(bad_E1) {
              if(Acts_E1.Grid().Rank() == 0) {
                printf("BAD ENTRY Acts_E1 %d x %d\n", row, col);
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
                printf("BAD ENTRY Acts_E2 %d x %d\n", row, col);
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
