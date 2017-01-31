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

#include "lbann/layers/lbann_layer_softmax.hpp"
#include "lbann/lbann_Elemental_extensions.h"
#include "lbann/io/lbann_file_io.hpp"
#include "lbann/utils/lbann_random.hpp"
#include "lbann/models/lbann_model.hpp"
#include <unistd.h>

using namespace std;
using namespace El;

lbann::SoftmaxLayer::SoftmaxLayer(const uint index,
                                  const int numPrevNeurons,
                                  const uint numNeurons,
                                  const uint miniBatchSize,
                                  const weight_initialization init,
                                  lbann_comm* comm,
                                  Optimizer *optimizer)
  :  Layer(index, comm, optimizer, miniBatchSize),
     m_weight_initialization(init),
     ZsColMax(comm->get_model_grid()),
     ZsNormExpSum(comm->get_model_grid()),
     norms(comm->get_model_grid()),
     ZsColMaxStar(comm->get_model_grid()),
     ZsNormExpSumStar(comm->get_model_grid())
{
    Index = index;
    NumNeurons = numNeurons;
    WBL2NormSum = 0.0;
}

void lbann::SoftmaxLayer::setup(int numPrevNeurons) {
  Layer::setup(numPrevNeurons);
    if(optimizer != NULL) {
      optimizer->setup(numPrevNeurons, NumNeurons);
    }

    // Initialize weight-bias matrix
    Zeros(*m_weights, NumNeurons, numPrevNeurons);

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
    Zeros(*m_weights_gradient, NumNeurons, numPrevNeurons);
    Zeros(*m_prev_error_signal, NumNeurons, m_mini_batch_size);
    Zeros(*m_error_signal, numPrevNeurons, m_mini_batch_size); // m_error_signal holds the product of m_weights^T * m_prev_error_signal
    Zeros(*m_weighted_sum, NumNeurons, m_mini_batch_size);
    Zeros(*m_activations, NumNeurons, m_mini_batch_size);
    Zeros(*m_prev_activations, numPrevNeurons, m_mini_batch_size);
}

// template <typename Dtype>
// void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//     vector<Blob<Dtype>*>* top) {

void lbann::SoftmaxLayer::fp_linearity()
{
  // _Z = m_weights * Xs                                               -- Xs is previous layer Activations
  // ZsColMax[c,0] = max(_Z[0..numNeurons-1, c])                -- (m_mini_batch_size x 1)
  // ZsNorm[r,c] = _Z[r,c] - ZsColMax[c,0]                      -- Column-wise normalized Zs matrix: _Z[r,c] - ZsColMax[1,c]
  // ZsNormExp[r,c] = exp(ZsNorm[r,c])
  // ZsNormExpSum[c,0] = sum(ZsNormExp[0..numNeurons-1, c])     -- Column-wise sum over normalized, exponentiated _Z
  // _Y[r,c] = ZsNormExp[r,c] / ZsNormExpSum[c,0]               -- exp(norm(_Z[r,c])) = Sum(exp(norm(Zs[r,c])))

  // Apply linear transform
  Gemm(NORMAL, NORMAL, (DataType) 1.0, *m_weights, *m_prev_activations_v, (DataType) 0.0, *m_weighted_sum_v);

  // For each minibatch (column) find the maximimum value
  Zeros(ZsColMax, m_mini_batch_size, 1); // Clear the entire matrix
  ColumnMax((DistMat&) *m_weighted_sum_v, ZsColMax);

  // Redistribute the per-minibatch maximum values
  Copy(ZsColMax, ZsColMaxStar);

  /// @todo - BVE FIXME I believe that this should be put into a softmax non-linearity / activation function

  // Compute exp(z) of each entry. Subtract the max of each column from its
  // entries to prevent the exp from blowing up. Large negative values are
  // expected to underflow to 0.
  IndexDependentMap(
    *m_weighted_sum_v,
    (std::function<DataType(El::Int,El::Int,const DataType&)>)
    ([this](El::Int r, El::Int c, const DataType& z)->DataType {
      El::Int rL = this->ZsColMaxStar.LocalRow(c);
      //      if(isnan(std::exp(z - this->ZsColMaxStar.GetLocal(rL, 0)))) { cout << "[" << comm->get_rank_in_world() << "] has a nan "<<std::exp(z - this->ZsColMaxStar.GetLocal(rL, 0)) << " at [" << r << ", " << c << "]="<<z<<endl;throw(new lbann_exception("Foo"));}
      return std::exp(z - this->ZsColMaxStar.GetLocal(rL, 0));
    }));

  // For each minibatch (column) sum up the exponentiated values
  Zeros(ZsNormExpSum, m_mini_batch_size, 1); // Clear the entire matrix
  //  ColSumMat ZsNormExpSum;
  ColumnSum((DistMat&) *m_weighted_sum_v, ZsNormExpSum);
  Copy(ZsNormExpSum, ZsNormExpSumStar);

  // Divide each entry: exp(x_ij) / Sum_i(exp(x_ij))
  Copy(*m_weighted_sum_v, *m_activations_v);

  IndexDependentMap(*m_activations_v,
                    (std::function<DataType(Int,Int,const DataType&)>)([this /*ZsNormExpSum*/](Int r, Int c, const DataType& z)->
                                                                DataType{Int rL = this->ZsNormExpSumStar.LocalRow(c); return z/this->ZsNormExpSumStar.GetLocal(rL,0);}));

#if 0
  ColSumMat Ycheck(_Y.Grid());
  Zeros(Ycheck, m_mini_batch_size, 1);
  ColumnSum((DistMat&) _Y /*ZsNormExp*/, Ycheck);
  StarMat YcheckStar(_Y.Grid());
  Copy(Ycheck, YcheckStar);
  DataType sum = 0.0;
  for(int i = 0; i < YcheckStar.Height(); i++) {
    Int l = YcheckStar.LocalRow(i);
    sum = YcheckStar.GetLocal(l, 0);
    //    sum += YcheckStar.GetLocal(l, 0);
    //  }
  if(YcheckStar.GetLocal(l, 0) < 0 || (sum >= 1.00001 || sum <= 0.99999)) {
    if(_Y.Grid().Rank() == 0) {
      printf("The softmax does not add up %lf\n", sum);
    }
    Print(_Y);
    Print(YcheckStar);
  }
  }
#endif
}

void lbann::SoftmaxLayer::bp_linearity()
{
  // Compute the partial delta update for the next lower layer (delta * activation_prev^T)
  Gemm(TRANSPOSE, NORMAL, (DataType) 1., *m_weights, *m_prev_error_signal_v, (DataType) 0., *m_error_signal_v);
  
  // Compute update for weights - include division by mini-batch size
  Gemm(NORMAL, TRANSPOSE, (DataType) 1.0/get_effective_minibatch_size(), *m_prev_error_signal_v,
       *m_prev_activations_v, (DataType) 0., *m_weights_gradient);
}

DataType lbann::SoftmaxLayer::WBL2norm() {
  DataType nrm2 = Nrm2(*m_weights);
  return nrm2 * nrm2;
}

inline DataType _sq(DataType x) { return (x * x); }
inline DataType _sqrt(DataType x) { return (1 / sqrt(x + 1e-8)); }

bool lbann::SoftmaxLayer::update()
{
  if(m_execution_mode == execution_mode::training) {
    optimizer->update_weight_bias_matrix(*m_weights_gradient, *m_weights);
  }
  return true;
}

DataType lbann::SoftmaxLayer::checkGradient(Layer& PrevLayer, const DataType Epsilon)
{
  return 0.0;
}

bool lbann::SoftmaxLayer::saveToCheckpoint(int fd, const char* filename, uint64_t* bytes)
{
  return Layer::saveToCheckpoint(fd, filename, bytes);
}

bool lbann::SoftmaxLayer::loadFromCheckpoint(int fd, const char* filename, uint64_t* bytes)
{
  return Layer::loadFromCheckpoint(fd, filename, bytes);
}

bool lbann::SoftmaxLayer::saveToCheckpointShared(lbann::persist& p)
{
  return Layer::saveToCheckpointShared(p);
}

bool lbann::SoftmaxLayer::loadFromCheckpointShared(lbann::persist& p)
{
    return Layer::loadFromCheckpointShared(p);
}
