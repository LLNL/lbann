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
     ZsNormExpSumStar(comm->get_model_grid()),
     Acts_Cost(comm->get_model_grid()),
     m_minibatch_cost(comm->get_model_grid())
{
    Index = index;
    NumNeurons = numNeurons;
    aggregate_cost = 0.0;
    num_backprop_steps = 0;
    WBL2NormSum = 0.0;
}

void lbann::SoftmaxLayer::setup(int numPrevNeurons) {
  Layer::setup(numPrevNeurons);
    if(optimizer != NULL) {
      optimizer->setup(numPrevNeurons+1, NumNeurons);
    }

    // Initialize weight-bias matrix
    Zeros(*WB, NumNeurons, numPrevNeurons+1);

    // Initialize weights
    DistMat weights;
    View(weights, *WB, IR(0,NumNeurons), IR(0,numPrevNeurons));
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
    Zeros(*WB_D, NumNeurons, numPrevNeurons + 1);
    Zeros(*Ds, NumNeurons, m_mini_batch_size);
    Zeros(*Ds_Temp, numPrevNeurons + 1, m_mini_batch_size); // Ds_Temp holds the product of WB^T * Ds
    Zeros(*Zs, NumNeurons, m_mini_batch_size);
    Zeros(*Acts, NumNeurons, m_mini_batch_size);
    Zeros(Acts_Cost, NumNeurons, m_mini_batch_size);
    Zeros(m_minibatch_cost, m_mini_batch_size, 1);

    /// Create a view of the weights matrix
    View(*m_weights_v, *WB, IR(0, WB->Height()), IR(0, WB->Width()));
    View(*m_weights_gradient_v, *WB_D, IR(0, WB_D->Height()), IR(0, WB_D->Width()));
}

// template <typename Dtype>
// void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//     vector<Blob<Dtype>*>* top) {

/** Override the standard matrix sub-view code so that the activations view includes all elements. */
void lbann::SoftmaxLayer::fp_set_std_matrix_view() {
  int64_t cur_mini_batch_size = neural_network_model->get_current_mini_batch_size();

  View(*m_preactivations_v, *Zs, IR(0, Zs->Height()), IR(0, cur_mini_batch_size));
  View(*m_prev_error_signal_v, *Ds, IR(0, Ds->Height()), IR(0, cur_mini_batch_size));
  View(*m_error_signal_v, *Ds_Temp, IR(0, Ds_Temp->Height()), IR(0, cur_mini_batch_size));
  View(*m_activations_v, *Acts, IR(0, Acts->Height()), IR(0, cur_mini_batch_size));
  View(m_activations_cost_v, Acts_Cost, IR(0, Acts_Cost.Height()), IR(0, cur_mini_batch_size));

  // Update the layer's effective mini-batch size so it averages properly.
  if(cur_mini_batch_size != m_mini_batch_size) { /// When the current mini-batch is partial, check with the other models to figure out the entire size of the complete mini-batch
    int total_mini_batch_size = comm->intermodel_allreduce((int) cur_mini_batch_size);
    //    cout << "[" << comm->get_rank_in_world() << "] total_mini_batch_size " << total_mini_batch_size << " and cur mini batch size " << cur_mini_batch_size << endl;
    set_effective_minibatch_size(total_mini_batch_size);
  }else {
    set_effective_minibatch_size(cur_mini_batch_size * comm->get_num_models());
  }
}

void lbann::SoftmaxLayer::fp_linearity()
{
  // _Z = WB * Xs                                               -- Xs is previous layer Activations
  // ZsColMax[c,0] = max(_Z[0..numNeurons-1, c])                -- (m_mini_batch_size x 1)
  // ZsNorm[r,c] = _Z[r,c] - ZsColMax[c,0]                      -- Column-wise normalized Zs matrix: _Z[r,c] - ZsColMax[1,c]
  // ZsNormExp[r,c] = exp(ZsNorm[r,c])
  // ZsNormExpSum[c,0] = sum(ZsNormExp[0..numNeurons-1, c])     -- Column-wise sum over normalized, exponentiated _Z
  // _Y[r,c] = ZsNormExp[r,c] / ZsNormExpSum[c,0]               -- exp(norm(_Z[r,c])) = Sum(exp(norm(Zs[r,c])))

  // Convert forward prop matrix to MC,MR format
  // TODO: store this matrix for back prop
  DistMatrixReadProxy<DataType,DataType,MC,MR> XProxy(*fp_input);
  DistMat& X = XProxy.Get();
  DistMat X_v;

  int64_t curr_mini_batch_size = neural_network_model->get_current_mini_batch_size();

  View(X_v, X, IR(0, X.Height()), IR(0, curr_mini_batch_size));

  // Apply linear transform
  Gemm(NORMAL, NORMAL, (DataType) 1.0, *m_weights_v, X_v, (DataType) 0.0, *m_preactivations_v);

  // For each minibatch (column) find the maximimum value
  Zeros(ZsColMax, m_mini_batch_size, 1); // Clear the entire matrix
  ColumnMax((DistMat&) *m_preactivations_v, ZsColMax);

  // Redistribute the per-minibatch maximum values
  Copy(ZsColMax, ZsColMaxStar);

  // Compute exp(z) of each entry. Subtract the max of each column from its
  // entries to prevent the exp from blowing up. Large negative values are
  // expected to underflow to 0.
  IndexDependentMap(
    *m_preactivations_v,
    (std::function<DataType(int,int,DataType)>)
    ([this](int r, int c, DataType z)->DataType {
      Int rL = this->ZsColMaxStar.LocalRow(c);
      //      if(isnan(std::exp(z - this->ZsColMaxStar.GetLocal(rL, 0)))) { cout << "[" << comm->get_rank_in_world() << "] has a nan "<<std::exp(z - this->ZsColMaxStar.GetLocal(rL, 0)) << " at [" << r << ", " << c << "]="<<z<<endl;throw(new lbann_exception("Foo"));}
      return std::exp(z - this->ZsColMaxStar.GetLocal(rL, 0));
    }));

  // For each minibatch (column) sum up the exponentiated values
  Zeros(ZsNormExpSum, m_mini_batch_size, 1); // Clear the entire matrix
  //  ColSumMat ZsNormExpSum;
  ColumnSum((DistMat&) *m_preactivations_v, ZsNormExpSum);
  Copy(ZsNormExpSum, ZsNormExpSumStar);

  // Divide each entry: exp(x_ij) / Sum_i(exp(x_ij))
  Copy(*m_preactivations_v, *m_activations_v);

  IndexDependentMap(*m_activations_v,
                    (std::function<DataType(Int,Int,DataType)>)([this /*ZsNormExpSum*/](Int r, Int c, DataType z)->
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

    // Convert forward and backward prop matrices to MC,MR formats
    DistMatrixReadProxy<DataType,DataType,MC,MR> DsNextProxy(*bp_input);
    DistMat& DsNext = DsNextProxy.Get();
    DistMatrixReadProxy<DataType,DataType,MC,MR> XProxy(*fp_input);
    DistMat& X = XProxy.Get();
    DistMat X_v;
    DistMat DsNext_v;
    int64_t cur_mini_batch_size = neural_network_model->get_current_mini_batch_size();
    View(X_v, X, IR(0, X.Height()), IR(0, cur_mini_batch_size));
    View(DsNext_v, DsNext, IR(0, DsNext.Height()), IR(0, cur_mini_batch_size));

    // delta = (activation - y)
    // delta_w = delta * activation_prev^T
    Copy(*m_activations_v, *m_prev_error_signal_v);
    Axpy(-1., DsNext_v, *m_prev_error_signal_v); // Per-neuron error

    // Compute the partial delta update for the next lower layer
    Gemm(TRANSPOSE, NORMAL, (DataType) 1., *m_weights_v, *m_prev_error_signal_v, (DataType) 0., *m_error_signal_v);

    if (m_execution_mode == execution_mode::training) {
      DataType avg_error = this->computeCost(DsNext_v);
      aggregate_cost += avg_error;
      num_backprop_steps++;
    }

    // by divide mini-batch size
    Gemm(NORMAL, TRANSPOSE, (DataType) 1.0/get_effective_minibatch_size(), *m_prev_error_signal_v,
         X_v, (DataType) 0., *m_weights_gradient_v);
}

DataType lbann::SoftmaxLayer::computeCost(const DistMat& Y) {
    // Compute the cost function
    // cost=-1/m*(sum(sum(groundTruth.*log(a3))))
    DataType avg_error = 0.0, total_error = 0.0;
    int64_t cur_mini_batch_size = neural_network_model->get_current_mini_batch_size();

    EntrywiseMap(*m_activations_v, (std::function<DataType(DataType)>)([](DataType z)->DataType{return log(z);}));

    Hadamard(Y, *m_activations_v, m_activations_cost_v);
    Zeros(m_minibatch_cost, m_mini_batch_size, 1); // Clear the entire array
    ColumnSum(m_activations_cost_v, m_minibatch_cost);

    // Sum the local, total error
    const Int local_height = m_minibatch_cost.LocalHeight();
    for(int r = 0; r < local_height; r++) {
      total_error += m_minibatch_cost.GetLocal(r, 0);
    }
    total_error = mpi::AllReduce(total_error, m_minibatch_cost.DistComm());

    avg_error = -1.0 * total_error / cur_mini_batch_size;
    return avg_error;
}

DataType lbann::SoftmaxLayer::WBL2norm() {
  DataType nrm2 = Nrm2(*WB);
  return nrm2 * nrm2;
}

inline DataType _sq(DataType x) { return (x * x); }
inline DataType _sqrt(DataType x) { return (1 / sqrt(x + 1e-8)); }

bool lbann::SoftmaxLayer::update()
{
  if(m_execution_mode == execution_mode::training) {
    optimizer->update_weight_bias_matrix(*WB_D, *WB);
  }
  return true;
}

void lbann::SoftmaxLayer::summarize(lbann_summary& summarizer, int64_t step) {
  Layer::summarize(summarizer, step);
  std::string tag = "layer" + std::to_string(static_cast<long long>(Index))
    + "/SoftmaxCost";
  summarizer.reduce_scalar(tag, avgCost(), step);
}

void lbann::SoftmaxLayer::epoch_print() const {
  double avg_cost = avgCost();
  if (comm->am_world_master()) {
    std::vector<double> avg_costs(comm->get_num_models());
    comm->intermodel_gather(avg_cost, avg_costs);
    for (size_t i = 0; i < avg_costs.size(); ++i) {
      std::cout << "Model " << i << " average softmax cost: " << avg_costs[i] <<
        std::endl;
    }
  } else {
    comm->intermodel_gather(avg_cost, comm->get_world_master());
  }
}

void lbann::SoftmaxLayer::epoch_reset() {
  Layer::epoch_reset();
  resetCost();
}

DataType lbann::SoftmaxLayer::checkGradient(Layer& PrevLayer, const DataType Epsilon)
{
  return 0.0;
}

void lbann::SoftmaxLayer::resetCost() {
  aggregate_cost = 0.0;
  num_backprop_steps = 0;
}

DataType lbann::SoftmaxLayer::avgCost() const {
  return aggregate_cost / num_backprop_steps;
}

bool lbann::SoftmaxLayer::saveToCheckpoint(int fd, const char* filename, uint64_t* bytes)
{
  ssize_t write_rc = write(fd, &aggregate_cost, sizeof(aggregate_cost));
  if (write_rc != sizeof(aggregate_cost)) {
    // error!
  }
  *bytes += write_rc;

  write_rc = write(fd, &num_backprop_steps, sizeof(num_backprop_steps));
  if (write_rc != sizeof(num_backprop_steps)) {
    // error!
  }
  *bytes += write_rc;

  return Layer::saveToCheckpoint(fd, filename, bytes);
}

bool lbann::SoftmaxLayer::loadFromCheckpoint(int fd, const char* filename, uint64_t* bytes)
{
  ssize_t read_rc = read(fd, &aggregate_cost, sizeof(aggregate_cost));
  if (read_rc != sizeof(aggregate_cost)) {
    // error!
  }
  *bytes += read_rc;

  read_rc = read(fd, &num_backprop_steps, sizeof(num_backprop_steps));
  if (read_rc != sizeof(num_backprop_steps)) {
    // error!
  }
  *bytes += read_rc;

  return Layer::loadFromCheckpoint(fd, filename, bytes);
}

bool lbann::SoftmaxLayer::saveToCheckpointShared(const char* dir, uint64_t* bytes)
{
  // get our rank
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // rank 0 writes softmax cost to file
  if (rank == 0) {
      // define the filename
      char file[1024];
      sprintf(file, "%s/SoftmaxCost_L%d", dir, Index);

      // open the file
      int fd = lbann::openwrite(file);
      if (fd != -1 ) {
          ssize_t write_rc = write(fd, &aggregate_cost, sizeof(aggregate_cost));
          if (write_rc != sizeof(aggregate_cost)) {
            // error!
          }
          *bytes += write_rc;

          write_rc = write(fd, &num_backprop_steps, sizeof(num_backprop_steps));
          if (write_rc != sizeof(num_backprop_steps)) {
            // error!
          }
          *bytes += write_rc;

          // close the file
          lbann::closewrite(fd, file);
      }
  }

  return Layer::saveToCheckpointShared(dir, bytes);
}

bool lbann::SoftmaxLayer::loadFromCheckpointShared(const char* dir, uint64_t* bytes)
{
    // get our rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // rank 0 writes softmax cost to file
    if (rank == 0) {
        // define the filename
        char file[1024];
        sprintf(file, "%s/SoftmaxCost_L%d", dir, Index);

        // open the file
        int fd = lbann::openread(file);
        if (fd != -1 ) {
            ssize_t read_rc = read(fd, &aggregate_cost, sizeof(aggregate_cost));
            if (read_rc != sizeof(aggregate_cost)) {
              // error!
            }
            *bytes += read_rc;

            read_rc = read(fd, &num_backprop_steps, sizeof(num_backprop_steps));
            if (read_rc != sizeof(num_backprop_steps)) {
              // error!
            }
            *bytes += read_rc;

            // close the file
            lbann::closeread(fd, file);
        }
    }

    // get values from rank 0
    MPI_Bcast(&aggregate_cost, 1, DataTypeMPI, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_backprop_steps, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    return Layer::loadFromCheckpointShared(dir, bytes);
}
