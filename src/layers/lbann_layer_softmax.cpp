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
#include <unistd.h>

using namespace std;
using namespace El;

lbann::SoftmaxLayer::SoftmaxLayer(const uint index, const int numPrevNeurons, const uint numNeurons,
                                  uint miniBatchSize, lbann_comm* comm, Optimizer *optimizer)
  :  Layer(index, comm, optimizer, miniBatchSize)
   , ZsColMax(comm->get_model_grid()), ZsNormExpSum(comm->get_model_grid()),
     norms(comm->get_model_grid()), ZsColMaxStar(comm->get_model_grid()),
     ZsNormExpSumStar(comm->get_model_grid()), Acts_Cost(comm->get_model_grid())
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

    // Xavier random initialization - Derived from Caffe implementation
    DataType var_scale = sqrt(3.0 / (numPrevNeurons + 1));

    if (numPrevNeurons != -1) {
      // For the softmax layer we do not want to have an extra row propagating the bias term to the output
        Gaussian(*WB, NumNeurons, numPrevNeurons + 1, (DataType) 0.0, var_scale);
        if (comm->am_model_master()) {
          cout << "Softmax Layer " << Index << ": Xavier initialization: input size=" << (numPrevNeurons + 1) << " scale=" << var_scale << " and layer size " << NumNeurons << endl;
        }
        Zeros(*WB_D, NumNeurons, numPrevNeurons + 1);
        Zeros(*Ds, NumNeurons, m_mini_batch_size);
        Zeros(*Ds_Temp, numPrevNeurons + 1, m_mini_batch_size); // Ds_Temp holds the product of WB^T * Ds
        Zeros(*Zs, NumNeurons, m_mini_batch_size);
    }
    Zeros(*Acts, NumNeurons, m_mini_batch_size);
    Zeros(Acts_Cost, NumNeurons, m_mini_batch_size);
}

// template <typename Dtype>
// void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//     vector<Blob<Dtype>*>* top) {

void lbann::SoftmaxLayer::fp_linearity(ElMat& _WB, ElMat& _X, ElMat& _Z, ElMat& _Y)
{
  // _Z = WB * Xs                                               -- Xs is previous layer Activations
  // ZsColMax[c,0] = max(_Z[0..numNeurons-1, c])                -- (m_mini_batch_size x 1)
  // ZsNorm[r,c] = _Z[r,c] - ZsColMax[c,0]                      -- Column-wise normalized Zs matrix: _Z[r,c] - ZsColMax[1,c]
  // ZsNormExp[r,c] = exp(ZsNorm[r,c])
  // ZsNormExpSum[c,0] = sum(ZsNormExp[0..numNeurons-1, c])     -- Column-wise sum over normalized, exponentiated _Z
  // _Y[r,c] = ZsNormExp[r,c] / ZsNormExpSum[c,0]               -- exp(norm(_Z[r,c])) = Sum(exp(norm(Zs[r,c])))

  // Convert forward prop matrix to MC,MR format
  // TODO: store this matrix for back prop
  DistMatrixReadProxy<DataType,DataType,MC,MR> XProxy(_X);
  DistMat& X = XProxy.Get();

  // Apply linear transform
  Gemm(NORMAL, NORMAL, (DataType) 1.0, _WB, X, (DataType) 0.0, _Z);

  // For each minibatch (column) find the maximimum value
  Zeros(ZsColMax, m_mini_batch_size, 1);
  ColumnMax((DistMat&) _Z, ZsColMax);

  // Redistribute the per-minibatch maximum values
  Copy(ZsColMax, ZsColMaxStar);

  // Compute exp(z) of each entry. Subtract the max of each column from its
  // entries to prevent the exp from blowing up. Large negative values are
  // expected to underflow to 0.
  IndexDependentMap(
    _Z,
    (std::function<DataType(int,int,DataType)>)
    ([this](int r, int c, DataType z)->DataType {
      Int rL = this->ZsColMaxStar.LocalRow(c);
      return std::exp(z - this->ZsColMaxStar.GetLocal(rL, 0));
    }));

  // For each minibatch (column) sum up the exponentiated values
  Zeros(ZsNormExpSum, m_mini_batch_size, 1);
  //  ColSumMat ZsNormExpSum;
  ColumnSum((DistMat&) _Z /*ZsNormExp*/, ZsNormExpSum);
  Copy(ZsNormExpSum, ZsNormExpSumStar);

  // Divide each entry: exp(x_ij) / Sum_i(exp(x_ij))
  Copy(_Z /*ZsNormExp*/, _Y);
  IndexDependentMap(_Y,
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

    // delta = (activation - y)
    // delta_w = delta * activation_prev^T
    Copy(*Acts, *Ds);
    Axpy(-1., DsNext, *Ds); // Per-neuron error

    // Compute the partial delta update for the next lower layer
    Gemm(TRANSPOSE, NORMAL, (DataType) 1., *WB, *Ds, (DataType) 0., *Ds_Temp);

    // Compute and display the cost function
    DistMat Y(Acts->Grid());
    Copy(DsNext, Y);

    DataType avg_error = this->computeCost(Y);// + Lambda/2 * WBL2NormSum;
    // DataType cost = (aggregate_cost / num_backprop_steps);
    // DataType delta = 0.01 * cost;
    // if(Output.Grid().Rank() == 0 && (1 || avg_error <= (cost-delta) || avg_error > (cost+delta))) {
    //   cout << "Average of the softmax cost function across the mini-batch " << avg_error << endl;
    // }
    aggregate_cost += avg_error;
    num_backprop_steps++;

    // by divide mini-batch size
    Gemm(NORMAL, TRANSPOSE, (DataType) 1.0/get_effective_minibatch_size(), *Ds,
         X, (DataType) 0., *WB_D);
}

DataType lbann::SoftmaxLayer::computeCost(DistMat& Y) {
    // Compute the cost function
    // cost=-1/m*(sum(sum(groundTruth.*log(a3))))
    DataType avg_error = 0.0, total_error = 0.0;
    EntrywiseMap(*Acts, (std::function<DataType(DataType)>)([](DataType z)->DataType{return log(z);}));

    // DistMat Y(Acts.Grid());
    // Copy(Output, Y);
    Hadamard(Y, *Acts, Acts_Cost);
    ColSumMat MBCost(m_mini_batch_size, 1, Acts->Grid());
    Zeros(MBCost, m_mini_batch_size, 1);
    ColumnSum(Acts_Cost, MBCost);

    int c = 0;
    // Sum the local, total error
    for(int r = 0; r < MBCost.LocalHeight(); r++) {
      total_error += MBCost.GetLocal(r,c);
    }
    total_error = mpi::AllReduce(total_error, MBCost.DistComm());

    avg_error = -1.0 * total_error / m_mini_batch_size;
    return avg_error;
}

DataType lbann::SoftmaxLayer::WBL2norm() {
  DataType nrm2 = Nrm2(*WB);
  return nrm2 * nrm2;
}

inline DataType _sq(DataType x) { return (x * x); }
inline DataType _sqrt(DataType x) { return (1 / sqrt(x + 1e-8)); }

bool lbann::SoftmaxLayer::update(/*const float LearnRate, const int LearnRateMethod, const DataType DecayRate*/)
{
  optimizer->update_weight_bias_matrix(*WB_D, *WB);
  WBL2NormSum = 0.0;
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
