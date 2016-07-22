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
// [W0 ...   B0]
// [|         |]
// [Wn       Bn]
// [0  ...  0 1] - Initialize the final row to be all zeros and 1 in the bias to properly
//                 set the bias for the next layer
// WB_D structure:
// [dW     dB]
// [0 ... 0 0]
// D structure:
// [D        ]
// [0 ... 0 0]
// Z, Zs, Act, Acts structure:
// [Acts     ]
// [1 ... 1 1]

lbann::FullyConnectedLayer::FullyConnectedLayer(
  const uint index, const int numPrevNeurons, const uint numNeurons,
  uint miniBatchSize, activation_type activationType,
  lbann_comm* comm, Optimizer *optimizer,
  std::vector<regularizer*> regs)
  : Layer(index, comm, optimizer, miniBatchSize, regs),
    WB_view(comm->get_model_grid()), WB_D_view(comm->get_model_grid()),
    Acts_view(comm->get_model_grid())
{
    Index = index;
    NumNeurons = numNeurons;
    ActivationType = activationType;
    WBL2NormSum = 0.0;

    activation_fn = new_activation(activationType);
}

lbann::FullyConnectedLayer::FullyConnectedLayer(
  const uint index, const int numPrevNeurons, const uint numNeurons,
  uint miniBatchSize, activation_type activationType,
  lbann_comm* comm, Optimizer *optimizer) :
  FullyConnectedLayer(index, numPrevNeurons, numNeurons, miniBatchSize, activationType,
                      comm, optimizer, {}) {}

lbann::FullyConnectedLayer::~FullyConnectedLayer() {
  delete activation_fn;
}

void lbann::FullyConnectedLayer::setup(int numPrevNeurons) {
  Layer::setup(numPrevNeurons);
    if(optimizer != NULL) {
      optimizer->setup(numPrevNeurons+1, NumNeurons+1);
    }

    // Xavier random initialization - Derived from Caffe implementation
    DataType var_scale = sqrt(3.0 / (numPrevNeurons + 1));

    if (numPrevNeurons != -1) {
        Gaussian(*WB, NumNeurons + 1, numPrevNeurons + 1, (DataType) 0.0, var_scale); // use var_scale, instead of 0.1
        if (comm->am_model_master()) {
          cout << "Fully Connected Layer " << Index << ": Xavier initialization: input size=" << (numPrevNeurons + 1) << " scale=" << var_scale << " and layer size " << NumNeurons << endl;
        }
        // Set the last row to all zeros and a 1 in the last column to set the bias term for
        // activation layer: 0 0 0 0 ... 0 1
        Int ngrows = WB->Height();
        Int ngcols = WB->Width();
        Int nrows = WB->LocalHeight();
        Int ncols = WB->LocalWidth();
        Int r = nrows-1;
        Int gr = WB->GlobalRow(r);
        Int gc = WB->GlobalCol(ncols-1);
        if(gr == ngrows - 1) { // Bias initialization row
          for (int c = 0; c < ncols; c++) {
            WB->SetLocal(r, c, 0.0); // Set the bias row back to 0.0
          }
          if(gc == ngcols-1) {
            WB->SetLocal(r, ncols-1, 1.0); // and 1.0
          }
        }
        Zeros(*WB_D, NumNeurons + 1, numPrevNeurons + 1);
        Zeros(*Ds, NumNeurons + 1, m_mini_batch_size);
        Zeros(*Ds_Temp, numPrevNeurons + 1, m_mini_batch_size); // Ds_Temp holds the product of WB^T * Ds
        Zeros(*Zs, NumNeurons + 1, m_mini_batch_size);
        View(WB_view, *WB, IR(0, WB->Height() - 1), IR(0, WB->Width()));
        View(WB_D_view, *WB_D, IR(0, WB_D->Height() - 1), IR(0, WB_D->Width()));
    }
    Zeros(*Acts, NumNeurons + 1, m_mini_batch_size);
    View(Acts_view, *Acts, IR(0, Acts->Height() - 1), IR(0, Acts->Width()));

#if 0
    printf("Layer[%d] has %d neurons and %d inputs\n", Index, NumNeurons + 1, numPrevNeurons + 1);
    printf("trainMB have allocated Layers[%d]->Acts %d entries (%d x %d)\n", Index, Acts->AllocatedMemory(), Acts->Height(), Acts->Width());
    printf("trainMB have allocated Layers[%d]->WB %d entries (%d x %d)\n", Index, WB->AllocatedMemory(), WB->Height(), WB->Width());
    printf("trainMB have allocated Layers[%d]->WB_D %d entries (%d x %d)\n", Index, WB_D->AllocatedMemory(), WB_D->Height(), WB_D->Width());
    printf("trainMB have allocated Layers[%d]->Ds %d entries (%d x %d)\n", Index, Ds->AllocatedMemory(), Ds->Height(), Ds->Width());
    printf("trainMB have allocated Layers[%d]->Ds_Temp %d entries (%d x %d)\n", Index, Ds_Temp.AllocatedMemory(), Ds_Temp.Height(), Ds_Temp.Width());
#endif
}

void lbann::FullyConnectedLayer::fp_linearity(ElMat& _WB, ElMat& _X, ElMat& _Z, ElMat& _Y)
{
  // Convert matrices to desired format
  DistMatrixReadProxy<DataType,DataType,MC,MR> WBProxy(_WB);
  DistMatrixReadProxy<DataType,DataType,MC,MR> XProxy(_X); // TODO: Store for bp step
  DistMatrixWriteProxy<DataType,DataType,MC,MR> ZProxy(_Z);
  DistMatrixWriteProxy<DataType,DataType,MC,MR> YProxy(_Y);
  DistMat& WB = WBProxy.Get();
  DistMat& X = XProxy.Get();
  DistMat& Z = ZProxy.Get();
  DistMat& Y = YProxy.Get();

  // Apply forward prop linearity
  Gemm(NORMAL, NORMAL, (DataType) 1., WB, X, (DataType) 0., Z);
  Copy(Z, Y);
}

void lbann::FullyConnectedLayer::bp_linearity()
{
    // Convert forward and backward prop matrices to MC,MR format
    DistMatrixReadProxy<DataType,DataType,MC,MR> DsNextProxy(*bp_input);
    DistMatrixReadProxy<DataType,DataType,MC,MR> XProxy(*fp_input); // TODO: store from fp step
    DistMat& DsNext = DsNextProxy.Get();
    DistMat& X = XProxy.Get();
    
    // Compute the delta using the results from "next" deeper layer
    Hadamard(DsNext, *Zs, *Ds);
    // Compute the partial delta update for the next lower layer
    Gemm(TRANSPOSE, NORMAL, (DataType) 1., *WB, DsNext, (DataType) 0., *Ds_Temp);
    // BVE - an alternative approach is to compute the mean 
    Gemm(NORMAL, TRANSPOSE, (DataType) 1.0/get_effective_minibatch_size(), *Ds,
         X, (DataType) 0., *WB_D);
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
  DataType nrm2 = Nrm2(*WB);
  return nrm2 * nrm2;
}

inline DataType _sq(DataType x) { return (x * x); }
inline DataType _sqrt(DataType x) { return (1 / sqrt(x + 1e-8)); }

bool lbann::FullyConnectedLayer::update(/*const float LearnRate, const int LearnRateMethod, const DataType DecayRate*/)
{
  optimizer->update_weight_bias_matrix(*WB_D, *WB);
  WBL2NormSum = 0.0;
  return true;
}

DataType lbann::FullyConnectedLayer::checkGradient(Layer& PrevLayer, const DataType Epsilon)
{
    DistMat WB_E1(WB->Grid());
    DistMat WB_E2(WB->Grid());
    DistMat Zs_E1(Zs->Grid());
    DistMat Zs_E2(Zs->Grid());
    DistMat Acts_E1(Acts->Grid());
    DistMat Acts_E2(Acts->Grid());
    DataType grad_diff = 0;
    DataType grad_sum = 0;

    Zeros(WB_E1, WB->Height(), WB->Width());
    Zeros(WB_E2, WB->Height(), WB->Width());
    Zeros(Zs_E1, Zs->Height(), Zs->Width());
    Zeros(Zs_E2, Zs->Height(), Zs->Width());
    Zeros(Acts_E1, Acts->Height(), Acts->Width());
    Zeros(Acts_E2, Acts->Height(), Acts->Width());

    Copy(*WB, WB_E1);
    Copy(*WB, WB_E2);

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
    for (int row = 0; row < WB->Height(); row++) {
        for (int col = 0; col < WB->Width(); col++) {
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
            this->fp_linearity(*WB, *(PrevLayer.Acts), *Zs, *Acts);
            this->fp_nonlinearity();

            // J(thetaPlus(i))
            this->fp_linearity(WB_E1, *(PrevLayer.Acts), Zs_E1, Acts_E1);
            this->fp_nonlinearity();

            // J(thetaMinus(i))
            this->fp_linearity(WB_E2, *(PrevLayer.Acts), Zs_E2, Acts_E2);
            this->fp_nonlinearity();

            //            this->getCost();

            // gradApprox(i) = J(thetaPlus(i)) - J(thetaMinus(i)) / (2*Epsilon)
            Axpy(-1.0, *Acts, Acts_E1);
            Axpy(-1.0, *Acts, Acts_E2);

            bool bad_E1 = false, bad_E2 = false;
            for(int r = 0; r < Acts->Height(); r++) {
              for(int c = 0; c < Acts->Width(); c++) {
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
              Print(*Acts);
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
