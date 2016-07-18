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
// lbann_layer_activations .hpp .cpp - Basic activations: sigmoid, tanh, reLU
////////////////////////////////////////////////////////////////////////////////

#include "lbann/layers/lbann_layer_activations.hpp"
#include "lbann/utils/lbann_exception.hpp"

using namespace std;
using namespace El;

namespace lbann {

Activation* new_activation(activation_type act_fn) {
  switch (act_fn) {
  case activation_type::SIGMOID:
    return new sigmoid_layer();
  case activation_type::TANH:
    return new tanh_layer();
  case activation_type::RELU:
    return new reLU_layer();
  case activation_type::ID:
    return new id_layer();
  default:
    throw lbann_exception("Unsupported activation type.");
  }
  return nullptr;  // Never reached.
}

// Activation class
DataType sigmoid_layer::sigmoid(DataType z)
{
    return (1.0 / (1.0 + exp(-z)));
}

DataType sigmoid_layer::sigmoidPrime(DataType z)
{
    DataType sigz = sigmoid(z);
    return sigz * (1 - sigz);
}

DataType tanh_layer::tanh(DataType z)
{
#ifdef __ICC
    // If using Intel compiler, use the MKL specific Tanh function
    return Tanh(z);
#else
    // Otherwise force the system to use the C++ version - glibc version is having problems with memory leaks
    return std::tanh(z);
#endif
}

DataType tanh_layer::tanhPrime(DataType z)
{
    float e = exp(2 * z);
    return ((e - 1) / (e + 1));
}

DataType reLU_layer::reLU(DataType z)
{
    return max((DataType) 0.0, z);
}

DataType reLU_layer::reLUPrime(DataType z)
{
    if (z > 0.0) {
      return 1.0;
    }else {
      return 0.0;
    }
}

////////////////////////////////////////////////////////////////////////////////
// There are three mechanisms for updating all fields of a distributed matrix
// 1) EntrywiseMap + independent reset of the bias row - Fastest (about ~30% faster)
// 2) IndexDependentMap (conditional application of function to mask bias row) - Slow
//        IndexDependentMap(m, 
//          (std::function<double(int,int,double)>)[m](int r, int c, double z)->
//          double{int bias_row = m.Height(); 
//                 if(r == bias_row - 1){return 1.0;}else{return sigmoid(z);}
//                 });
// 3) Dual nested loop over local matrix indices with global matrix lookups - Slow
////////////////////////////////////////////////////////////////////////////////
void sigmoid_layer::forwardProp(ElMat& m)
{
  const Int ngrows = m.Height();
  const Int nrows = m.LocalHeight();
  const Int ncols = m.LocalWidth();
  EntrywiseMap(m, std::function<DataType(DataType)>(sigmoid));

  const Int r = nrows-1;
  const Int gr = m.GlobalRow(r);
  if(gr == ngrows - 1) { // Bias row
    for (int c = 0; c < ncols; c++) {
      m.SetLocal(r, c, 1.0); // Set the bias row back to 1.0
    }
  }
}

void sigmoid_layer::backwardProp(ElMat& m)
{
  const Int ngrows = m.Height();
  const Int nrows = m.LocalHeight();
  const Int ncols = m.LocalWidth();
  EntrywiseMap(m, std::function<DataType(DataType)>(sigmoidPrime));

  const Int r = nrows-1;
  const Int gr = m.GlobalRow(r);
  if(gr == ngrows - 1) { // Bias row
    for (int c = 0; c < ncols; c++) {
      m.SetLocal(r, c, 0.0); // Set the bias row back to 0.0
    }
  }
}

void tanh_layer::forwardProp(ElMat& m)
{
  const Int ngrows = m.Height();
  const Int nrows = m.LocalHeight();
  const Int ncols = m.LocalWidth();
  EntrywiseMap(m, std::function<DataType(DataType)>(tanh));

  const Int r = nrows-1;
  const Int gr = m.GlobalRow(r);
  if(gr == ngrows - 1) { // Bias row
    for (int c = 0; c < ncols; c++) {
      m.SetLocal(r, c, 1.0); // Set the bias row back to 1.0
    }
  }
}

void tanh_layer::backwardProp(ElMat& m)
{
  const Int ngrows = m.Height();
  const Int nrows = m.LocalHeight();
  const Int ncols = m.LocalWidth();
  EntrywiseMap(m, std::function<DataType(DataType)>(tanhPrime));

  const Int r = nrows-1;
  const Int gr = m.GlobalRow(r);
  if(gr == ngrows - 1) { // Bias row
    for (int c = 0; c < ncols; c++) {
      m.SetLocal(r, c, 0.0); // Set the bias row back to 0.0
    }
  }
}

void reLU_layer::forwardProp(ElMat& m)
{
    const Int ngrows = m.Height();
    const Int ngcols = m.Width();
    const Int nrows = m.LocalHeight();
    const Int ncols = m.LocalWidth();

    EntrywiseMap(m, std::function<DataType(DataType)>(reLU));

    const Int r = nrows-1;
    const Int gr = m.GlobalRow(r);
    if(gr == ngrows - 1) { // Bias row
        for (int c = 0; c < ncols; c++) {
            m.SetLocal(r, c, 1.0); // Set the bias row back to 1.0
        }
    }
}

void reLU_layer::backwardProp(ElMat& m)
{
  const Int ngrows = m.Height();
  const Int nrows = m.LocalHeight();
  const Int ncols = m.LocalWidth();

  EntrywiseMap(m, std::function<DataType(DataType)>(reLUPrime));

  const Int r = nrows-1;
  const Int gr = m.GlobalRow(r);
  if(gr == ngrows - 1) { // Bias row
    for (int c = 0; c < ncols; c++) {
      m.SetLocal(r, c, 0.0); // Set the bias row back to 0.0
    }
  }
}

}  // namespace lbann
