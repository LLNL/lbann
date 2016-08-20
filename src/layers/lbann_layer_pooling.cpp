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
// lbann_layer_pooling .hpp .cpp - Pooling layer
////////////////////////////////////////////////////////////////////////////////

#include "lbann/layers/lbann_layer_pooling.hpp"

using namespace std;
using namespace El;
using namespace lbann;

pooling_layer::pooling_layer(const uint index,
                             const int  num_dims,
                             const int  num_channels,
                             const int* input_dims,
                             const int* pool_dims,
                             const int* pool_pads,
                             const int* pool_strides,
                             const int  pool_mode,
                             const uint mini_batch_size,
                             activation_type activation,
                             lbann_comm* comm,
                             std::vector<regularizer*> regs,
                             cudnn::cudnn_manager* cudnn)
  : Layer(index, comm, NULL, mini_batch_size, activation, regs),
    m_pool_mode(pool_mode),
    m_num_dims(num_dims), m_num_channels(num_channels)
{

  // Initialize input dimensions and pooling parameters
  m_input_dims.resize(num_dims);
  m_pool_dims.resize(num_dims);
  m_pool_pads.resize(num_dims);
  m_pool_strides.resize(num_dims);
  for(int i=0; i<num_dims; ++i) {
    m_input_dims[i] = input_dims[i];
    m_pool_dims[i] = pool_dims[i];
    m_pool_pads[i] = pool_pads[i];
    m_pool_strides[i] = pool_strides[i];
  }

  // Calculate output dimensions
  m_output_dims.resize(num_dims);
  NumNeurons = num_channels;
  for(int i=0; i<num_dims; ++i) {
    m_output_dims[i] = input_dims[i]+2*pool_pads[i]-pool_dims[i]+1;
    m_output_dims[i] = (m_output_dims[i]+pool_strides[i]-1)/pool_strides[i];
    NumNeurons *= m_output_dims[i];
  }
  
  // Matrices should be in Star,VC distributions
  delete Zs;
  delete Ds;
  delete Ds_Temp;
  delete Acts;
  Zs = new StarVCMat(comm->get_model_grid());
  Ds = new StarVCMat(comm->get_model_grid());
  Ds_Temp = new StarVCMat(comm->get_model_grid());
  Acts = new StarVCMat(comm->get_model_grid());

  // Initialize cuDNN pooling layer
  m_cudnn_layer = NULL;
#ifdef __LIB_CUDNN
  if(cudnn)
    m_cudnn_layer = new cudnn::cudnn_pooling_layer(num_dims,
                                                   num_channels,
                                                   input_dims,
                                                   pool_mode,
                                                   pool_dims,
                                                   pool_pads,
                                                   pool_strides,
                                                   cudnn);
#endif // __LIB_CUDNN

}

pooling_layer::~pooling_layer()
{
#ifdef __LIB_CUDNN
  delete m_cudnn_layer;
#endif // __LIB_CUDNN
}

void pooling_layer::setup(const int num_prev_neurons)
{
  Layer::setup(num_prev_neurons);

#ifdef __LIB_CUDNN
  if(m_cudnn_layer) {
    // Setup cuDNN pooling layer
    m_cudnn_layer->setup();

    // Get output dimensions
    NumNeurons = m_cudnn_layer->m_dst_size;
    for(int i=0; i<m_num_dims; ++i)
      m_output_dims[i] = m_cudnn_layer->m_dst_dims[i+2];
  }
#endif // __LIB_CUDNN

  // Check if input dimensions are valid
  int num_inputs = m_num_channels;
  for(int i=0; i<m_num_dims; ++i)
    num_inputs *= m_input_dims[i];
  if(num_inputs != num_prev_neurons) {
    std::cerr << "Error: pooling layer input dimensions don't match number of input neurons\n";
    exit(EXIT_FAILURE);
  }

  // Initialize matrices
  Ones(*Zs, NumNeurons+1, m_mini_batch_size);
  Zeros(*Ds, NumNeurons+1, m_mini_batch_size);
  Zeros(*Ds_Temp, num_prev_neurons+1, m_mini_batch_size);
  Ones(*Acts, NumNeurons+1, m_mini_batch_size);

}

void lbann::pooling_layer::fp_linearity(ElMat& _WB,
                                        ElMat& _X,
                                        ElMat& _Z,
                                        ElMat& _Y) {
  
  // Convert matrices to desired formats
  DistMatrixReadProxy<DataType,DataType,STAR,VC> XProxy(_X);
  DistMatrixWriteProxy<DataType,DataType,STAR,VC> ZProxy(_Z);
  DistMatrixWriteProxy<DataType,DataType,STAR,VC> YProxy(_Y);
  StarVCMat& X = XProxy.Get();
  StarVCMat& Z = ZProxy.Get();
  StarVCMat& Y = YProxy.Get();

  // Get local matrices
  const Mat& XLocal = X.LockedMatrix();
  Mat& ZLocal = Z.Matrix();
  Mat& YLocal = Y.Matrix();

  // Apply pooling on local data samples
  if(m_cudnn_layer) {
#ifdef __LIB_CUDNN
    m_cudnn_layer->forward(XLocal, ZLocal);
#else
    std::cerr << "Error: cuDNN not detected\n";
    exit(EXIT_FAILURE);
#endif
  }
  else {
    

    // TODO: implement pooling on CPU
    std::cerr << "Error: pooling forward pass not implemented on CPU\n";
    exit(EXIT_FAILURE);
  }

  // Z and Y are identical after fp linearity step
  Copy(ZLocal, YLocal);

}

void lbann::pooling_layer::bp_linearity() {

  // Convert matrices to desired formats
  DistMatrixReadProxy<DataType,DataType,STAR,VC> InputProxy(*fp_input); // TODO: store from fp step
  StarVCMat& Input = InputProxy.Get();

  // Get local matrices
  const Mat& InputLocal = Input.LockedMatrix();
  const Mat& OutputLocal = Acts->LockedMatrix();
  const Mat& OutputDeltaLocal = Ds->LockedMatrix();
  Mat& InputDeltaLocal = Ds_Temp->Matrix();

  // Compute gradients on local data samples
  if(m_cudnn_layer) {
#ifdef __LIB_CUDNN
    m_cudnn_layer->backward(InputLocal,
                            OutputLocal,
                            OutputDeltaLocal,
                            InputDeltaLocal);
#else
    std::cerr << "Error: cuDNN not detected\n";
    exit(EXIT_FAILURE);
#endif
  }
  else {
    // TODO: implement backward pass on CPU
    std::cerr << "Error: pooling backward pass not implemented on CPU\n";
    exit(EXIT_FAILURE);
  }
  
}

bool pooling_layer::update()
{
  return true;
}
