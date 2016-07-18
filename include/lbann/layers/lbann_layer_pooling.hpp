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
// lbann_layer_pooling .hpp .cpp - Pooling Layer (max, average)
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_POOLING_HPP_INCLUDED
#define LBANN_LAYER_POOLING_HPP_INCLUDED

#include "lbann/layers/lbann_layer.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include <string>

namespace lbann
{
    class Sequential;

    class PoolingLayer: public Layer
    {
    public:
        PoolingLayer(uint index, int poolDim, int channels,
                     const int* inputDims, const int* poolWindowdim,
                     const int* poolPad, const int* poolStride,
                     int poolMode, uint miniBatchSize,
                     lbann_comm* comm, Optimizer *optimizer);
        ~PoolingLayer();

        void setup(CudnnNet<DataType> *cudnnNet);

        bool update();

    public:
        int PoolDim;            // 1D, 2D, or 3D (NCDHW)
        int Channels;
        int InputDims[3];     // 1D (W), 2D (HW), or 3D (DHW)
        int OutputDims[3];    // 1D (W), 2D (HW), or 3D (DHW)
        int PoolWindowDim[3];   // window dimension, eg., (2, 2, 2)
        int PoolPad[3];        // padding, eg., (0, 0, 0)
        int PoolStride[3];     // strides, e.g., (2, 2, 2)
        int PoolMode;           // 0: max, 1: average

#ifdef __LIB_CUDNN
        CudnnLayer<DataType>* cudnnLayer;
#endif

    protected:
      void fp_linearity(ElMat& _WB, ElMat& _X, ElMat& _Z, ElMat& _Y);
      void bp_linearity();

    };
}


#endif // LBANN_LAYER_POOLING_HPP_INCLUDED
