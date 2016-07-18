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
// lbann_layer_convolutional .hpp .cpp - Convolutional Layer
// 07/06/2016: changing distributed matrices to STAR,VC format
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_CONVOLUTIONAL_HPP_INCLUDED
#define LBANN_LAYER_CONVOLUTIONAL_HPP_INCLUDED

#include "lbann/layers/lbann_layer.hpp"
#include "lbann/layers/lbann_layer_activations.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include <string>

namespace lbann
{

    // ConvolutionalLayer: convolutional/pool layer class
    class ConvolutionalLayer: public Layer
    {
    public:
      ConvolutionalLayer(uint index, int _ConvDim,
                         int _InputChannels, const int* _InputDims,
                         int _OutputChannels, const int* _FilterDims,
                         uint miniBatchSize,
                         lbann_comm* comm, Optimizer *optimizer);
        ~ConvolutionalLayer();

        void setup(CudnnNet<DataType> *cudnnNet);

        bool update();

    public:
        int ConvDim;
        int InputChannels;
        int InputDims[3];    // 1D (W), 2D (HW), or 3D (DHW)
        int OutputChannels;
        int OutputDims[3];   // 1D (W), 2D (HW), or 3D (DHW)
        int FilterDims[3];   // 1D (W), 2D (HW), or 3D (DHW)

#ifdef __LIB_CUDNN
        CudnnLayer<DataType>* cudnnLayer;
#endif

    protected:
      void fp_linearity(ElMat& _WB, ElMat& _X, ElMat& _Z, ElMat& _Y);
      void bp_linearity();

    };
}


#endif // LBANN_LAYER_CONVOLUTIONAL_HPP_INCLUDED
