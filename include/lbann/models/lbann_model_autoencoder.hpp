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
// lbann_model_stacked_autoencoder .hpp .cpp - Stacked autoencoder for
//                                             sequential neural network models
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_MODEL_AUTOENCODER_HPP
#define LBANN_MODEL_AUTOENCODER_HPP

#include "lbann/models/lbann_model_sequential.hpp"
#include "lbann/layers/lbann_layer.hpp"
#include "lbann/layers/lbann_layer_fully_connected.hpp"
#include "lbann/layers/lbann_layer_softmax.hpp"
#include <vector>
#include <string>



namespace lbann
{
    // stacked_autoenocer : auto encoder class with greedy layer-wise training
  class AutoEncoder : public Sequential
  {
  public:
    AutoEncoder(Optimizer_factory *optimizer_factory, const uint MiniBatchSize, lbann_comm* comm);
    AutoEncoder(Optimizer_factory *optimizer_factory, const uint MiniBatchSize, lbann_comm* comm,
                const std::vector<int>& EncoderLayerDim, const std::vector<int>& DecoderLayerDim,
				const bool LayerMirror, const uint ActivationType,
				const float DropOut, const double lambda);
    ~AutoEncoder();

    void train(CircMat& X, const float LearnRate);
    void test(CircMat& X, CircMat& XP);
    void test(CircMat& X, CircMat& YP, CircMat& XP);
    void global_update();
    void train(DataReader* Dataset, int NumEpoch);
    void trainBatch(DataReader* Dataset);

    DataType evaluate(DataReader* Dataset, int ErrorType);
    DataType evaluateBatch(DataReader* Dataset, int ErrorType);

  public:
    std::vector<Layer*>    DecoderLayers;

  };
}
#endif  // LBANN_MODEL_AUTOENCODER_HPP
