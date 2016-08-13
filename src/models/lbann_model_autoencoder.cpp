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
// lbann_model_autoencoder .hpp .cpp - Autoencoder for sequential neural network models
////////////////////////////////////////////////////////////////////////////////

#include "lbann/models/lbann_model_autoencoder.hpp"
#include "lbann/models/lbann_model_dnn.hpp"
#include "lbann/layers/lbann_layer_fully_connected.hpp"
#include "lbann/layers/lbann_layer_softmax.hpp"
#include "lbann/optimizers/lbann_optimizer.hpp"
#include "lbann/optimizers/lbann_optimizer_sgd.hpp"
#include "lbann/optimizers/lbann_optimizer_adagrad.hpp"
#include "lbann/optimizers/lbann_optimizer_rmsprop.hpp"
#include <string>

using namespace std;
using namespace El;


////////////////////////////////////////////////////////////////////////////////
// stacked_autoencoder : main auto encoder class
////////////////////////////////////////////////////////////////////////////////

lbann::AutoEncoder::AutoEncoder(Optimizer_factory *optimizer_factory, const uint MiniBatchSize, lbann_comm* comm)
	: Sequential(optimizer_factory, MiniBatchSize, comm)
{
}

lbann::AutoEncoder::AutoEncoder(Optimizer_factory *optimizer_factory, const uint MiniBatchSize, lbann_comm* comm,
				const std::vector<int>& EncoderLayerDim, const std::vector<int>& DecoderLayerDim,
				const bool LayerMirror, const uint ActivationType,
				const float DropOut, const double lambda)
  : Sequential(optimizer_factory, MiniBatchSize, comm)
{
  // initalize layers
  for (int l = 0; l < (int)EncoderLayerDim.size(); l++) {
    Layers.push_back(new FullyConnectedLayer(Layers.size(), ((l == 0) ? -1 : EncoderLayerDim[l-1]),
                     EncoderLayerDim[l], MiniBatchSize, ActivationType, DropOut, lambda, NULL, comm));
  }

  if (LayerMirror) {
    for (int l = (int)EncoderLayerDim.size() - 2; l >= 0; l--) {
      DecoderLayers.push_back(new FullyConnectedLayer(DecoderLayers.size(), EncoderLayerDim[l+1], EncoderLayerDim[l],
                              MiniBatchSize, ActivationType, DropOut, lambda, NULL, comm));
    }
  }
  else {
    for (int l = 0; l < (int)DecoderLayerDim.size(); l++) {
      DecoderLayers.push_back(new FullyConnectedLayer(DecoderLayers.size(), ((l == 0) ? -1 : DecoderLayerDim[l-1]),
                              DecoderLayerDim[l], MiniBatchSize, ActivationType, DropOut,
                              lambda, NULL, comm));
    }
  }
}

lbann::AutoEncoder::~AutoEncoder()
{
    // free decoder layers
    for (size_t l = 0; l < DecoderLayers.size(); l++)
         delete DecoderLayers[l];
}

#if 0
void lbann::AutoEncoder::add(string layerName, int LayerDim, int ActivationType, int DropOut, double lambda)
{
  int prevLayerDim = -1;
  int layerIndex = Layers.size();
  int prevLayerIndex = -1;
  Optimizer *optimizer = optimizer_factory->create_optimizer();

  if(Layers.size() != 0) {
    Layer *prev = Layers.back();
    prevLayerDim = prev->NumNeurons;
    prevLayerIndex = prev->Index;
  }

  cout << "Adding a layer with input " << prevLayerDim << " and index " << layerIndex << " preve layer index " << prevLayerIndex << endl;

  if(layerName.compare("FullyConnected") == 0) {
    // initalize neural network (layers)
    Layers.push_back(new FullyConnectedLayer(layerIndex, prevLayerDim, LayerDim, MiniBatchSize, ActivationType, DropOut, lambda, optimizer, comm));
  }else if(layerName.compare("SoftMax") == 0) {
    Layers.push_back(new SoftmaxLayer(layerIndex, prevLayerDim, LayerDim, MiniBatchSize, lambda, optimizer, comm));
  }else {
    std::cout << "Unknown layer type " << layerName << std::endl;
  }
  return;
}
#endif

void lbann::AutoEncoder::train(CircMat& X, const float LearnRate)
{
    // setup input (last/additional row should always be 1)
    Copy(X, Layers[0]->Acts);

    // forward propagation (mini-batch)
    DataType L2NormSum = 0;
    for (size_t l = 1; l < Layers.size(); l++)
      L2NormSum = Layers[l]->forwardProp(L2NormSum);

    // backward propagation (mini-batch)
    for (size_t l = Layers.size() - 1; l >= 1; l--) {
      Layers[l]->backProp();
    }

    // update weights, biases
    global_update();
    for (size_t l = 1; l < Layers.size(); l++)
        Layers[l]->update(LearnRate);
}

void lbann::AutoEncoder::test(CircMat& X, CircMat& XP)
{
    // setup input (last/additional row should always be 1)
    Copy(X, Layers[0]->Acts);

    // forward propagation (mini-batch)
    DataType L2NormSum = 0;
    for (size_t l = 1; l < Layers.size(); l++)
      L2NormSum = Layers[l]->forwardProp(L2NormSum);

    // get output
    Copy(Layers[Layers.size()-1]->Acts, XP);
}

void lbann::AutoEncoder::test(CircMat& X, CircMat& YP, CircMat& XP)
{
    // setup input (last/additional row should always be 1)
    Copy(X, Layers[0]->Acts);

    // forward propagation (mini-batch)
    DataType L2NormSum = 0;
    for (size_t l = 1; l < Layers.size(); l++)
      L2NormSum = Layers[l]->forwardProp(L2NormSum);

    // get output
    Copy(Layers[Layers.size()/2]->Acts, YP);
    Copy(Layers[Layers.size()-1]->Acts, XP);

}

void lbann::AutoEncoder::global_update() {
  for (size_t l = 1; l < Layers.size(); ++l) {
    Layers[l]->global_update();
  }
}

void lbann::AutoEncoder::train(DataReader* Dataset, int NumEpoch)
{
	if (!Dataset || Dataset->getNumData() == 0)
        return;

	// setup input for forward, backward pass (last/additional row should always be 1)
  //	this->setup(Dataset->getX(), Dataset->getX());

	// do main loop for epoch
    for (int epoch = 0; epoch < NumEpoch; epoch++) {
        // notify epoch begin to event handler
        //...
      if (comm->am_world_master())
            printf("[%d] Epoch begin ..........................\n", epoch);

        // generate a seed number <- need to change later
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

        // restart dataset position
    	Dataset->begin(true, (int)seed);

        // train
        while (Dataset->getPosition() < Dataset->getNumData()) {
            // notify mini-batch begin to event handler
            // ...
          if (comm->am_world_master())
                printf("\t[%d] Mini-batch ...\n", Dataset->getPosition());

            this->trainBatch(Dataset);

            // notify mini-batch end to event handler
            // ...

            if (!Dataset->next())
                break;
        }

        // notify epoch end to event handler
        //...
        if (comm->am_world_master())
            printf("[%d] Epoch end !\n", epoch);
    }
}

void lbann::AutoEncoder::trainBatch(DataReader* Dataset)
{
	if (!Dataset || Dataset->getNumData() == 0)
        return;

	// setup input for forward, backward pass (last/additional row should always be 1)
  //    this->setup(Dataset->getX(), Dataset->getX());

    // forward propagation (mini-batch)
    DataType L2NormSum = 0;
    for (size_t l = 1; l < Layers.size(); l++) {
        L2NormSum = Layers[l]->forwardProp(L2NormSum);
    }

    // backward propagation (mini-batch)
    for (size_t l = Layers.size() - 1; l >= 1; l--) {
        Layers[l]->backProp();
    }

    // Make sure that the output error is periodically reported in backpropagation

    // update weights, biases
    for (size_t l = 1; l < Layers.size(); l++)
        Layers[l]->update(0, 0, 0);
}

DataType lbann::AutoEncoder::evaluate(DataReader* Dataset, int ErrorType)
{
	if (!Dataset || Dataset->getNumData() == 0)
        return -1;

    // setup input for forward, backward pass (last/additional row should always be 1)
  //    this->setup(Dataset->getX(), NULL);

    // restart dataset position
	Dataset->begin(false, 0);

    // test
        if (comm->am_world_master())
        printf("Evaluate .........................\n");

    DataType error = 0;
    while (Dataset->getPosition() < Dataset->getNumData()) {
        // notify mini-batch begin to event handler
        // ...
      if (comm->am_world_master())
            printf("\t[%d] Mini-batch ...\n", Dataset->getPosition());

        error += this->evaluateBatch(Dataset, ErrorType); // need to change!

        // notify mini-batch end to event handler
        // ...

        if (!Dataset->next())
            break;
    }

    return error;
}

DataType lbann::AutoEncoder::evaluateBatch(DataReader* Dataset, int ErrorType)
{
	if (!Dataset || Dataset->getNumData() == 0)
        return -1;

    // setup input for forward, backward pass (last/additional row should always be 1)
  //    this->setup(Dataset->getX(), NULL);

    // forward propagation (mini-batch)
    DataType L2NormSum = 0;
    for (size_t l = 1; l < Layers.size(); l++) {
        L2NormSum = Layers[l]->forwardProp(L2NormSum);
    }

    Copy(*(Layers[Layers.size()-1]->fp_output()), *(Dataset->getYP()));

    // temporary: need to fix later, depending on error type
    DataType error = 0;
    if (comm->am_world_master()) {
        for (int n = 0; n < Dataset->getBatchSize(); n++) {

        }
    }

    return error;
}
