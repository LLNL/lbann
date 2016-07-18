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

#include "lbann/models/lbann_model_autoencoder_stacked.hpp"
#include <string>

using namespace std;
using namespace El;


////////////////////////////////////////////////////////////////////////////////
// stacked_autoencoder : main auto encoder class
////////////////////////////////////////////////////////////////////////////////

lbann::StackedAutoencoder::StackedAutoencoder(Optimizer_factory *optimizer_factory, const uint MiniBatchSize, lbann_comm* comm)
	: AutoEncoder(optimizer_factory, MiniBatchSize, comm)
{
#if 0
	// create input layer only (no hidden/out layer)
	Layers.push_back(new FullyConnectedLayer(0, -1, inputNeurons, MiniBatchSize, ActivationType, DropOut, lambda, comm));
#endif
}

lbann::StackedAutoencoder::~StackedAutoencoder()
{
}

#if 0
void lbann::beginStack(int LayerNeurons, lbann_comm* comm)
{
	// get prev neurons
	int mid = (int)Layers.size() / 2;
	FullyConnectedLayer* player = (FullyConnectedLayer*)Layers[mid];
	int pneurons = player->NumNeurons;

	if (Layers.size() == 1) {
		// create first hidden layer
		Layers.push_back(new FullyConnectedLayer(1, pneurons, LayerNeurons, MiniBatchSize, ActivationType, DropOut, Lambda, comm));
		// create output layer
		Layers.push_back(new FullyConnectedLayer(2, LayerNeurons, pneurons, MiniBatchSize,
												 ActivationType, DropOut, Lambda, comm));
	}
	else {
		// create hiden layer
		Layers.insert(Layers.begin() + mid + 1,
					  new FullyConnectedLayer(-1, pneurons, LayerNeurons, MiniBatchSize,
											  ActivationType, DropOut, Lambda, comm));

		// create mirror layer
		Layers.insert(Layers.begin() + mid + 2,
					  new FullyConnectedLayer(-1, LayerNeurons, pneurons, MiniBatchSize,
											  ActivationType, DropOut, Lambda, comm));

		// re-number layer index
		for (int n = 0; n < (int)Layers.size(); n++) {
			Layer* layer = Layers[n];
			layer->Index = n;
		}
	}

	/// temp
	for (int n = 0; n < (int)Layers.size(); n++) {
		FullyConnectedLayer* layer = (FullyConnectedLayer*)Layers[n];
		printf("[%d] layer dim: %d\n", n, layer->NumNeurons);

	}
}

void lbann::trainStack(CircMat& X, const float LearnRate, const int LearnRateMethod, const DataType DecayRate)
{
	// setup input (last/additional row should always be 1)
	Copy(X, Layers[0]->Acts);

	int l1 = (int)Layers.size() / 2;
	int l0 = l1 - 1;
	int l2 = l1 + 1;

	// forward propagation (until current hidden layer + 1)
  DataType L2NormSum = 0;
	for (size_t l = 1; l <= l2; l++) {
		L2NormSum = Layers[l]->forwardProp(L2NormSum);
	}

	// backward propagation (from current hidden layer + 1)
	Layers[l2]->backProp(*Layers[l1], Layers[l0]->Acts);
	Layers[l1]->backProp(*Layers[l0], *Layers[l2]);

	// update weights, biases
	Layers[l1]->updateMB(LearnRate, LearnRateMethod, DecayRate);
	Layers[l2]->updateMB(LearnRate, LearnRateMethod, DecayRate);
}

void lbann::endStack()
{

}

#endif
