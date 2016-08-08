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
// lbann_model_sequential .hpp .cpp - Sequential neural network models
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_MODEL_SEQUENTIAL_HPP
#define LBANN_MODEL_SEQUENTIAL_HPP

#include "lbann/models/lbann_model.hpp"
#include "lbann/layers/lbann_layer.hpp"
#include "lbann/layers/lbann_layer_activations.hpp"
#include "lbann/data_readers/lbann_data_reader.hpp"
#include "lbann/layers/lbann_layer_factory.hpp"
#include <vector>
#include <string>

namespace lbann
{
class Sequential : public Model
  {
  public:
    Sequential(Optimizer_factory *optimizer_factory, const uint MiniBatchSize, lbann_comm* comm, layer_factory* layer_fac);
    ~Sequential();

    bool saveToFile(std::string FileDir);
    bool loadFromFile(std::string FileDir);

    bool saveToCheckpoint(int fd, const char* filename, uint64_t* bytes);
    bool loadFromCheckpoint(int fd, const char* filename, uint64_t* bytes);

    bool saveToCheckpointShared(const char* dir, uint64_t* bytes);
    bool loadFromCheckpointShared(const char* dir, uint64_t* bytes);

    // add layer, setup layers for forward/backward pass
    std::vector<Layer*>& get_layers() { return Layers; }
    virtual uint add(std::string layerName, int LayerDim, activation_type ActivationType=activation_type::RELU, std::vector<regularizer*> regs={});
    virtual uint add(Layer *new_layer);
    virtual Layer* remove(int index);
    virtual void insert(int index, Layer *new_layer);
    virtual Layer* swap(int index, Layer *new_layer);

    virtual void setup();

    virtual void train(int NumEpoch, bool EvaluateEveryEpoch=false) = 0;
    virtual bool trainBatch(long *num_samples, long *num_errors) = 0;

    virtual DataType evaluate(execution_mode mode) = 0;
    virtual bool evaluateBatch(long *num_samples, long *num_errors) = 0;
#if 0
    virtual DistMat* predictBatch(DistMat* X);
#endif

  public:
    std::vector<Layer*> Layers; //@todo replace with layer factory
    std::vector<Layer*>::iterator it;
    Optimizer_factory*  optimizer_factory;
    int                 MiniBatchSize;
    layer_factory* lfac;

  };
}
#endif  //  LBANN_MODEL_SEQUENTIAL_HPP
