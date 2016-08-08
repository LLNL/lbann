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
// lbann_model_dnn .hpp .cpp - Deep Neural Networks models
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_MODEL_DNN_HPP
#define LBANN_MODEL_DNN_HPP

#include "lbann/models/lbann_model_sequential.hpp"
#include "lbann/layers/lbann_layer.hpp"
#include <vector>
#include <string>

namespace lbann
{
  class Dnn : public Sequential
  {
  public:
    Dnn(Optimizer_factory *optimizer_factory, const uint MiniBatchSize, lbann_comm* comm, layer_factory* layer_fac);
    Dnn(const uint MiniBatchSize,
        const double Lambda, Optimizer_factory *optimizer_factory,
        lbann_comm* comm,layer_factory* layer_fac);
    ~Dnn();

    void checkGradient(CircMat& X, CircMat& Y, double* GradientErrors);

    void summarize(lbann_summary& summarizer);

    void train(int NumEpoch, bool EvaluateEveryEpoch=false);
    bool trainBatch(long *num_samples, long *num_errors);

    DataType evaluate(execution_mode mode=execution_mode::testing);
    bool evaluateBatch(long *num_samples, long *num_errors);

    DataType get_train_accuracy() const { return training_accuracy; }
    DataType get_validate_accuracy() const { return validation_accuracy; }
    DataType get_test_accuracy() const { return test_accuracy; }

  public:
    //int MiniBatchSize;
    //Optimizer_factory *optimizer_factory;

  protected:
    DataType training_accuracy;
    DataType validation_accuracy;
    DataType test_accuracy;
  };
}


#endif // LBANN_MODEL_DNN_HPP
