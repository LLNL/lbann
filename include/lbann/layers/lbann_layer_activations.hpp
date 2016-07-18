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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_ACTIVATION_HPP_INCLUDED
#define LBANN_LAYER_ACTIVATION_HPP_INCLUDED

#include "lbann/lbann_base.hpp"
#include <string>



namespace lbann {

/** Represent the type of activation function. */
enum class activation_type {
  SIGMOID = 1,
  TANH,
  RELU,
  ID
};

/** Base activation function class. */
class Activation {
public:
  virtual ~Activation() {}
  virtual void forwardProp(ElMat& m) = 0;
  virtual void backwardProp(ElMat& m) = 0;
};

/** Sigmoid activation function. */
class sigmoid_layer : public Activation {
public:
  void forwardProp(ElMat& m);
  void backwardProp(ElMat& m);
private:
  static DataType sigmoid(DataType z);
  static DataType sigmoidPrime(DataType z);
};

/** Hyperbolic tangent activation function. */
class tanh_layer : public Activation {
public:
  void forwardProp(ElMat& m);
  void backwardProp(ElMat& m);
private:
  static DataType tanh(DataType z);
  static DataType tanhPrime(DataType z);
};

/** Rectified linear unit activation function. */
class reLU_layer : public Activation {
public:
  void forwardProp(ElMat& m);
  void backwardProp(ElMat& m);
private:
  static DataType reLU(DataType z);
  static DataType reLUPrime(DataType z);
};

/** Identity activation function -- does nothing. */
class id_layer : public Activation {
public:
  void forwardProp(ElMat& m) {}
  void backwardProp(ElMat& m) {}
};

/** Return a new Activation class of type act_fn. */
Activation* new_activation(activation_type act_fn);

}  // namespace lbann

#endif  // LBANN_LAYER_ACTIVATION_HPP_INCLUDED
