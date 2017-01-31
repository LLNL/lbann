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

#ifndef LBANN_LAYER_ACTIVATIONS_HPP_INCLUDED
#define LBANN_LAYER_ACTIVATIONS_HPP_INCLUDED

#include "lbann/lbann_base.hpp"

namespace lbann {

/** Represent the type of activation function. */
enum class activation_type {
  //if you add or change the following enums, please also edit the 
  //activation_name() method in the Activation class
  SIGMOID = 1,
  TANH,
  RELU,
  ID,
  LEAKY_RELU,
#if 0
  SOFTPLUS,
#else
  SMOOTH_RELU,
#endif
  ELU
};

/** Base activation function class. */
class Activation {
public:
  virtual ~Activation() {}
  virtual void forwardProp(ElMat& m) = 0;
  virtual void backwardProp(ElMat& m) = 0;
  static const std::string activation_name(activation_type id);
};

/** Sigmoid activation function. */
class sigmoid_layer : public Activation {
public:
  void forwardProp(ElMat& m);
  void backwardProp(ElMat& m);
private:
  static DataType sigmoid(const DataType& z);
  static DataType sigmoidPrime(const DataType& z);
};

/** Hyperbolic tangent activation function. */
class tanh_layer : public Activation {
public:
  void forwardProp(ElMat& m);
  void backwardProp(ElMat& m);
private:
  static DataType tanh(const DataType& z);
  static DataType tanhPrime(const DataType& z);
};

/** Rectified linear unit activation function. */
class reLU_layer : public Activation {
public:
  void forwardProp(ElMat& m);
  void backwardProp(ElMat& m);
private:
  static DataType reLU(const DataType& z);
  static DataType reLUPrime(const DataType& z);
};

/** Identity activation function -- does nothing. */
class id_layer : public Activation {
public:
  void forwardProp(ElMat& m) {}
  void backwardProp(ElMat& m) {}
};

/**
 * Leaky rectified linear unit activation function.
 * This is a ReLU variant that avoids the dying ReLU problem where a ReLU neuron
 * can stop updating. See:
 * Maas, Andrew L., Awni Y. Hannun, and Andrew Y. Ng. "Rectifier nonlinearities
 * improve neural network acoustic models." Proc. ICML. Vol. 30. No. 1. 2013.
 */
class leaky_reLU_layer : public Activation {
public:
  /** Leak is the amount of signal to permit for negative values. */
  leaky_reLU_layer(DataType leak = 0.01f);
  void forwardProp(ElMat& m);
  void backwardProp(ElMat& m);
private:
  static DataType leaky_reLU(const DataType& z, DataType k);
  static DataType leaky_reLUPrime(const DataType& z, DataType k);
  DataType leak;
};

#if 0
/** softplus (Smooth Rectified linear unit) activation function. ln(1+e^x) */
class softplus_layer : public Activation {
public:
  void forwardProp(ElMat& m);
  void backwardProp(ElMat& m);
private:
  static DataType softplus(const DataType& z);
  static DataType softplusPrime(const DataType& z);
};
#else
/** Smooth Rectified linear unit activation function. x*e^x */
class smooth_reLU_layer : public Activation {
public:
  void forwardProp(ElMat& m);
  void backwardProp(ElMat& m);
private:
  static DataType smooth_reLU(const DataType& z);
  static DataType smooth_reLUPrime(const DataType& z);
};
#endif

/**
 * Exponential linear unit.
 * Tries to speed up learning by pushing the mean of activations more towards
 * zero by allowing negative values. Helps avoid the need for batch
 * normalization.
 * See:
 * Djork-Arne Clevert, Thomas Unterthiner, and Sepp Hochreiter
 * "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)"
 * ICLR 2016.
 */
class ELU_layer : public Activation {
public:
  /**
   * alpha controls the value to which the ELU saturates for negative inputs.
   * alpha must be >= 0.
   * If alpha = 0, this turns into a ReLU.
   * Paper uses alpha = 1.0 as a good starting point.
   */
  ELU_layer(DataType alpha = 1.0f);
  void forwardProp(ElMat& m);
  void backwardProp(ElMat& m);
private:
  static DataType elu(const DataType& z, DataType alpha);
  static DataType eluPrime(const DataType& z, DataType alpha);
  DataType alpha;
};

/** Return a new Activation class of type act_fn. */
Activation* new_activation(activation_type act_fn,
                           DataType param = 0.0f);

}  // namespace lbann

#endif  // LBANN_LAYER_ACTIVATIONS_HPP_INCLUDED
