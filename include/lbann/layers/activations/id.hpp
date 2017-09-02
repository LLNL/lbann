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

#ifndef ID_HPP_INCLUDED
#define ID_HPP_INCLUDED

#include "lbann/layers/activations/activation.hpp"

namespace lbann {

/** Identity activation function -- does nothing. */
template <data_layout T_layout>
class id_layer : public entrywise_activation_layer {
 public:
  // TODO: Optimize this to copy instead of applying elementwise.
  id_layer(int index,
           lbann_comm *comm) :
    entrywise_activation_layer(index, comm) {
     initialize_distributed_matrices(); 
  }
  id_layer* copy() const { return new id_layer(*this); }

  std::string get_name() const { return "id"; }

  virtual DataType activation_function(DataType x)  {
     throw(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " this method should never be called; it's in place only to permit id_layer to have the same inheritance hierarchy as the other activation classes");
    return DataType(0);
  }

  virtual DataType activation_function_gradient(DataType x) {
     throw(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " this method should never be called; it's in place only to permit id_layer to have the same inheritance hierarchy as the other activation classes");
    return DataType(0);
  }

  virtual inline void initialize_distributed_matrices() {
    activation_layer::initialize_distributed_matrices<T_layout>();
  }
  virtual data_layout get_data_layout() const { return T_layout; }

  void fp_compute() {
    El::LockedView(*this->m_activations_v, *this->m_prev_activations);
  }

  void bp_compute() {
    El::LockedView(*this->m_error_signal_v, *this->m_prev_error_signal);
  }

};

}  // namespace lbann

#endif  // ID_HPP_INCLUDED
