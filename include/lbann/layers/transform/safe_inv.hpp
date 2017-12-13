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
// safe_inv.hpp - Safe inversion layer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_SAFE_INV_HPP_INCLUDED
#define LBANN_LAYER_SAFE_INV_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/**
 * Safe inversion (reciprocal) of unpooling layer
 * y = (x == DataType(0)) ? (Datatype(0)) : (DataType(1)/x);
 * Reference: https://arxiv.org.abs/1606.06582
 *   
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class safe_inv_layer : public transform_layer {
 private:

 public:
  /// Constructor
  safe_inv_layer(lbann_comm *comm,
              cudnn::cudnn_manager *cudnn = nullptr)
    : transform_layer(comm){

    // Setup the data distribution
    initialize_distributed_matrices();

  #ifdef LBANN_HAS_CUDNN
    // Initialize GPU if available
    if(cudnn) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
  #endif // LBANN_HAS_CUDNN

  }

  safe_inv_layer(const safe_inv_layer& other) :
    transform_layer(other) { }

  safe_inv_layer& operator=(const safe_inv_layer& other) {
    transform_layer::operator=(other);
    return *this;
  }

  ~safe_inv_layer() override {
  #ifdef LBANN_HAS_CUDNN
    // GPU memory for activations is a copy of previous layer's activations
    this->m_error_signal_d.clear();
  #endif // LBANN_HAS_CUDNN
  }

  /** Returns description of ctor params */
  std::string get_description() const override {
    std::stringstream s;
     s << " dataLayout: " << this->get_data_layout_string(get_data_layout());
     return s.str();
  }

  safe_inv_layer* copy() const override { return new safe_inv_layer(*this); }

  std::string get_type() const override { return "safe_inv"; }

  virtual inline void initialize_distributed_matrices() {
    transform_layer::initialize_distributed_matrices<T_layout>();
  }
  data_layout get_data_layout() const override { return T_layout; }


  void setup_gpu() override {
    transform_layer::setup_gpu();
  #ifndef LBANN_HAS_CUDNN
    throw lbann_exception("safe_inv_layer: cuDNN not detected");
  #else
    m_copy_bp_output_from_gpus = true;

  #endif // #ifndef LBANN_HAS_CUDNN
  }

  protected:

  void fp_compute() override {
    if(this->m_using_gpus) {
      fp_compute_cudnn();
    } else {
      fp_compute_cpu();
    }
  }

  void bp_compute() override {
    if(this->m_using_gpus) {
      throw lbann_exception("safe_inv_layer: cuDNN not implemented");
    } else {
      El::LockedView(*this->m_error_signal_v, *this->m_prev_error_signal_v);
    }
  }

  void fp_compute_cudnn() {
    throw lbann_exception("safe_inv_layer: cuDNN not implemented");
  }

  void fp_compute_cpu() {
    El::Copy(*this->m_prev_activations_v, *this->m_activations_v);
    
    auto inv = [&](const DataType& z) { 
         return (z == DataType(0)) ? (DataType(0)) : (DataType(1)/z);
    };
    EntrywiseMap(*this->m_activations_v, El::MakeFunction(inv));
  }

};

}  // namespace lbann

#endif  // LBANN_LAYER_SAFE_INV_HPP_INCLUDED
