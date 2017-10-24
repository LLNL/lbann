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
// noise.hpp - Noise layer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_NOISE_HPP_INCLUDED
#define LBANN_LAYER_NOISE_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/**
 * Add synthetic (Gaussian) noise to input data (the layer below)
 * @param noise_factor controls the noise level
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class noise_layer : public transform {
 private:

 public:
  /// Constructor
  noise_layer(int index,
            lbann_comm *comm,
            float noise_factor=0.5f,
            cudnn::cudnn_manager *cudnn = NULL)
    : transform(index, comm), m_noise_factor(noise_factor) {

    // Setup the data distribution
    initialize_distributed_matrices();

  #ifdef __LIB_CUDNN
    // Initialize GPU if available
    if(cudnn) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
  #endif // __LIB_CUDNN

  }

  noise_layer(const noise_layer& other) :
    transform(other),
    m_noise_factor(other.m_noise_factor) { }

  noise_layer& operator=(const noise_layer& other) {
    transform::operator=(other);
    m_noise_factor = other.m_noise_factor;
    return *this;
  }

  ~noise_layer() {
  #ifdef __LIB_CUDNN
    // GPU memory for activations is a copy of previous layer's activations
    this->m_error_signal_d.clear();
  #endif // __LIB_CUDNN
  }

  /** Returns description of ctor params */
  std::string get_description() const {
    std::stringstream s;
     s << "noise_layer  noise_factor: " << m_noise_factor
       << " dataLayout: " << this->get_data_layout_string(get_data_layout());
     return s.str();
  }

  noise_layer* copy() const { return new noise_layer(*this); }

  std::string get_type() const { return "noise"; }

  virtual inline void initialize_distributed_matrices() {
    transform::initialize_distributed_matrices<T_layout>();
  }
  virtual data_layout get_data_layout() const { return T_layout; }


  void setup_gpu() {
    transform::setup_gpu();
  #ifndef __LIB_CUDNN
    throw lbann_exception("noise_layer: cuDNN not detected");
  #else
    m_copy_bp_output_from_gpus = true;

  #endif // #ifndef __LIB_CUDNN
  }

  protected:
  /** noise factor */
  float m_noise_factor;

  void fp_compute() {
    if(this->m_using_gpus) {
      fp_compute_cudnn();
    } else {
      fp_compute_cpu();
    }
  }

  void bp_compute() {
    if(this->m_using_gpus) {
      throw lbann_exception("noise_layer: cuDNN not implemented");
    } else {
      El::LockedView(*this->m_error_signal_v, *this->m_prev_error_signal);
    }
  }

  void fp_compute_cudnn() {
    throw lbann_exception("noise_layer: cuDNN not implemented");
  }

  void fp_compute_cpu() {
    El::Copy(*this->m_prev_activations, *this->m_activations_v);
    AbsDistMat* noise_mat = this->m_activations_v->Construct(this->m_activations_v->Grid(),
                                                             this->m_activations_v->Root());
    El::Gaussian(*noise_mat, this->m_activations_v->Height(), 
                 this->m_activations_v->Width(),
                 DataType(0), DataType(1));
    El::Axpy(m_noise_factor,*noise_mat,*this->m_activations_v);
    
    //@todo - clip to min and max of input entry
    auto clip = [&](const DataType& z) { 
         return std::max(DataType(0), std::min(z,DataType(1)));
    };
    EntrywiseMap(*this->m_activations_v, El::MakeFunction(clip));
  }

};

}  // namespace lbann

#endif  // LBANN_LAYER_NOISE_HPP_INCLUDED
