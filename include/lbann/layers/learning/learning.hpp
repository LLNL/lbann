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
// lbann_layer .h .cpp - Parent class for all layer types
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_LEARNING_HPP_INCLUDED
#define LBANN_LAYER_LEARNING_HPP_INCLUDED

#include "lbann/layers/layer.hpp"
#include "lbann/layers/optimizable_layer.hpp"
#include <string>
#include <vector>

namespace lbann {

class learning : public Layer, public optimizable_layer {
 protected:
  optimizer  *m_optimizer;

  AbsDistMat *m_weights;             ///< Weight matrix (computes weight sum of inputs ((# neurons) x (# previous layer's neurons))
  AbsDistMat *m_weights_gradient;    ///< Gradient w.r.t. weight matrix ((# neurons) x (# previous layer's neurons))


  /** Factor for L2 regularization; 0 to disable. */
  DataType m_l2_regularization_factor = DataType(0);

  /** Add L2 regularization term to objective function. */
  virtual void l2_regularize_objective_function() {
    if (m_l2_regularization_factor != DataType(0)) {

      // Get local weight data
      const DataType *weights_buffer = m_weights->LockedBuffer();
      const int weights_ldim = m_weights->LDim();
      const int local_height = m_weights->LocalHeight();
      const int local_width = m_weights->LocalWidth();

      // Compute sum of squares
      DataType sum = 0;
      const int block_size = std::max((int) (64 / sizeof(DataType)), 1);
      #pragma omp parallel for reduction(+:sum) collapse(2)
      for(int col = 0; col < local_width; ++col) {
        for(int block_start = 0; block_start < local_height; block_start += block_size) {
          double block_sum = 0;
          const int block_end = std::min(block_start + block_size, local_height);
          for(int row = block_start; row < block_end; ++row) {
              const DataType x = weights_buffer[row + col * weights_ldim];
              block_sum += x * x;
          }
          sum += block_sum;
        }
      }
      sum = El::mpi::AllReduce(sum, m_weights->DistComm());
      
      // Add regularization term to objective function
      const DataType regularization_term = m_l2_regularization_factor * sum / 2;
      this->m_neural_network_model->m_obj_fn->add_to_value(regularization_term);

    }
  }

  /** Add L2 regularization term to gradient. */
  virtual void l2_regularize_gradient() {
    if (m_l2_regularization_factor != DataType(0)) {
      El::Axpy(m_l2_regularization_factor, *m_weights, *m_weights_gradient);
    }
  }

  /** Factor for group lasso regularization */ 
  DataType m_group_lasso_regularization_factor = DataType(0);

  /** Add group lasso regularization term to objective function. */
  virtual void group_lasso_regularize_objective_function() {
    if (m_group_lasso_regularization_factor != DataType(0)) {

      // Get local weight data
      const DataType *weights_buffer = m_weights->LockedBuffer();
      const int weights_ldim = m_weights->LDim();
      const int width = m_weights->Width();
      const int local_height = m_weights->LocalHeight();
      const int local_width = m_weights->LocalWidth();

      // Compute sum of squares with Kahan summation
      Mat colSumSqrs = Mat(1, width);
      El::Zero(colSumSqrs);
      for (int col = 0; col < local_width; ++col) {
        DataType sum = 0;
        DataType correction = 0;
        for (int row = 0; row < local_height; ++row) {
          const DataType x = weights_buffer[row + col * weights_ldim];
          const DataType term = x * x + correction;
          const double next_sum = sum + term;
          correction = term - (next_sum - sum);
          sum = next_sum;
        }
        colSumSqrs(0, m_weights->GlobalCol(col)) = sum;
      }
      El::AllReduce(colSumSqrs, m_weights->DistComm(), El::mpi::SUM);
    
      // Add regularization term to objective function
      DataType sumColL2Norms = 0;
      for (int i = width - 1; i >= 0; --i) sumColL2Norms += std::sqrt(colSumSqrs(0, i)); 
      const DataType regularization_term = m_group_lasso_regularization_factor * sumColL2Norms;
      this->m_neural_network_model->m_obj_fn->add_to_value(regularization_term);
    }
  }

  /** Add group lasso regularization term to gradient. */
  virtual void group_lasso_regularize_gradient() {
    if (m_group_lasso_regularization_factor != DataType(0)) {
      //Group lasso gradient will be m_weights where each column c of m_weights is normalized by $||c||_{2}$.
      //So first we compute $L_2$ norm of each column of m_weights as in group_lasso_regularize_objective_function. 
      const DataType *weights_buffer = m_weights->LockedBuffer();
      const int weights_ldim = m_weights->LDim();
      const int width = m_weights->Width();
      const int local_height = m_weights->LocalHeight();
      const int local_width = m_weights->LocalWidth();

      // Compute sum of squares with Kahan summation
      Mat colSumSqrs = Mat(1, width);
      for (int col = 0; col < local_width; ++col) {
        DataType sum = 0;
        DataType correction = 0;
        for (int row = 0; row < local_height; ++row) {
          const DataType x = weights_buffer[row + col * weights_ldim];
          const DataType term = x * x + correction;
          const double next_sum = sum + term;
          correction = term - (next_sum - sum);
          sum = next_sum;
        }
        colSumSqrs(0, m_weights->GlobalCol(col)) = sum;
      }
      El::AllReduce(colSumSqrs, m_weights->DistComm(), El::mpi::SUM);
      El::EntrywiseMap(colSumSqrs, std::function<DataType(const DataType&)>(
									   [](const DataType& x) {
									     return std::sqrt(x);
									   }));
   
      //update m_weights_graident using colSumSqrs.
      Mat& m_weights_gradient_local = m_weights_gradient->Matrix();
      Mat& m_weights_local = m_weights->Matrix();
      AbsDistMat& boo = *m_weights_gradient;

      El::IndexDependentMap(m_weights_gradient_local, 
			    (std::function<DataType(El::Int,El::Int,const DataType&)>)
			    ([&colSumSqrs, &m_weights_local, &boo](El::Int r, El::Int c, const DataType& x)
			     -> DataType {
			      const DataType colL2Norm = colSumSqrs(0, boo.GlobalCol(c));
                              if (colL2Norm != DataType(0)) {
                                return x + m_weights_local(r, c)/colL2Norm;
			      } else {
                                return x;			
			      };  
			    }));
    }
  }

 public:
  learning(int index, 
           lbann_comm *comm,
           optimizer *opt)
    : Layer(index, comm), m_optimizer(opt) {}

  learning(const learning& other) :
    Layer(other),
    m_l2_regularization_factor(other.m_l2_regularization_factor) {
    m_weights = other.m_weights->Copy();
    m_weights_gradient = other.m_weights_gradient->Copy();
    m_optimizer = other.m_optimizer->copy();
  }

  learning& operator=(const learning& other) {
    Layer::operator=(other);
    m_l2_regularization_factor = other.m_l2_regularization_factor;
    if (m_weights) {
      delete m_weights;
      delete m_weights_gradient;
    }
    m_weights = other.m_weights->Copy();
    m_weights_gradient = other.m_weights_gradient->Copy();
    if (m_optimizer) {
      delete m_optimizer;
    }
    if (other.m_optimizer) {
      m_optimizer = other.m_optimizer->copy();
    }
    return *this;
  }

  virtual ~learning() {
    delete m_optimizer;
    delete m_weights;
    delete m_weights_gradient;
  }

  /** Return the weights associated with this layer. */
  virtual AbsDistMat& get_weights() const { return *m_weights; }
  /** Return the gradients associated with this layer. */
  virtual AbsDistMat& get_weights_gradient() const { return *m_weights_gradient; }

  /// Following function tells this layer has weights
  bool is_learning_layer() { return true; }

  template <data_layout T_layout>
  inline void initialize_distributed_matrices();

  /// @todo BVE should the learning layer be able to initialize the
  /// matrix, or is that purely a function of the children classes
  //enum class weight_initialization {zero, uniform, normal, glorot_normal, glorot_uniform, he_normal, he_uniform};
  static std::string weight_initialization_name(weight_initialization id) {
    switch(id) {
    case weight_initialization::zero :
      return "zero";
      break;
    case weight_initialization::uniform :
      return "uniform";
      break;
    case weight_initialization::normal :
      return "normal";
      break;
    case weight_initialization::glorot_normal :
      return "glorot_normal";
      break;
    case weight_initialization::glorot_uniform :
      return "glorot_uniform";
      break;
    case weight_initialization::he_normal :
      return "he_normal";
      break;
    case weight_initialization::he_uniform :
      return "he_uniform";
      break;
    default:
      throw lbann_exception(
        std::string(__FILE__) + " " + std::to_string(__LINE__) + " :: "
        "unknown weight_initialization: " + std::to_string((int) id));
    }
  }

  virtual void summarize_matrices(lbann_summary& summarizer, int step) {
    Layer::summarize_matrices(summarizer, step);
    std::string prefix = "layer" + std::to_string(static_cast<long long>(m_index)) + "/weights/";
    const AbsDistMat& wb = get_weights();
    summarizer.reduce_mean(prefix + "mean", wb, step);
    summarizer.reduce_min(prefix + "min", wb, step);
    summarizer.reduce_max(prefix + "max", wb, step);
    summarizer.reduce_stdev(prefix + "stdev", wb, step);
    prefix = "layer" + std::to_string(static_cast<long long>(m_index)) + "/weights_gradient/";
    const AbsDistMat& wb_d = get_weights_gradient();
    summarizer.reduce_mean(prefix + "mean", wb_d, step);
    summarizer.reduce_min(prefix + "min", wb_d, step);
    summarizer.reduce_max(prefix + "max", wb_d, step);
    summarizer.reduce_stdev(prefix + "stdev", wb_d, step);
  }

  /** Validate that the setup is reasonable. */
  virtual void check_setup() {
    Layer::check_setup();
    // If these two are sendable, the other matrices should be fine.
    if (!lbann::lbann_comm::is_sendable(*m_weights)) {
      throw lbann::lbann_exception("Weights too large to send");
    }
    if (!lbann::lbann_comm::is_sendable(*m_activations)) {
      throw lbann::lbann_exception("Activations too large to send");
    }
  }

  /** Return the layer's optimizer. */
  optimizer *get_optimizer() const override {
    return m_optimizer;
  }

  /** Set the layer's L2 regularization factor (0 to disable). */
  void set_l2_regularization_factor(DataType f) {
    m_l2_regularization_factor = f;
  }

  bool saveToFile(int fd, const char *dirname) {
    Layer::loadFromFile(fd, dirname);
    char filepath[512];
    sprintf(filepath, "%s/weights_L%d_%03lldx%03lld", dirname, m_index, m_weights->Height()-1, m_weights->Width()-1);
    
    uint64_t bytes;
    return lbann::write_distmat(-1, filepath, (DistMat *)m_weights, &bytes);
  }
  
  bool loadFromFile(int fd, const char *dirname) {
    Layer::loadFromFile(fd, dirname);
    char filepath[512];
    sprintf(filepath, "%s/weights_L%d_%03lldx%03lld.bin", dirname, m_index, m_weights->Height()-1, m_weights->Width()-1);
    
    uint64_t bytes;
    return lbann::read_distmat(-1, filepath, (DistMat *)m_weights, &bytes);
  }

  virtual bool saveToCheckpointShared(lbann::persist& p) {
    Layer::saveToCheckpointShared(p);
    // define name to store our parameters
    char name[512];
    sprintf(name, "weights_L%d_%lldx%lld", m_index, m_weights->Height(), m_weights->Width());

    // write out our weights to the model file
    p.write_distmat(persist_type::model, name, (DistMat *)m_weights);

    // if saving training state, also write out state of optimizer
    // m_optimizer->saveToCheckpointShared(p, m_index);

    return true;
  }

  virtual bool loadFromCheckpointShared(lbann::persist& p) {
    Layer::loadFromCheckpointShared(p);
    // define name to store our parameters
    char name[512];
    sprintf(name, "weights_L%d_%lldx%lld.bin", m_index, m_weights->Height(), m_weights->Width());

    // read our weights from model file
    p.read_distmat(persist_type::model, name, (DistMat *)m_weights);

    // if loading training state, read in state of optimizer
    // m_optimizer->loadFromCheckpointShared(p, m_index);

    return true;
  }

};

/// Matrices should be in MC,MR distributions
template<> inline void learning::initialize_distributed_matrices<data_layout::MODEL_PARALLEL>() {
  Layer::initialize_distributed_matrices<data_layout::MODEL_PARALLEL>();
  m_weights          = new DistMat(m_comm->get_model_grid());
  m_weights_gradient = new DistMat(m_comm->get_model_grid());
}

/// Weight matrices should be in Star,Star and data matrices Star,VC distributions
template<> inline void learning::initialize_distributed_matrices<data_layout::DATA_PARALLEL>() {
  Layer::initialize_distributed_matrices<data_layout::DATA_PARALLEL>();
  m_weights          = new StarMat(m_comm->get_model_grid());
  m_weights_gradient = new StarMat(m_comm->get_model_grid());
}

}  // namespace lbann

#endif  // LBANN_LAYER_LEARNING_HPP_INCLUDED
