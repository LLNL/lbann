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

#ifndef LBANN_WEIGHTS_HPP
#define LBANN_WEIGHTS_HPP

#include <string>

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/weights/initializer.hpp"
#include "lbann/io/persist.hpp"
#include <lbann.pb.h>
namespace lbann {

// Forward declaration
class optimizer;

/** Neural network weights.
 *  Weights are tensors that act as trainable parameters for a neural
 *  network. The values can be initialized with a weights initializer
 *  and are optimized with first-order methods (e.g. stochastic
 *  gradient descent).
 *
 *  Internally, the weight values are stored in a 2D distributed
 *  matrix. The "matrix height dimensions" are tensor dimensions that
 *  correspond to the matrix height. The remaining dimensions, the
 *  "matrix width dimensions," correspond to the matrix width.
 *
 *  Note that LBANN weights are similar to Tensorflow variables and
 *  Caffe parameters.
 */
class weights {
  friend class optimizer;

 public:
  weights(lbann_comm* comm,
          cudnn::cudnn_manager* cudnn = nullptr);
  weights(const weights& other);
  weights& operator=(const weights& other);
  virtual ~weights();

  /** Set weights name.
   *  Each set of weights in a model should have a unique,
   *  human-readable name.
   */
  inline void set_name(const std::string name) { m_name = name; }
  /** Get weights name. */
  inline std::string get_name() const { return m_name; }

  /** Create a copy of the weights.
   *  This function dynamically allocates memory for a weights
   *  instance and instantiates a copy. The caller is responsible for
   *  deallocating the instance.
   */
  virtual weights* copy() const { return new weights(*this); }

  /** Setup weights as a vector.
   *  The weight matrix is setup as a (size x 1) matrix in STAR,STAR
   *  format.
   */
  virtual void setup(int size, El::Device dev);
  /** Setup weights as a tensor.
   *  The weight matrix is setup as a (prod(dims) x 1) matrix in
   *  STAR,STAR format.
   */
  virtual void setup(std::vector<int> dims, El::Device dev);
  /** Setup weights as a matrix.
   *  The weight matrix is setup as a (matrix_height x matrix_width)
   *  matrix in col_dist,row_dist format.
   */
  virtual void setup(int matrix_height,
                     int matrix_width,
                     El::Distribution col_dist,
                     El::Distribution row_dist,
                     El::Device dev);
  /** Setup weights as a matrix with tensor dimensions.
   *  The weight matrix is setup as a (prod(matrix_height_dims) x
   *  prod(matrix_width_dims)) matrix in col_dist,row_dist format.
   */
  virtual void setup(std::vector<int> matrix_height_dims,
                     std::vector<int> matrix_width_dims,
                     El::Distribution col_dist,
                     El::Distribution row_dist,
                     El::Device dev);

  /** Get weight tensor dimensions.
   *  The dimensions are sorted in decreasing order of the data
   *  strides. This is a generalization of the "NCHW/NHWC" notation
   *  commonly used to describe image data.
   *
   *  These dimensions are obtained by concatenating the matrix width
   *  dimensions with the matrix height dimensions (in that order). If
   *  the weight matrix is duplicated on all processes (i.e. in
   *  STAR,STAR layout) and the local matrices are fully-packed, the
   *  tensor data is fully-packed. If the matrix is STAR,STAR and the
   *  local matrices are not fully-packed, the tensor data is
   *  fully-packed w.r.t. the matrix height dimensions.
   */
  std::vector<int> get_dims() const;
  /** Get number of weights. */
  inline int get_size() const { return get_matrix_height() * get_matrix_width(); }

  /** Get tensor dimensions corresponding to weight matrix height.
   *  The dimensions are sorted in decreasing order of strides. Matrix
   *  rows are fully-packed w.r.t. the matrix height dimensions.
   */
  inline const std::vector<int>& get_matrix_height_dims() const { return m_matrix_height_dims; }
  /** Get tensor dimensions corresponding to weight matrix width.
   *  The dimensions are sorted in decreasing order of strides. Matrix
   *  columns are fully-packed w.r.t. the matrix width dimensions.
   */
  inline const std::vector<int>& get_matrix_width_dims() const { return m_matrix_width_dims; }
  /** Get weight matrix height.
   *  If there are no matrix height dimensions, the height is one.
   */
  int get_matrix_height() const;
  /** Get weight matrix width.
   *  If there are no matrix width dimensions, the width is one.
   */
  int get_matrix_width() const;

  /** Get reference to cuDNN manager. */
  inline cudnn::cudnn_manager* get_cudnn_manager() { return m_cudnn; }

  /** Get weights initializer. */
  inline weights_initializer& get_initializer() { return *m_initializer; }
  /** Get weights initializer (const). */
  inline const weights_initializer& get_initializer() const { return *m_initializer; }
  /** Set weights initializer.
   *  This takes ownership of the initializer and deallocates it
   *  during destruction.
   */
  void set_initializer(weights_initializer* initializer);

  /** Get weights optimizer. */
  optimizer* get_optimizer() { return m_optimizer; }
  /** Get weights optimizer (const). */
  const optimizer* get_optimizer() const { return m_optimizer; }
  /** Set weights optimizer.
   *  This takes ownership of the optimizer and deallocates it during
   *  destruction.
   */
  void set_optimizer(optimizer* opt);

  /** Get the weight matrix. */
  const AbsDistMat& get_values();
  /** Set the weight matrix. */
  void set_values(const AbsDistMat& values);

  /** Set a weight value. */
  void set_value(DataType value, int index);
  /** Set an entry in the weight tensor. */
  void set_value(DataType value, std::vector<int> pos);
  /** Set an entry in the weight matrix. */
  void set_value(DataType value, int row, int col);

  /** Get a view into the weight matrix.
   *  If values_v has a different matrix distribution than the weight
   *  matrix, the matrix values are copied into values_v.
   */
  void get_values_view(AbsDistMat& values_v);

#ifdef LBANN_HAS_CUDNN
  /** Get the weight matrix on GPU. */
  std::vector<DataType*> get_values_gpu();
#endif // LBANN_HAS_CUDNN

  bool save_to_checkpoint_shared(persist& p);
  bool load_from_checkpoint_shared(persist& p);

  /** Write weights to proto file */
  virtual void write_proto(lbann_data::Weights* proto) const;
 private:

  /** Weights name.
   *  See get_name function.
   */
  std::string m_name;

  /** Reference to LBANN communicator. */
  lbann_comm* m_comm;
  /** Reference to cuDNN manager. */
  cudnn::cudnn_manager* m_cudnn;

  /** Tensor dimensions corresponding to matrix height.
   *  See get_matrix_height_dims function.
   */
  std::vector<int> m_matrix_height_dims;
  /** Tensor dimensions corresponding to matrix width.
   *  See get_matrix_width_dims function.
   */
  std::vector<int> m_matrix_width_dims;

  /** Weights matrix. */
  AbsDistMat* m_values = nullptr;

  /** Weights initializer.
   *  Default is zero initialization.
   */
  weights_initializer* m_initializer = nullptr;
  /** Weights optimizer.
   *  Default is nullptr, which corresponds to no optimizer.
   */
  optimizer* m_optimizer = nullptr;

#ifdef LBANN_HAS_CUDNN
  /** GPU memory for weights matrix. */
  std::vector<DataType*> m_values_d;
#endif // LBANN_HAS_CUDNN

  /** Setup GPU objects for weights. */
  virtual void setup_gpu();

  /** Get string describing weight tensor dimensions.
   *  height_dims and width_dims are the dimensions of the weight
   *  matrix.
   */
  static std::string get_dims_string(const std::vector<int>& height_dims,
                                     const std::vector<int>& width_dims);

};

} // namespace lbann

#endif // LBANN_WEIGHTS_HPP
