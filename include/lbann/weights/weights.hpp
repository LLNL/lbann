////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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
#include <vector>
#include <memory>

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/weights/initializer.hpp"
#include "lbann/optimizers/optimizer.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/utils/description.hpp"
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
  weights(lbann_comm* comm);
  weights(const weights& other);
  weights& operator=(const weights& other);

  /** Set weights name.
   *  Each set of weights in a model should have a unique,
   *  human-readable name.
   */
  void set_name(std::string name) { m_name = name; }
  /** Get weights name. */
  std::string get_name() const { return m_name; }

  /** Create a copy of the weights.
   *  This function dynamically allocates memory for a weights
   *  instance and instantiates a copy. The caller is responsible for
   *  deallocating the instance.
   */
  weights* copy() const { return new weights(*this); }

  /** Human-readable description. */
  description get_description() const;

  // -----------------------------------------------
  // Dimension accessors
  // -----------------------------------------------
  /** Get weight tensor dimensions.
   *  The dimensions are sorted in decreasing order of the data
   *  strides. This is a generalization of the "NCHW/NHWC" notation
   *  commonly used to describe image data.
   *
   *  These dimensions are obtained by concatenating the matrix width
   *  dimensions with the matrix height dimensions (in that order).
   *  If the weight matrix is duplicated on all processes (i.e. in
   *  STAR,STAR layout), the tensor data is packed w.r.t. the matrix
   *  height dimensions. If the local matrices are also fully packed,
   *  the tensor data is fully packed.
   */
  std::vector<int> get_dims() const;
  /** Get number of entries in weight tensor. */
  int get_size() const;
  /** Get tensor dimensions corresponding to weight matrix height.
   *  The dimensions are sorted in decreasing order of strides. Matrix
   *  rows are fully-packed w.r.t. the matrix height dimensions.
   */
  std::vector<int> get_matrix_height_dims() const;
  /** Get tensor dimensions corresponding to weight matrix width.
   *  The dimensions are sorted in decreasing order of strides. Matrix
   *  columns are fully-packed w.r.t. the matrix width dimensions.
   */
  std::vector<int> get_matrix_width_dims() const;
  /** Get weight matrix height.
   *  If there are no matrix height dimensions, the height is one.
   */
  int get_matrix_height() const;
  /** Get weight matrix width.
   *  If there are no matrix width dimensions, the width is one.
   */
  int get_matrix_width() const;
  /** Set weight tensor dimensions.
   *  See the 'get_dims' function for an explanation of the notation.
   */
  void set_dims(std::vector<int> matrix_height_dims,
                std::vector<int> matrix_width_dims = std::vector<int>());
  /** Set weight tensor dimensions as a 1D tensor. */
  void set_dims(int size) { set_dims({size}, {}); }

  // -----------------------------------------------
  // Initializer accessors
  // -----------------------------------------------
  /** Get weights initializer. */
  weights_initializer* get_initializer();
  /** Get weights initializer (const). */
  const weights_initializer* get_initializer() const;
  /** Set weights initializer.
   *  The contents of 'init' are moved to a class member.
   */
  void set_initializer(std::unique_ptr<weights_initializer>& init);

  // -----------------------------------------------
  // Optimizer accessors
  // -----------------------------------------------
  /** Get weights optimizer.
   *  Returns a null pointer if the weights are frozen.
   */
  optimizer* get_optimizer();
  /** Get weights optimizer.
   *  Returns a null pointer if the weights are frozen.
   */
  const optimizer* get_optimizer() const;
  /** Set weights optimizer.
   *  The contents of opt are moved to a class member.
   */
  void set_optimizer(std::unique_ptr<optimizer>& opt);

  // -----------------------------------------------
  // Matrix distribution accessors
  // -----------------------------------------------
  El::DistData get_matrix_distribution() const;
  void set_matrix_distribution(El::DistData dist);

  // -----------------------------------------------
  // Setup
  // -----------------------------------------------
  void setup();

  // -----------------------------------------------
  // Weight matrix accessors
  // -----------------------------------------------

  /** Get the weight matrix. */
  AbsDistMat& get_values();
  /** Get the weight matrix. */
  const AbsDistMat& get_values() const;
  /** Set the weight matrix. */
  void set_values(const AbsDistMat& values);

  /** Set a weight value. */
  void set_value(DataType value, int index);
  /** Set an entry in the weight tensor. */
  void set_value(DataType value, std::vector<int> pos);
  /** Set an entry in the weight matrix. */
  void set_value(DataType value, int row, int col);

  /** Reconcile weight values.
   *  If weight values are duplicated across multiple processes, they
   *  are set to the average across the processes.
   */
  void reconcile_values();
  /** Asynchronously reconcile weight values.
   *  If weight values are duplicated across multiple processes, they
   *  are set to the average across the processes.
   */
  void reconcile_values(Al::request& req);

  // -----------------------------------------------
  // Freezing
  // -----------------------------------------------
  /** Disable weight optimization. */
  void freeze() { m_frozen = true; }
  /** Enable weight optimization. */
  void unfreeze() { m_frozen = false; }
  /** Whether weight optimization is enabled. */
  bool is_frozen() const { return m_frozen; }

  // -----------------------------------------------
  // Checkpointing
  // -----------------------------------------------
  bool save_to_checkpoint_shared(persist& p);
  bool load_from_checkpoint_shared(persist& p);
  bool load_from_save(std::string const& ckpt_dir, std::vector<std::string> const& weight_list);
  bool save_to_checkpoint_distributed(persist& p);
  bool load_from_checkpoint_distributed(persist& p);

  /** Write weights to proto file */
  void write_proto(lbann_data::WeightsData* proto) const;

private:

  /** Weights name.
   *  Each set of weights in a model should have a unique,
   *  human-readable name.
   */
  std::string m_name;

  /** Reference to LBANN communicator. */
  lbann_comm* m_comm;

  /** Tensor dimensions corresponding to matrix height.
   *  See the 'get_matrix_height_dims' function.
   */
  std::vector<int> m_matrix_height_dims;
  /** Tensor dimensions corresponding to matrix width.
   *  See the 'get_matrix_width_dims' function.
   */
  std::vector<int> m_matrix_width_dims;
  /** Distribution of weights matrix. */
  El::DistData m_matrix_dist;

  /** Whether weight optimization is disabled. */
  bool m_frozen;

  /** Weight matrix. */
  std::unique_ptr<AbsDistMat> m_values;

  /** Weights initializer.
   *  Default is nullptr, which corresponds to zero initialization.
   */
  std::unique_ptr<weights_initializer> m_initializer;
  /** Weights optimizer.
   *  Default is nullptr, which corresponds to no optimizer.
   */
  std::unique_ptr<optimizer> m_optimizer;

};

} // namespace lbann

#endif // LBANN_WEIGHTS_HPP
