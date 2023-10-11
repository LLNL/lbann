////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#include "lbann/base.hpp"
#include "lbann/utils/cloneable.hpp"
#include "lbann/utils/description.hpp"

#include <memory>
#include <string>
#include <vector>
#ifdef LBANN_HAS_ONNX
#include <onnx/onnx_pb.h>
#endif // LBANN_HAS_ONNX

namespace lbann_data {
class Weights;
}

namespace lbann {

namespace Al {
// Forward declaration
struct request;
} // namespace Al

// Forward declaration
class lbann_comm;
class weights;
class weights_initializer;
class optimizer;

/** @brief Smart pointer to manage ownership of a weights object
 *
 *  This should be treated @b exactly like a @c
 *  std::unique_ptr<weights> , i.e. there should be exactly one
 *  instance per pointer and the copy constructor and copy-assignment
 *  operators should never be used. Using this like a @c
 *  std::shared_ptr may lead to unexpected behavior.
 *
 *  The @b only reason this is not a @c std::unique_ptr is because
 *  Cereal cannot natively serialize raw pointers. However, it can
 *  accommodate @c std::weak_ptr . In an ideal world, Cereal would
 *  support a non-owning smart pointer to an object in @c
 *  std::unique_ptr (possibly the experimental @c observer_ptr ), but
 *  we can make do by managing weights with @c std::shared_ptr .
 *
 *  @todo Replace with @c std::unique_ptr<weights> when C++ and Cereal
 *  support @c std::observer_ptr .
 */
using OwningWeightsPtr = std::shared_ptr<weights>;
/** @brief Smart pointer to reference a weights object
 *
 *  See @c OwningWeightsPtr
 *
 *  @todo Replace with @c std::observer_ptr<Weights> when supported by
 *  C++ and Cereal.
 */
using ViewingWeightsPtr = std::weak_ptr<weights>;

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
class weights : public Cloneable<HasAbstractFunction<weights>>
{
private:
  weights();
  // -----------------------------------------------
  // Internal method for setting the comm pointer
  // -----------------------------------------------
  void set_comm(lbann_comm& comm);
  void setup_default_matrix_distribution();

public:
  weights(lbann_comm& comm);
  virtual ~weights() = default;

  /** Set weights name.
   *  Each set of weights in a model should have a unique,
   *  human-readable name.
   */
  void set_name(std::string name) { m_name = name; }

  /** Get weights name. */
  std::string get_name() const { return m_name; }

  lbann_comm& get_comm() const { return *m_comm; }

  /** Human-readable description. */
  description get_description() const;

  /** Get a string representing the weights's datatype. */
  virtual std::string get_datatype_name() const = 0;

  virtual bool has_optimizer() const = 0;

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
  std::vector<size_t> get_dims() const;
  /** Get number of entries in weight tensor. */
  size_t get_size() const;
  /** Get tensor dimensions corresponding to weight matrix height.
   *  The dimensions are sorted in decreasing order of strides. Matrix
   *  rows are fully-packed w.r.t. the matrix height dimensions.
   */
  std::vector<size_t> get_matrix_height_dims() const;
  /** Get tensor dimensions corresponding to weight matrix width.
   *  The dimensions are sorted in decreasing order of strides. Matrix
   *  columns are fully-packed w.r.t. the matrix width dimensions.
   */
  std::vector<size_t> get_matrix_width_dims() const;
  /** Get weight matrix height.
   *  If there are no matrix height dimensions, the height is one.
   */
  size_t get_matrix_height() const;
  /** Get weight matrix width.
   *  If there are no matrix width dimensions, the width is one.
   */
  size_t get_matrix_width() const;
  /** Set weight tensor dimensions.
   *  See the 'get_dims' function for an explanation of the notation.
   */
  void set_dims(std::vector<size_t> matrix_height_dims,
                std::vector<size_t> matrix_width_dims = {});
  /** Set weight tensor dimensions as a 1D tensor. */
  void set_dims(size_t size) { set_dims({size}, {}); }

  // -----------------------------------------------
  // Matrix distribution accessors
  // -----------------------------------------------
  El::DistData get_matrix_distribution() const;
  void set_matrix_distribution(El::DistData dist);

  /** @name Matrix accessors */
  ///@{
  /** @brief Set the values matrix to the given matrix.
   *
   *  The input matrix must be compatible with the established matrix
   *  dimensions. If the data type of the input matrix is different
   *  from that expected by the weights object, they will be cast to
   *  the data type expected by the weights object.
   *
   *  @throws lbann::exception If the input matrix has incompatible
   *                           dimensions.
   *
   *  @todo (trb 05/28/2020): Should this check the DistData of the
   *  input against the expected DistData for the weights object?
   */
  void set_values(El::BaseDistMatrix const& values);

  /** @brief Access the matrix of weights values. */
  virtual El::BaseDistMatrix& get_values() = 0;
  virtual El::BaseDistMatrix const& get_values() const = 0;
  ///@}

  // -----------------------------------------------
  // Initializer accessors
  // -----------------------------------------------
  /** Get weights initializer. */
  virtual weights_initializer* get_initializer() = 0;
  /** Get weights initializer (const). */
  virtual const weights_initializer* get_initializer() const = 0;
  /** Set weights initializer.
   *  The contents of 'init' are moved to a class member.
   */
  virtual void set_initializer(std::unique_ptr<weights_initializer>&& init) = 0;

  // -----------------------------------------------
  // Optimizer accessors
  // -----------------------------------------------
  /** Get weights optimizer.
   *  Returns a null pointer if the weights are frozen.
   */
  virtual optimizer* get_optimizer() = 0;
  /** Get weights optimizer.
   *  Returns a null pointer if the weights are frozen.
   */
  virtual const optimizer* get_optimizer() const = 0;
  /** Set weights optimizer.
   *  The contents of opt are moved to a class member.
   */
  virtual void set_optimizer(std::unique_ptr<optimizer>&& opt) = 0;

  // -----------------------------------------------
  // Setup
  // -----------------------------------------------
  void setup();

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
  // Weight matrix accessors
  // -----------------------------------------------

  /** Reconcile weight values.
   *  If weight values are duplicated across multiple processes, they
   *  are set to the average across the processes.
   */
  virtual void reconcile_values() = 0;
  /** Asynchronously reconcile weight values.
   *  If weight values are duplicated across multiple processes, they
   *  are set to the average across the processes.
   */
  virtual void reconcile_values(Al::request& req) = 0;

  virtual bool load_from_save(std::string const& ckpt_dir,
                              std::vector<std::string> const& weight_list) = 0;

  /** Write weights to proto file */
  virtual void write_proto(lbann_data::Weights& proto) const = 0;

  /** @name Serialization */
  ///@{

  /** @brief Serialize the weights object to the archive.
   *  @tparam ArchiveT (Inferred.) The archive type.
   *  @param[in,out] ar The archive to which to write or from which to
   *                    read.
   */
  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

#ifdef LBANN_HAS_ONNX
  /** @brief Add serialized weights initializers to onnx graph */
  virtual void fill_onnx_node(onnx::GraphProto& graph) const = 0;
#endif // LBANN_HAS_ONNX

  ///@}
  /** @name Expert interface */
  ///@{

  /** @brief Take the values matrix from another weights object.
   *
   *  If the other object has the same underlying DistData, the values
   *  matrix will simply be moved over. In this case, the other object
   *  will not have valid weights after this operation
   *  completes. Otherwise, the values will be copied.
   *
   *  @pre Other weights has the same dimenions.
   *  @post Other weights objects values may be invalidated.
   *
   *  @param[in,out] other The object from which to steal values.
   *
   *  @throws lbann::exception If the weights objects don't have the
   *  same dimensions.
   */
  void steal_values(weights& other);

  ///@}
protected:
  weights(const weights& other) = default;
  weights& operator=(const weights& other) = default;

private:
  virtual void do_augment_description_(description&) const = 0;
  virtual void do_setup_() = 0;

  virtual void do_set_dims_(std::vector<size_t> const& matrix_height_dims,
                            std::vector<size_t> const& matrix_width_dims) = 0;
  virtual void do_steal_values_(weights& other) = 0;

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
  std::vector<size_t> m_matrix_height_dims;
  /** Tensor dimensions corresponding to matrix width.
   *  See the 'get_matrix_width_dims' function.
   */
  std::vector<size_t> m_matrix_width_dims;
  /** Distribution of weights matrix. */
  El::DistData m_matrix_dist;

  /** Whether weight optimization is disabled. */
  bool m_frozen;
};

} // namespace lbann

#endif // LBANN_WEIGHTS_HPP
