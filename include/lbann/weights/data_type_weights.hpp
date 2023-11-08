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

#ifndef LBANN_DATA_TYPE_WEIGHTS_HPP
#define LBANN_DATA_TYPE_WEIGHTS_HPP

#include "lbann/optimizers/data_type_optimizer.hpp"
#include "lbann/utils/typename.hpp"
#include "lbann/weights/initializer.hpp"
#include "lbann/weights/weights.hpp"

namespace cereal {
class access;
} // namespace cereal

namespace lbann {

// Forward declaration
// template <typename TensorDataType>
// class data_type_optimizer;

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
template <typename TensorDataType>
class data_type_weights
  : public Cloneable<data_type_weights<TensorDataType>, weights>
{
  using BaseType = Cloneable<data_type_weights<TensorDataType>, weights>;

public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  /** @brief This type. */
  using WeightsType = data_type_weights<TensorDataType>;

  /** @brief The Optimizer type used by this object. */
  using OptimizerType = data_type_optimizer<TensorDataType>;

  /** @brief The Initializer type used by this object. */
  using InitializerType = data_type_weights_initializer<TensorDataType>;

  ///@}

public:
  data_type_weights(lbann_comm& comm);
  data_type_weights(const data_type_weights& other);
  data_type_weights& operator=(const data_type_weights& other);
  virtual ~data_type_weights() = default;

  std::string get_datatype_name() const override
  {
    return TypeName<TensorDataType>();
  }

  bool has_optimizer() const override { return m_optimizer != nullptr; }

  // -----------------------------------------------
  // Dimension accessors
  // -----------------------------------------------
  // -----------------------------------------------
  // Initializer accessors
  // -----------------------------------------------
  /** Get weights initializer. */
  InitializerType* get_initializer() override;
  /** Get weights initializer (const). */
  const InitializerType* get_initializer() const override;
  /** Set weights initializer.
   *  The contents of 'init' are moved to a class member.
   */
  void set_initializer(std::unique_ptr<weights_initializer>&& init) override;

  // -----------------------------------------------
  // Optimizer accessors
  // -----------------------------------------------
  /** Get weights optimizer.
   *  Returns a null pointer if the weights are frozen.
   */
  OptimizerType* get_optimizer() override;
  /** Get weights optimizer.
   *  Returns a null pointer if the weights are frozen.
   */
  const OptimizerType* get_optimizer() const override;
  /** Set weights optimizer.
   *  The contents of opt are moved to a class member.
   */
  void set_optimizer(std::unique_ptr<optimizer>&& opt) override;

  // -----------------------------------------------
  // Weight matrix accessors
  // -----------------------------------------------

  /** Get the weight matrix (a constant, full view of the weights). */
  const AbsDistMatrixType& get_values() const override;

  /** @brief Access the local shard of weight values. If sharded is false,
   *         equivalent to 'get_values'.
   */
  AbsDistMatrixType& get_values_sharded() override;
  AbsDistMatrixType const& get_values_sharded() const override;

  using weights::set_values;
  /** Set the weight matrix. */
  void set_values(const AbsDistMatrixType& values);

  /** Set a weight value. */
  void set_value(TensorDataType value, size_t index);
  /** Set an entry in the weight tensor. */
  void set_value(TensorDataType value, std::vector<size_t> pos);
  /** Set an entry in the weight matrix. */
  void set_value(TensorDataType value, size_t row, size_t col);

  // -----------------------------------------------
  // Weight memory management
  // -----------------------------------------------

  /** @brief Start an asynchronous request for the full view of weights.
   *
   *  This is a noop if the weights are not sharded.
   */
  void request_full_weights_async() const override;
  /** @brief Wait for an asynchronous request for the full view of weights.
   *
   *  This is a noop if the weights are not sharded, or already requested.
   */
  void wait_for_full_weights() const override;
  /** @brief Releases the full view of the weights for memory reclamation. */
  void release_full_weights() const override;

  /** Reconcile weight values.
   *  If weight values are duplicated across multiple processes, they
   *  are set to the average across the processes.
   */
  void reconcile_values() override;
  /** Asynchronously reconcile weight values.
   *  If weight values are duplicated across multiple processes, they
   *  are set to the average across the processes.
   */
  void reconcile_values(Al::request& req) override;

  bool load_from_save(std::string const& ckpt_dir,
                      std::vector<std::string> const& weight_list,
                      El::FileFormat el_mode);
  bool load_from_save(std::string const& ckpt_dir,
                      std::vector<std::string> const& weight_list) override;

  /** Write weights to proto file */
  void write_proto(lbann_data::Weights& proto) const final;

  /** @name Serialization */
  ///@{

  /** @brief Serialize the weights object to the archive.
   *  @tparam ArchiveT (Inferred.) The archive type.
   *  @param[in,out] ar The archive to which to write or from which to
   *                    read.
   */
  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

#ifdef LBANN_HAS_ONNX
  /** @brief Add weights data to onnx graph.
   *  Adds serialized weights into graph initializers. Each initializer
   *  will have the name of the corresponding weights object.
   */
  void fill_onnx_node(onnx::GraphProto& graph) const override;
#endif // LBANN_HAS_ONNX

private:
  friend cereal::access;
  data_type_weights();

  void do_augment_description_(description&) const override;
  void do_setup_() override;
  void do_set_dims_(std::vector<size_t> const& matrix_height_dims,
                    std::vector<size_t> const& matrix_width_dims) override;
  void do_move_values_(data_type_weights& other);
  void do_steal_values_(weights& other) override;

private:
  /** Weight matrix (potentially sharded). */
  std::unique_ptr<AbsDistMatrixType> m_values;

  /** View of the full weight matrix view for layers to use. The field is
   *  mutable because it acts as a cache of a constant view of m_values.
   */
  mutable std::unique_ptr<AbsDistMatrixType> m_values_view;

  /** Weights initializer.
   *  Default is nullptr, which corresponds to zero initialization.
   */
  std::unique_ptr<InitializerType> m_initializer;
  /** Weights optimizer.
   *  Default is nullptr, which corresponds to no optimizer.
   */
  std::unique_ptr<OptimizerType> m_optimizer;

  friend class data_type_optimizer<TensorDataType>;
};

#ifndef LBANN_DATA_TYPE_WEIGHTS_INSTANTIATE
#define PROTO(T) extern template class data_type_weights<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#undef LBANN_INSTANTIATE_CPU_HALF
#undef LBANN_INSTANTIATE_GPU_HALF
#endif // LBANN_DATA_TYPE_WEIGHTS_INSTANTIATE

} // namespace lbann

#endif // LBANN_DATA_TYPE_WEIGHTS_HPP
