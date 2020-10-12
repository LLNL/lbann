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
//
// callback_kfac .hpp .cpp - Callbacks for the K-FAC 2nd-order opt. method
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_KFAC_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_KFAC_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include "lbann/callbacks/kfac/kfac_block.hpp"

namespace lbann {
namespace callback {

enum kfac_inverse_strategy {
  ALL,  // Apply round-robin assingment to all of the layers. may cause load imbalance.
  EACH, // Apply round-robin assingment to every type of layers. may
  // not work well for small networks.
  ROOT, // Use only the root GPU. This is only for testing.
};

// Forward declarations
// TODO: Remove if kfac_block no longer refers kfac
class kfac_block;

/** Callback hooks for the K-FAC method.
 *
 * Martens, James and Roger Grosse. "Optimizing neural networks with
 * kronecker-factored approximate curvature." International conference
 * on machine learning. 2015.
 *
 * Grosse, Roger, and James Martens. "A kronecker-factored approximate
 * fisher matrix for convolution layers." International Conference on
 * Machine Learning. 2016.
 *
 * Osawa, Kazuki, et al. "Large-scale distributed second-order
 * optimization using kronecker-factored approximate curvature for
 * deep convolutional neural networks." Proceedings of the IEEE
 * Conference on Computer Vision and Pattern Recognition. 2019.
 */
class kfac : public callback_base {
 public:

  /** Constructor.
   */
  kfac(const std::vector<double> damping_act_params,
       const std::vector<double> damping_err_params,
       const std::vector<double> damping_bn_act_params,
       const std::vector<double> damping_bn_err_params,
       const size_t damping_warmup_steps,
       const double kronecker_decay,
       const bool print_time, const bool print_matrix,
       const bool print_matrix_summary,
       const bool use_pi,
       const std::vector<size_t> update_intervals,
       const size_t update_interval_steps,
       const kfac_inverse_strategy inverse_strategy)
  : callback_base(),
    m_damping_act_params(damping_act_params),
    m_damping_err_params(damping_err_params),
    m_damping_bn_act_params(damping_bn_act_params), m_damping_bn_err_params(damping_bn_err_params),
    m_damping_warmup_steps(damping_warmup_steps),
    m_kronecker_decay(kronecker_decay),
    m_print_time(print_time), m_print_matrix(print_matrix),
    m_print_matrix_summary(print_matrix_summary),
    m_use_pi(use_pi),
    m_update_intervals(update_intervals),
    m_update_interval_steps(update_interval_steps),
    m_inverse_strategy(inverse_strategy) {
    m_damping_act = m_damping_act_params[0];
    m_damping_err = m_damping_err_params[0];
    m_damping_bn_act = m_damping_bn_act_params[0];
    m_damping_bn_err = m_damping_bn_err_params[0];
  }
  kfac(const kfac&) = default;
  kfac& operator=(const kfac&) = default;
  kfac* copy() const override { return new kfac(*this); }
  void setup(model *m) override;
  void setup(trainer *t) override {}
  void on_backward_prop_end(model *m) override;
  void on_epoch_end(model *m) override;
  void on_backward_prop_end(model *m, Layer *l) override {}
  std::string name() const override { return "K-FAC test"; }

  /** @brief Gets the Kronecker factor matrix of a FC layer.
   *  The same key is tied with the same matrix instance. */
  El::Matrix<DataType, El::Device::GPU>& get_workspace_matrix(
      const std::string key, const size_t height, const size_t width);

  /** @brief Gets the inverse matrix of A. **/
  static void get_matrix_inverse(
      El::Matrix<DataType, El::Device::GPU>& Ainv,
      El::Matrix<DataType, El::Device::GPU>& Linv,
      const El::Matrix<DataType, El::Device::GPU>& A,
      const bool report_time,
      const DataType damping,
      const DataType damping_bn_err,
      const bool is_bn,
      const cudaStream_t& stream);

  /** @brief Gets statistics of a given matrix. **/
  static std::string get_matrix_stat(
      const El::Matrix<DataType, El::Device::GPU>& X,
      const char *name);

  /** @brief Perform all-reduce on the lower triangular of a symmetric matrix. **/
  static void allreduce_lower_tri(
      El::Matrix<DataType, El::Device::GPU>& A,
      El::Matrix<DataType, El::Device::GPU>& AL,
      lbann_comm *comm,
      const cudaStream_t& stream);

  /** @brief Add the damping value to the diagonal elements of A. **/
  template <typename TensorDataType>
  static void add_to_diagonal(
      TensorDataType * __restrict__ A,
      const size_t height,
      const TensorDataType value,
      const TensorDataType value_bn_err,
      const bool is_bn,
      const cudaStream_t& stream);

  /** @brief Fill the upper trianglar with the lower trianglar. **/
  template <typename TensorDataType>
  static void fill_upper_tri(
      TensorDataType * __restrict__ A,
      const size_t height,
      const cudaStream_t& stream);

  /** @brief Update a Kronecker factor matrix using decay.
   *
   * Aave = Aave * decay + A * (1-decay) **/
  template <typename TensorDataType>
  static void update_kronecker_average(
      TensorDataType * __restrict__ Aave,
      const TensorDataType * __restrict__ A,
      const size_t count, const DataType decay,
      const cudaStream_t& stream);

  /** @brief Substitute the identity matrix.
   *  TODO: Replace with El::Identity<El::Device::GPU>
   *   once it gets supported. **/
  template <typename TensorDataType>
  static void identity(
      TensorDataType * __restrict__ A,
      const size_t height,
      const cudaStream_t& stream);

  /** @brief Pack the lower triangular of a symmetric matrix. **/
  template <typename TensorDataType>
  static void pack_lower_tri(
      TensorDataType * __restrict__ L,
      const TensorDataType * __restrict__ A,
      const size_t height,
      const cudaStream_t& stream);

  /** @brief Unpack the lower triangular of a symmetric matrix. **/
  template <typename TensorDataType>
  static void unpack_lower_tri(
      TensorDataType * __restrict__ A,
      const TensorDataType * __restrict__ L,
      const size_t height,
      const cudaStream_t& stream);

  /** @brief The default parameters of a Tikhonov damping technique. */
  constexpr static const double damping_0_default = 3e-2;
  constexpr static const size_t damping_warmup_steps_default = 100;

  /** @brief The default parameters of the decay factor. */
  constexpr static const double kronecker_decay_default = 0.99;

 private:

  /** @brief Pairs of the initial and the target damping value.
   *  If only one value is specified, it will be used throughout training.
   */
  const std::vector<double> m_damping_act_params, m_damping_err_params,
    m_damping_bn_act_params, m_damping_bn_err_params;

  /** @brief The number of warmup steps of the Tikhnov damping technique. */
  const size_t m_damping_warmup_steps;

  /** @brief The decay factor of kronecker factors. */
  const double m_kronecker_decay;

  /** @brief Knobs to print information for debugging. */
  const bool m_print_time, m_print_matrix, m_print_matrix_summary;

  /** @brief Weather to use the pi constant to adjust the damping
      constant. */
  const bool m_use_pi;

  /** @brief Space-separated pairs of the initial and the target update intervals.
   *If only one value is specified, it will be used throughout
   *training.
   */
  const std::vector<size_t> m_update_intervals;

  /** @brief The number of steps for changing the update interval. */
  const size_t m_update_interval_steps;

  /** @brief The current damping values. */
  double m_damping_act, m_damping_err,
    m_damping_bn_act, m_damping_bn_err;

  /** @brief The current update interval. */
  size_t m_update_interval;

  /** @brief K-FAC per-layer blocks. */
  std::vector<std::shared_ptr<kfac_block>> m_blocks;

  /** @brief Assignment strategy for the model-parallel part. */
  kfac_inverse_strategy m_inverse_strategy;

  std::unordered_map<std::string,
                     El::Matrix<DataType, El::Device::GPU>> m_workspace;
};

// Builder function
std::unique_ptr<callback_base>
build_kfac_callback_from_pbuf(
    const google::protobuf::Message&,std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_KFAC_HPP_INCLUDED
