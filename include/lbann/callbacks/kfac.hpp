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
#include "lbann/layers/learning/convolution.hpp"

namespace lbann {
namespace callback {

// Add the damping value to the diagonal elements of A.
template <typename TensorDataType>
void kfac_add_to_diagonal(
    TensorDataType * __restrict__ A,
    const size_t height,
    const TensorDataType value,
    const TensorDataType value_bn_err=0,
    const bool is_bn=false);

// Fill the upper trianglar with the lower trianglar.
template <typename TensorDataType>
void kfac_fill_upper_tri(
    TensorDataType * __restrict__ A,
    const size_t height);

// Aave = Aave * decay + A * (1-decay)
template <typename TensorDataType>
void kfac_update_kronecker_average(
    TensorDataType * __restrict__ Aave,
    const TensorDataType * __restrict__ A,
    const size_t count, const DataType decay);

// Transpose NC(D)HW matrix to N(D)HWC.
template <typename TensorDataType>
void kfac_conv_transpose(
    const TensorDataType * __restrict__ activations,
    TensorDataType * __restrict__ act_columns,
    const size_t mini_batch_size, const size_t num_channels,
    const size_t spatial_prod);

// Compute the factor of a batch-normalization layer.
template <typename TensorDataType>
void kfac_compute_bn_factor(
    const TensorDataType * __restrict__ activations,
    const TensorDataType * __restrict__ errors,
    const TensorDataType * __restrict__ scales,
    const TensorDataType * __restrict__ biases,
    TensorDataType * __restrict__ factor,
    const size_t batch_size,
    const size_t num_channels,
    const size_t spatial_prod);

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
  kfac(std::vector<double> damping_act_params,
       std::vector<double> damping_err_params,
       std::vector<double> damping_bn_act_params,
       std::vector<double> damping_bn_err_params,
       double damping_warmup_steps,
       double kronecker_decay,
       bool print_time, bool print_matrix,
       bool print_matrix_summary,
       bool use_pi)
      : callback_base(),
        m_damping_act_params(damping_act_params),
        m_damping_err_params(damping_err_params),
        m_damping_bn_act_params(damping_bn_act_params), m_damping_bn_err_params(damping_bn_err_params),
        m_damping_warmup_steps(damping_warmup_steps),
        m_kronecker_decay(kronecker_decay),
        m_print_time(print_time), m_print_matrix(print_matrix),
        m_print_matrix_summary(print_matrix_summary),
        m_use_pi(use_pi) {
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
  void on_backward_prop_end(model *m, Layer *l) override;
  std::string name() const override { return "K-FAC test"; }

  /** @brief The default parameters of a Tikhonov damping technique. */
  constexpr static const double damping_0_default = 3e-2;
  constexpr static const double damping_warmup_steps_default = 100;

  /** @brief The default parameters of the decay factor. */
  constexpr static const double kronecker_decay_default = 0.99;

 private:

  /** @brief Gets the Kronecker factor matrix of a FC layer. **/
  static El::Matrix<DataType, El::Device::GPU> get_kronecker_factor_fc(
      const El::AbstractMatrix<DataType>& A,
      const DataType alpha);

  /** @brief Gets the Kronecker factor matrix of a convolutional layer. **/
  static El::Matrix<DataType, El::Device::GPU> get_kronecker_factor_conv(
      const El::Matrix<DataType, El::Device::GPU>& A,
      const DataType alpha,
      const size_t local_batch_size, const size_t num_channels,
      const std::vector<int> spatial_dims,
      const convolution_layer<DataType, data_layout::DATA_PARALLEL, El::Device::GPU> *l_conv,
      const bool use_im2col);

  /** @brief Gets the inverse matrix of A. **/
  static El::Matrix<DataType, El::Device::GPU> get_matrix_inverse(
      const El::Matrix<DataType, El::Device::GPU>& A,
      const bool report_time=false,
      const DataType damping=0,
      const DataType damping_bn_err=0,
      const bool is_bn=false);

  /** @brief Returns the pi constant. **/
  static double compute_pi(
      const El::Matrix<DataType, El::Device::GPU>& A,
      const El::Matrix<DataType, El::Device::GPU>& G);

  /** @brief Gets statistics of a given matrix. **/
  static std::string get_matrix_stat(
      const El::Matrix<DataType, El::Device::GPU>& X,
      const char *name);

  /** @brief Pairs of the initial and the target damping value.
   *  If only one value is specified, it will be used throughout trainig.
   */
  const std::vector<double> m_damping_act_params, m_damping_err_params,
    m_damping_bn_act_params, m_damping_bn_err_params;

  /** @brief The number of warmup steps of the Tikhnov damping technique. */
  const double m_damping_warmup_steps;

  /** @brief The decay factor of kronecker factors. */
  const double m_kronecker_decay;

  /** @brief Knobs to print information for debugging. */
  const bool m_print_time, m_print_matrix, m_print_matrix_summary;

  /** @brief Weather to use the pi constant to adjust the damping
      constant. */
  const bool m_use_pi;

  /** @brief The current damping values. */
  double m_damping_act, m_damping_err,
    m_damping_bn_act, m_damping_bn_err;

  /** @brief Exponential moving average of kronecker factors. */
  std::unordered_map<
    size_t,
    std::pair<El::Matrix<DataType, El::Device::GPU>,
              El::Matrix<DataType, El::Device::GPU>>> m_kronecker_average;

};

// Builder function
std::unique_ptr<callback_base>
build_kfac_callback_from_pbuf(
    const google::protobuf::Message&,std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_KFAC_HPP_INCLUDED
