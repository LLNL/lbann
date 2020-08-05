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
////////////////////////////////////////////////////////////////////////////////

#include <iomanip>
#include <sstream>

#include "lbann/callbacks/kfac_test.hpp"
#include "lbann/layers/data_type_layer.hpp"

#include "cblas.h"
#include "lapacke.h"
#include <magma.h>

namespace lbann {
namespace callback {

// Use the MAGMA library instead of OpenBLAS.
#define KFAC_CALLBACK_USE_MAGMA

// Always-assert from the DiHydrogen library.
#define assert_always(...) do {                 \
    if ((__VA_ARGS__) == 0) {                   \
      std::stringstream ss;                     \
      ss << __FILE__ << ":" << __LINE__         \
         << ": " << __func__ << " Assertion "   \
         << #__VA_ARGS__ << " failed.\n";       \
      std::cerr << ss.str();                    \
      abort();                                  \
    } } while (0)


void kfac_test::setup(model *m) {
#if defined(KFAC_CALLBACK_USE_MAGMA)
  const auto ret = magma_init();
  assert_always(ret == MAGMA_SUCCESS);
  // TODO: Call magma_finalize at last
#endif

  const auto comm = m->get_comm();
  if(comm->am_trainer_master()) {
    std::ostringstream oss;
    oss << "K-FAC callback setup:"
        << " damping_act_0=" << m_damping_act_0
        << " damping_act_target=" << m_damping_act_target
        << " damping_err_0=" << m_damping_err_0
        << " damping_err_target=" << m_damping_err_target
        << " damping_warmup_steps=" << m_damping_warmup_steps
        << " kronecker_decay=" << m_kronecker_decay
        << std::endl;
    std::cout << oss.str();
  }
}

void kfac_test::on_backward_prop_end(model *m) {
  // Using a modified Tikhonov damping tequnique from
  // http://arxiv.org/abs/1811.12019.
  const auto get_next_damping =
      [](const double damping_prev,
         const double damping_0,
         const double damping_target,
         const double damping_warmup_steps) {
        const DataType alpha = 2.0 * log10(damping_0 / damping_target) / damping_warmup_steps;
        return (1.0-alpha) * damping_prev + alpha * damping_target;
      };

  m_damping_act = get_next_damping(
      m_damping_act, m_damping_act_0, m_damping_act_target,
      m_damping_warmup_steps);
  m_damping_err = get_next_damping(
      m_damping_err, m_damping_err_0, m_damping_err_target,
      m_damping_warmup_steps);
}

void kfac_test::on_epoch_end(model *m) {
  const auto comm = m->get_comm();
  if(comm->am_trainer_master()) {
    const auto& c = static_cast<const sgd_execution_context&>(m->get_execution_context());
    const auto epoch = c.get_epoch();
    std::ostringstream oss;
    // TODO: Print m_damping_err as well
    oss << "K-FAC callback: changing damping value to " << m_damping_act
        << " at " << epoch << " epochs"
        << " (g_0=" << m_damping_act_0
        << ", g_target=" << m_damping_act_target
        << ", t_warmup=" << m_damping_warmup_steps
        << ")"
        << std::endl;
    std::cout << oss.str();
  }
}

void kfac_test::on_backward_prop_end(model *m, Layer *l) {

  // TODO: Static functions
  const auto get_kronecker_factor =
      [](const El::AbstractMatrix<DataType>& A,
         const DataType alpha) {
        assert_always(A.GetDevice() == El::Device::GPU);
        El::Matrix<DataType, El::Device::GPU> factor(A.Height(), A.Height());
        El::Gemm(
            El::NORMAL, El::TRANSPOSE,
            alpha, A, A,
            El::TypeTraits<DataType>::Zero(), factor);
        return factor;
      };

  // TODO: Static functions
  const auto get_inverse =
      [](const El::Matrix<DataType, El::Device::GPU>& A,
         const bool report_time=false,
         const DataType damping,
         const std::string layer_name,
         const std::string matrix_name) {
        assert_always(A.Width() == A.Height());
        El::Matrix<DataType, El::Device::GPU> Ainv(A);

        const double t_start = get_time();
        kfac_test_add_to_diagonal(
            Ainv.Buffer(), Ainv.Height(), damping);

#if defined(KFAC_CALLBACK_USE_MAGMA)
        const magma_uplo_t uplo = MagmaLower;
#else
        const int matrix_layout = LAPACK_COL_MAJOR;
        const char uplo = 'L';
        El::Matrix<DataType> AinvCPU(Ainv.Height(), Ainv.Width());
#endif

        const double t_damping = get_time();

#if defined(KFAC_CALLBACK_USE_MAGMA)
        { // TODO: CHECK_MAGMA
          magma_int_t ret;
          magma_spotrf_gpu(
              uplo,
              Ainv.Height(), Ainv.Buffer(),
              Ainv.Height(),
              &ret);

          if(ret != 0) {
            std::cerr << "K-FAC callback: Cholesky decomposition of "
                      << matrix_name << " failed at "
                      << layer_name << std::endl;
            abort();
          }
        }

#else
        El::Copy(Ainv, AinvCPU);
        // TODO: CHECK_LAPACK
        LAPACKE_spotrf(
            matrix_layout,
            uplo,
            AinvCPU.Height(), AinvCPU.Buffer(),
            AinvCPU.Height() );
        El::Copy(AinvCPU, Ainv);
#endif

        const double t_spotrf = get_time();

#if defined(KFAC_CALLBACK_USE_MAGMA)
        { // TODO: CHECK_MAGMA
          magma_int_t ret;
          magma_spotri_gpu(
              uplo,
              Ainv.Height(), Ainv.Buffer(),
              Ainv.Height(),
              &ret);
          assert_always(ret == 0);
        }

#else
        El::Copy(Ainv, AinvCPU);
        // TODO: CHECK_LAPACK
        LAPACKE_spotri(
            matrix_layout,
            'L',
            AinvCPU.Height(), AinvCPU.Buffer(),
            AinvCPU.Height() );
        El::Copy(AinvCPU, Ainv);
#endif

        const double t_spotri = get_time();
        kfac_test_fill_upper_tri(Ainv.Buffer(), Ainv.Height());

        const double t_fill = get_time();

        if(report_time) {
          std::cout << "K-FAC callback: get_inverse of"
                    << " " << A.Height() << "x" << A.Width()
#if defined(KFAC_CALLBACK_USE_MAGMA)
                    << " using MAGMA"
#else
                    << " using OpenBLAS @ " << openblas_get_num_threads() << " threads"
#endif
                    << " (damping=" << damping << "): "
                    << " t_damping=" << (t_damping-t_start)
                    << ", t_spotrf=" << (t_spotrf-t_damping)
                    << ", t_spotri=" << (t_spotri-t_spotrf)
                    << ", t_fill=" << (t_fill-t_spotri)
                    << std::endl;
        }

        return Ainv;
      };

  // TODO: Static functions
  const auto compute_pi =
      [](const El::Matrix<DataType, El::Device::GPU> A,
         const El::Matrix<DataType, El::Device::GPU> G) {
        // OPTIMIZE: El::Trace is defined but not implemented yet.
        const auto get_trace =
            [](const El::Matrix<DataType, El::Device::GPU> X) {
              const El::Matrix<DataType> XCPU(X);
              DataType s = 0.0;
              for(int i = 0; i < XCPU.Height(); i++)
                s += XCPU(i, i);
              return s;
            };
        return sqrt((get_trace(A)/A.Height())/(get_trace(G)/G.Height()));
      };

  const auto comm = m->get_comm();
  if(l->get_type() == "fully connected") {

    // Get the layer ID
    const auto layers = m->get_layers();
    const auto layer_it_in_list = std::find(layers.begin(), layers.end(), l);
    assert_always(layer_it_in_list != layers.end());
    const size_t layer_id = std::distance(layers.begin(), layer_it_in_list);

    // Get activations, errors, and gradients
    assert_always(l->get_num_parents() == 1);
    assert_always(l->get_num_children() == 1);
    const auto parent = l->get_parent_layers()[0];
    const auto child = l->get_child_layers()[0];
    const auto& dtl_parent = dynamic_cast<const data_type_layer<DataType>&>(*parent);
    const auto& dtl_child = dynamic_cast<const data_type_layer<DataType>&>(*child);
    const El::AbstractMatrix<DataType>& activations = dtl_parent.get_local_activations();
    const El::AbstractMatrix<DataType>& error_signals = dtl_child.get_local_error_signals();
    const auto mini_batch_size = dtl_parent.get_activations().Width();
    assert_always(mini_batch_size == dtl_child.get_error_signals().Width());
    // The current implementation assumes that bias is not used in any layers.
    assert_always(l->num_weights() == 1);
    auto& w = l->get_weights(0);
    optimizer *opt = w.get_optimizer();
    auto* dto = dynamic_cast<data_type_optimizer<DataType>*>(opt);
    El::Matrix<DataType, El::Device::GPU> gradient = dto->get_gradient().Matrix();
    assert_always(activations.Height() == gradient.Width());
    assert_always(error_signals.Height() == gradient.Height());

    // Compute Kronecker factors, assuming that error_signals are
    // already multiplied by 1/N in the loss layer.
    auto A = get_kronecker_factor(activations, 1.0/mini_batch_size);
    auto G = get_kronecker_factor(error_signals, mini_batch_size);
    // OPTIMIZE: Communicate only the lower triangulars
    comm->allreduce((El::AbstractMatrix<DataType>&) A, comm->get_trainer_comm());
    comm->allreduce((El::AbstractMatrix<DataType>&) G, comm->get_trainer_comm());

    // Compute exponential moving average of the factors
    if(m_kronecker_average.find(layer_id) == m_kronecker_average.end())
      m_kronecker_average.emplace(layer_id, std::make_pair(A, G));
    auto& AGave = (*m_kronecker_average.find(layer_id)).second;
    auto& Aave = AGave.first;
    auto& Gave = AGave.second;
    kfac_test_update_kronecker_average(
        Aave.Buffer(), A.Buffer(), A.Height()*A.Width(), m_kronecker_decay);
    kfac_test_update_kronecker_average(
        Gave.Buffer(), G.Buffer(), G.Height()*G.Width(), m_kronecker_decay);

    // Compute the pi constant
    const DataType pi = m_use_pi ? compute_pi(Aave, Gave) : 1.0;

    // Compute the inverse of the factors
    const bool print_time = comm->am_trainer_master() && m_print_time;
    // Since setting different damping constants for A and G is an
    // alternative heuristics to pi, they should be the same if pi is used.
    if(m_use_pi)
      assert_always(m_damping_act == m_damping_err);
    const auto Ainv = get_inverse(Aave, print_time, DataType(m_damping_act*pi), l->get_name(), "A");
    const auto Ginv = get_inverse(Gave, print_time, DataType(m_damping_err/pi), l->get_name(), "G");

    // Compute preconditioned gradients
    El::Matrix<DataType, El::Device::GPU> Gg(G.Height(), gradient.Width());
    El::Gemm(
        El::NORMAL, El::NORMAL,
        El::TypeTraits<DataType>::One(), Ginv, gradient,
        El::TypeTraits<DataType>::Zero(), Gg);
    El::Matrix<DataType, El::Device::GPU> Fgrad(G.Height(), A.Width());
    El::Gemm(
        El::NORMAL, El::NORMAL,
        El::TypeTraits<DataType>::One(), Gg, Ainv,
        El::TypeTraits<DataType>::Zero(), Fgrad);
    assert_always(Fgrad.Height() == gradient.Height());
    assert_always(Fgrad.Width() == gradient.Width());

    // Apply preconditioned grads
    DataType dst_scale = El::TypeTraits<DataType>::Zero(),
        gradient_scale = El::TypeTraits<DataType>::One();
    auto& grad_buffer = opt->get_gradient_buffer(
        dst_scale, gradient_scale, false);
    El::Copy(Fgrad, grad_buffer.Matrix());

    // Damp matrices for debugging
    if(comm->am_trainer_master() && m_print_matrix) {
      if(comm->am_trainer_master()) {
        std::cout << std::endl;
        El::Print(A, "A");
        std::cout << std::endl;
        El::Print(G, "G");
        std::cout << std::endl;
        El::Print(Aave, "Aave");
        std::cout << std::endl;
        El::Print(Gave, "Gave");
        std::cout << std::endl;
        El::Print(Ainv, "Ainv");
        std::cout << std::endl;
        El::Print(Ginv, "Ginv");
        std::cout << std::endl;
        El::Print(gradient, "grad");
        std::cout << std::endl;
        El::Print(Fgrad, "Fgrad");
        std::cout << std::endl;
      }
    }

    // damp L2 norm of matrices
    if(comm->am_trainer_master() && m_print_matrix_summary) {

      const auto get_stat =
          [](const El::Matrix<DataType, El::Device::GPU>& X,
             const char *name) {
            El::Matrix<DataType> XCPU(X);
            const auto nrm2 = El::Nrm2(El::Reshape(XCPU.Height()*XCPU.Width(), 1, XCPU));
            std::ostringstream oss;
            oss << name
                << "("
                << X.Height()
                << "x"
                << X.Width()
                << ")="
                << std::setprecision(2)
                << std::scientific
                << nrm2;
            return oss.str();
          };

      const auto &dtw = dynamic_cast<data_type_weights<DataType>*>(&w);
      const auto &weights = dtw->get_values();
      std::ostringstream oss;
      oss << "K-FAC callback: L2 norm @ "<< l->get_name() << ": "
          << get_stat(weights.LockedMatrix(), "W")
          << ", " << get_stat(activations, "acts")
          << ", " << get_stat(error_signals, "errs")
          << ", " << get_stat(A, "A")
          << ", " << get_stat(G, "G")
          << ", " << get_stat(Aave, "Aave")
          << ", " << get_stat(Gave, "Gave")
          << ", " << get_stat(Ainv, "Ainv")
          << ", " << get_stat(Ginv, "Ginv")
          << ", " << get_stat(gradient, "grad")
          << ", " << get_stat(Fgrad, "Finvgrad")
          << ", pi=" << pi
          << std::endl;
      std::cout << oss.str();
    }

  }

}

std::unique_ptr<callback_base>
build_kfac_test_callback_from_pbuf(
    const google::protobuf::Message& proto_msg,
    const std::shared_ptr<lbann_summary>&) {
  using MsgType = lbann_data::Callback::CallbackKFACTest;
  using CallbackType = kfac_test;
  const auto& params = dynamic_cast<const MsgType&>(proto_msg);
  double damping_act_0 = params.damping_act_0();
  double damping_err_0 = params.damping_err_0();
  if(damping_act_0 == 0.0) damping_act_0 = kfac_test::damping_0_default;
  if(damping_err_0 == 0.0) damping_err_0 = kfac_test::damping_0_default;
  double damping_act_target = params.damping_act_target();
  double damping_err_target = params.damping_err_target();
  if(damping_act_target == 0.0) damping_act_target = kfac_test::damping_target_default;
  if(damping_err_target == 0.0) damping_err_target = kfac_test::damping_target_default;
  double damping_warmup_steps = params.damping_warmup_steps();
  if(damping_warmup_steps == 0.0) damping_warmup_steps = kfac_test::damping_warmup_steps_default;
  double kronecker_decay = params.kronecker_decay();
  if(kronecker_decay == 0.0)
    kronecker_decay = kfac_test::kronecker_decay_default;
  const bool print_time = params.print_time();
  const bool print_matrix = params.print_matrix();
  const bool print_matrix_summary = params.print_matrix_summary();
  const bool use_pi = params.use_pi();
  return make_unique<CallbackType>(
      damping_act_0, damping_err_0,
      damping_act_target, damping_err_target,
      damping_warmup_steps,
      kronecker_decay,
      print_time, print_matrix, print_matrix_summary,
      use_pi);
}

#undef KFAC_CALLBACK_USE_MAGMA

} // namespace callback
} // namespace lbann
