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
  std::cerr << "kfac_test::setup" << std::endl;
  // TODO: Call magma_finalize at last
#endif
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

  const auto get_inverse =
      [](const El::Matrix<DataType, El::Device::GPU>& A,
         const bool report_time=false,
         const DataType damping) {
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
          assert_always(ret == 0);
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
          std::cout << "get_inverse of"
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

  auto comm = m->get_comm();
  if(l->get_type() == "fully connected") {
    assert_always(l->get_num_parents() == 1);
    assert_always(l->get_num_children() == 1);
    const auto parent = l->get_parent_layers()[0];
    const auto child = l->get_child_layers()[0];
    const auto& dtl_parent = dynamic_cast<const data_type_layer<DataType>&>(*parent);
    const auto& dtl_child = dynamic_cast<const data_type_layer<DataType>&>(*child);
    const El::AbstractMatrix<DataType>& activations = dtl_parent.get_local_activations();
    const El::AbstractMatrix<DataType>& error_signals = dtl_child.get_local_error_signals();

    auto& w = l->get_weights(0);
    optimizer *opt = w.get_optimizer();
    auto* dto = dynamic_cast<data_type_optimizer<DataType>*>(opt);
    El::Matrix<DataType, El::Device::GPU> gradient = dto->get_gradient().Matrix();
    assert_always(activations.Height() == gradient.Width());
    assert_always(error_signals.Height() == gradient.Height());

    // TODO: Use the mini-batch size
    const DataType alpha = DataType(1.0/activations.Width()/comm->get_procs_per_trainer());
    auto A = get_kronecker_factor(activations, alpha);
    auto G = get_kronecker_factor(error_signals, alpha);
    // OPTIMIZE: Communicate only the lower triangulars
    const bool print_time = comm->am_trainer_master() && m_print_time;
    comm->allreduce((El::AbstractMatrix<DataType>&) A, comm->get_trainer_comm());
    comm->allreduce((El::AbstractMatrix<DataType>&) G, comm->get_trainer_comm());
    const auto Ainv = get_inverse(A, print_time, DataType(m_damping));
    const auto Ginv = get_inverse(G, print_time, DataType(m_damping));

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
        El::Print(Ainv, "Ainv");
        std::cout << std::endl;
        El::Print(Ginv, "Ginv");
        std::cout << std::endl;
        El::Print(gradient, "gradient");
        std::cout << std::endl;
        El::Print(Fgrad, "Fgrad");
        std::cout << std::endl;
      }
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
  double damping = params.damping();
  bool print_time = params.print_time();
  bool print_matrix = params.print_matrix();
  if(damping == 0.0)
    damping = 0.03;
  return make_unique<CallbackType>(damping, print_time, print_matrix);
}

#undef KFAC_CALLBACK_USE_MAGMA

} // namespace callback
} // namespace lbann
