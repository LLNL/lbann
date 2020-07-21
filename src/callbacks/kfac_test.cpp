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

#include "lbann/callbacks/kfac_test.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "cblas.h"
#include "lapacke.h"
#include <iomanip>
#include <sstream>

namespace lbann {
namespace callback {

void kfac_test::on_backward_prop_end(model *m, Layer *l) {

  // TODO: Static functions
  const auto get_kronecker_factor =
      [](const El::AbstractMatrix<DataType>& A,
         const DataType alpha) {
        assert(A.GetLocalDevice() == El::Device::GPU);
        El::Matrix<DataType, El::Device::GPU> factor(A.Height(), A.Height());
        El::Gemm(
            El::NORMAL, El::TRANSPOSE,
            alpha, A, A,
            El::TypeTraits<DataType>::Zero(), factor);
        return factor;
      };

  const auto get_inverse =
      [](const El::Matrix<DataType>& A,
         const bool report_time=false) {
        assert(A.Width() == A.Height());
        El::Matrix<DataType> Ainv(A);

        const double t_start = get_time();

        // TODO: Dynamic scheduling of the damping factor
        const DataType damping = DataType(1e-4);
        // OPTIMIZE
#pragma omp parallel for
        for(int i = 0; i < A.Width(); i++)
          Ainv(i, i) += damping;

        const double t_damping = get_time();

        // TODO: CHECK_LAPACK
        const int matrix_layout = LAPACK_COL_MAJOR;
        const char uplo = 'L';
        LAPACKE_spotrf(
            matrix_layout,
            uplo,
            Ainv.Height(), Ainv.Buffer(),
            Ainv.Height() );

        const double t_spotrf = get_time();

        // TODO: CHECK_LAPACK
        LAPACKE_spotri(
            matrix_layout,
            'L',
            Ainv.Height(), Ainv.Buffer(),
            Ainv.Height() );

        const double t_spotri = get_time();

        if(report_time) {
          std::cerr << "get_inverse of"
                    << " " << A.Height() << "x" << A.Width()
                    << " @ " << openblas_get_num_threads() << " threads: "
                    << " t_damping=" << (t_damping-t_start)
                    << ", t_spotrf=" << (t_spotrf-t_damping)
                    << ", t_spotri=" << (t_spotri-t_spotrf)
                    << std::endl;
        }

        return Ainv;
      };

  auto comm = m->get_comm();
  if(l->get_type() == "fully connected") {
    assert(l->get_num_parents() == 1);
    assert(l->get_num_children() == 1);
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
    assert(activations.Height() == gradient.Width());
    assert(error_signals.Height() == gradient.Height());

    const DataType alpha = DataType(1.0/activations.Width()/comm->get_procs_per_trainer());
    El::Matrix<DataType> A(get_kronecker_factor(activations, alpha));
    El::Matrix<DataType> G(get_kronecker_factor(error_signals, alpha));
    // OPTIMIZE: Communicate only the lower triangulars
    comm->allreduce((El::AbstractMatrix<DataType>&) A, comm->get_trainer_comm());
    comm->allreduce((El::AbstractMatrix<DataType>&) G, comm->get_trainer_comm());
    const auto Ainv = get_inverse(A, comm->am_trainer_master());
    const auto Ginv = get_inverse(G, comm->am_trainer_master());
    const El::Matrix<DataType, El::Device::GPU> AinvG(Ainv);
    const El::Matrix<DataType, El::Device::GPU> GinvG(Ginv);

    El::Matrix<DataType, El::Device::GPU> Gg(G.Height(), gradient.Width());
    El::Gemm(
        El::NORMAL, El::NORMAL,
        El::TypeTraits<DataType>::One(), GinvG, gradient,
        El::TypeTraits<DataType>::Zero(), Gg);

    El::Matrix<DataType, El::Device::GPU> Fgrad(G.Height(), A.Width());
    El::Gemm(
        El::NORMAL, El::NORMAL,
        El::TypeTraits<DataType>::One(),
        Gg, AinvG,
        El::TypeTraits<DataType>::Zero(),
        Fgrad);

    assert(Fgrad.Height() == gradient.Height() && Fgrad.Width() == gradient.Width());

    DataType dst_scale = El::TypeTraits<DataType>::Zero(),
        gradient_scale = El::TypeTraits<DataType>::One();
    auto& grad_buffer = opt->get_gradient_buffer(
        dst_scale, gradient_scale, false);
    El::Copy(Fgrad, grad_buffer.Matrix());

  }

}

} // namespace callback
} // namespace lbann
