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

#include "lbann/callbacks/kfac.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/learning/fully_connected.hpp"
#include "lbann/layers/learning/convolution.hpp"
#include "lbann/layers/regularizers/batch_normalization.hpp"
#include "lbann/utils/im2col.hpp"

#include "cblas.h"
#include "lapacke.h"

namespace lbann {
namespace callback {

void kfac::setup(model *m) {
  const auto v2s =
      [](const std::vector<double> v) {
        std::ostringstream oss;
        for(auto i = v.begin(); i != v.end(); i++) {
          if(i != v.begin())
            oss << ",";
          oss << *i;
        }
        return oss.str();
      };

  const auto comm = m->get_comm();
  if(comm->am_trainer_master()) {
    std::ostringstream oss;
    oss << "K-FAC callback setup:"
        << " damping_act=" << v2s(m_damping_act_params)
        << " damping_err=" << v2s(m_damping_err_params)
        << " damping_bn_act=" << v2s(m_damping_bn_act_params)
        << " damping_bn_err=" << v2s(m_damping_bn_err_params)
        << " damping_warmup_steps=" << m_damping_warmup_steps
        << " kronecker_decay=" << m_kronecker_decay
        << std::endl;
    std::cout << oss.str();
  }
}

void kfac::on_epoch_end(model *m) {
  const auto comm = m->get_comm();
  if(comm->am_trainer_master()) {
    const auto& c = static_cast<const sgd_execution_context&>(m->get_execution_context());
    const auto epoch = c.get_epoch();
    std::ostringstream oss;
    oss << "K-FAC callback: damping_value="
        << m_damping_act << " (act)"
        << ", " << m_damping_err << " (err)"
        << ", " << m_damping_bn_act << " (bn_act)"
        << ", " << m_damping_bn_err << " (bn_err)"
        << ", update_interval=" << m_update_interval
        << " at " << epoch << " epochs"
        << std::endl;
    std::cout << oss.str();
  }
}

void kfac::on_backward_prop_end(model *m) {
  // Update the damping value
  // using a modified Tikhonov damping tequnique from
  // http://arxiv.org/abs/1811.12019
  const auto get_next_damping =
      [](const double damping_prev,
         const std::vector<double> damping_params,
         const double damping_warmup_steps) {
        if(damping_params.size() == 1)
          return damping_params[0];
        const DataType alpha = 2.0 * log10(damping_params[0] / damping_params[1]) / damping_warmup_steps;
        return (1.0-alpha) * damping_prev + alpha * damping_params[1];
      };
  m_damping_act = get_next_damping(
      m_damping_act, m_damping_act_params, m_damping_warmup_steps);
  m_damping_err = get_next_damping(
      m_damping_err, m_damping_err_params, m_damping_warmup_steps);
  m_damping_bn_act = get_next_damping(
      m_damping_bn_act, m_damping_bn_act_params, m_damping_warmup_steps);
  m_damping_bn_err = get_next_damping(
      m_damping_bn_err, m_damping_bn_err_params, m_damping_warmup_steps);

  // Update the udpate interval
  if(m_update_intervals.size() == 1)
    m_update_interval = m_update_intervals[0];
  else {
    const auto& c = static_cast<const sgd_execution_context&>(m->get_execution_context());
    const auto num_steps = c.get_step();
    m_update_interval = m_update_intervals[0]
        + ((double) m_update_intervals[1]-m_update_intervals[0])
        * std::min((double) num_steps/ m_update_interval_steps, 1.0);
  }

  // Get some configs
  const auto comm = m->get_comm();
  const auto& context = static_cast<const sgd_execution_context&>(m->get_execution_context());
  const size_t num_steps = context.get_step();
  const auto layers = m->get_layers();

  // List up layers to be updated
  if(m_blocks.size() == 0){
    const size_t num_procs = comm->get_procs_per_trainer();
    std::unordered_map<std::string, int> proc_ranks;
    for(auto i_layer = layers.begin(); i_layer != layers.end(); i_layer++) {
      const size_t layer_id = std::distance(layers.begin(), i_layer);
      const auto &l = *i_layer;
      const auto l_fc = dynamic_cast<fully_connected_layer<DataType, data_layout::DATA_PARALLEL, El::Device::GPU>*>(l);
      const auto l_conv = dynamic_cast<convolution_layer<DataType, data_layout::DATA_PARALLEL, El::Device::GPU>*>(l);
      const auto l_bn = dynamic_cast<batch_normalization_layer<DataType, data_layout::DATA_PARALLEL, El::Device::GPU>*>(l);
      const bool is_fc = (l_fc != nullptr);
      const bool is_conv = (l_conv != nullptr);
      const bool is_bn = (l_bn != nullptr);
      if(!(is_fc || is_conv || is_bn))
        continue;

      std::string proc_rank_key = "all";
      if(m_inverse_strategy == EACH)
        proc_rank_key = (is_fc ? "fc" : (is_conv ? "conv" : "bn"));
      if(proc_ranks.find(proc_rank_key) == proc_ranks.end())
        proc_ranks[proc_rank_key] = 0;
      int& proc_rank = proc_ranks[proc_rank_key];

      // Check layer property
      const auto parent = l->get_parent_layers()[0];
      const auto child = l->get_child_layers()[0];
      const auto& dtl_parent = dynamic_cast<const data_type_layer<DataType>&>(*parent);
      const auto& dtl_child = dynamic_cast<const data_type_layer<DataType>&>(*child);
      const El::AbstractMatrix<DataType>& local_activations = dtl_parent.get_local_activations();
      const El::AbstractMatrix<DataType>& local_errors = dtl_child.get_local_error_signals();
      if(l->get_num_parents() != 1 || l->get_num_children() != 1) {
        std::stringstream err;
        err << "The K-FAC callback only supports layers who have exact one parent and child."
            << " layer: " << l->get_name()
            << ", #parent: " << l->get_num_parents()
            << ", #child: " << l->get_num_children();
        LBANN_ERROR(err.str());
      }
      if(local_activations.GetDevice() != El::Device::GPU
         || local_errors.GetDevice() != El::Device::GPU) {
        std::stringstream err;
        err << "The K-FAC callback only supports GPU layers."
            << " layer: " << l->get_name();
        LBANN_ERROR(err.str());
      }

      struct kfac_layer_metadata metadata;
      metadata.layer_id = layer_id;
      metadata.is_fc = false;
      metadata.is_conv = false;
      metadata.is_bn_after_fc = false;
      metadata.is_bn_after_conv = false;
      metadata.proc_rank = proc_rank;

      if(is_fc) {
        metadata.is_fc = true;

      } else if(is_conv) {
        size_t spatial_input_prod = 1, spatial_output_prod = 1;
        // std::accumulate might overflow on huge 3D layers
        std::vector<int> input_spatial_dims, output_spatial_dims;
        const auto input_dims = l->get_input_dims();
        for(auto i = input_dims.begin()+1; i != input_dims.end(); i++) {
          spatial_input_prod *= *i;
          input_spatial_dims.push_back(*i);
        }
        const auto output_dims = l->get_output_dims();
        for(auto i = output_dims.begin()+1; i != output_dims.end(); i++) {
          spatial_output_prod *= *i;
          output_spatial_dims.push_back(*i);
        }

        if(input_dims.size() != 3 && input_dims.size() != 4) {
          std::stringstream err;
          err << "The K-FAC callback only supports 2D or 3D tensors."
              << " layer: " << l->get_name()
              << ", input_dims: ";
          for(auto i = input_dims.begin(); i != input_dims.end(); i++)
            err << (std::distance(input_dims.begin(), i) > 0 ? "," : "") << *i;
          LBANN_ERROR(err.str());
        }

        metadata.l_conv = l_conv;
        metadata.is_conv = true;
        metadata.conv_input_spatial_prod = spatial_input_prod;
        metadata.conv_output_spatial_prod = spatial_output_prod;
        metadata.conv_input_spatial_dims = input_spatial_dims;
        metadata.conv_output_spatial_dims = output_spatial_dims;

      } else if(is_bn) {
        const bool is_bn_after_fc =
            (dynamic_cast<const fully_connected_layer<DataType,
             data_layout::DATA_PARALLEL, El::Device::GPU>*>(parent) != nullptr);
        const bool is_bn_after_conv =
            (dynamic_cast<const convolution_layer<DataType,
             data_layout::DATA_PARALLEL, El::Device::GPU>*>(parent) != nullptr);
        if(!is_bn_after_fc && !is_bn_after_conv) {
          std::stringstream err;
          err << "The K-FAC callback only supports batch-normalization layers after "
              << "fully-connected layers or convolutional layers."
              << " layer: " << l->get_name()
              << " parent type: " << parent->get_type();
          LBANN_ERROR(err.str());
        }

        size_t num_channels;
        size_t spatial_prod;
        if(is_bn_after_fc) {
          num_channels = local_activations.Height();
          spatial_prod = 1;
          assert(num_channels == (size_t) local_errors.Height());
        } else {
          const auto input_dims = l->get_input_dims();
          num_channels = input_dims[0];
          spatial_prod = 1;
          // std::accumulate might overflow for large 3D layers
          for(auto i = input_dims.begin()+1; i != input_dims.end(); i++)
            spatial_prod *= *i;
        }

        metadata.is_bn_after_fc = is_bn_after_fc;
        metadata.is_bn_after_conv = is_bn_after_conv;
        metadata.bn_num_channels = num_channels;
        metadata.bn_spatial_prod = spatial_prod;
      }

      m_blocks.emplace_back(l, this, metadata);
      if(m_inverse_strategy != ROOT)
        proc_rank = (proc_rank+1)%num_procs;
    }

    if(comm->am_trainer_master()) {
      for(const auto& block : m_blocks) {
        const auto& metadata = block.get_metadata();
        std::cout << "K-FAC callback setup: "
                  << "name=" << layers[metadata.layer_id]->get_name()
                  << ", id=" << metadata.layer_id
                  << ", is_fc=" << metadata.is_fc
                  << ", is_conv=" << metadata.is_conv
                  << ", is_bn_after_fc=" << metadata.is_bn_after_fc
                  << ", is_bn_after_conv=" << metadata.is_bn_after_conv
                  << ", proc_rank=" << metadata.proc_rank
                  << std::endl;
      }
    }
  }

  // Step 1: Ensure that each process has averaged Kronecker factors
  // for the model-parallel part.
  for(auto& block : m_blocks) {
    const auto& metadata = block.get_metadata();
    const bool is_update_required =
        ((num_steps%m_update_interval_steps) == 0
         || !block.has_kronecker_inverse());
    if(!is_update_required)
      continue;
    if(metadata.is_fc || metadata.is_conv)
      block.update_kronecker_factors_fc_conv(
          comm,
          m_kronecker_decay,
          m_print_matrix, m_print_matrix_summary);
    else
      block.update_kronecker_factors_bn(
          comm,
          m_kronecker_decay,
          m_print_matrix, m_print_matrix_summary);
  }

  // Step 2: Model-parallel inverse computation
  for(auto& block : m_blocks) {
    const auto& metadata = block.get_metadata();
    const bool is_update_required =
        ((num_steps%m_update_interval_steps) == 0
         || !block.has_kronecker_inverse())
        && comm->get_rank_in_trainer() == metadata.proc_rank;
    if(!is_update_required)
      continue;
    if(metadata.is_fc || metadata.is_conv)
      block.update_kronecker_inverse_fc_conv(
          comm, m_use_pi,
          m_damping_act, m_damping_err,
          m_print_matrix, m_print_matrix_summary,
          m_print_time);
    else
      block.update_kronecker_inverse_bn(
          comm, m_use_pi,
          m_damping_bn_act, m_damping_bn_err,
          m_print_matrix, m_print_matrix_summary,
          m_print_time);
  }

  // Step 3: All-gather of each preconditioned gradient tensor
  for(auto& block : m_blocks) {
    const auto& metadata = block.get_metadata();
    if(metadata.is_fc || metadata.is_conv)
      block.update_preconditioned_grads_fc_conv(comm);
    else
      block.update_preconditioned_grads_bn(comm);
  }

}

void kfac::get_kronecker_factor_fc(
    El::AbstractMatrix<DataType>& factor,
    const El::AbstractMatrix<DataType>& activations,
    const DataType alpha) {
  assert(activations.GetDevice() == El::Device::GPU);
  assert(factor.Height() == activations.Height());
  assert(factor.Width() == activations.Height());
  El::Gemm(
      El::NORMAL, El::TRANSPOSE,
      alpha, activations, activations,
      El::TypeTraits<DataType>::Zero(), factor);
}

void kfac::get_kronecker_factor_conv(
    El::Matrix<DataType, El::Device::GPU>& factor,
    El::Matrix<DataType, El::Device::GPU>& Acol,
    const El::Matrix<DataType, El::Device::GPU>& activations,
    const DataType alpha,
    const size_t local_batch_size, const size_t num_channels,
    const std::vector<int> spatial_dims,
    const convolution_layer<DataType, data_layout::DATA_PARALLEL, El::Device::GPU> *l_conv,
    const bool use_im2col,
    const cudaStream_t& stream) {
  assert(factor.GetDevice() == El::Device::GPU);
  assert(activations.GetDevice() == El::Device::GPU);

  const auto dilations = l_conv->get_dilations();
  for(auto i = dilations.begin(); i != dilations.end(); i++)
    if(*i != 1) {
      std::stringstream err;
      err << "The K-FAC callback onky supports dilation width of 1."
          << " layer: " << l_conv->get_name();
      LBANN_ERROR(err.str());
    }

  if(use_im2col) {
    im2col(activations, Acol,
           num_channels, spatial_dims.size(),
           &(spatial_dims[0]),
           &(l_conv->get_pads()[0]),
           &(l_conv->get_conv_dims()[0]),
           &(l_conv->get_strides()[0]),
           stream);
  } else {
    size_t spatial_prod = 1;
    for(auto i = spatial_dims.begin(); i != spatial_dims.end(); i++)
      spatial_prod *= *i;
    assert((size_t) Acol.Height() == num_channels);
    assert((size_t) Acol.Width() == local_batch_size*spatial_prod);
    conv_transpose(
        activations.LockedBuffer(), Acol.Buffer(),
        local_batch_size, num_channels, spatial_prod,
        stream);
  }

  assert(factor.Height() == Acol.Height());
  assert(factor.Width() == Acol.Height());
  El::Gemm(
      El::NORMAL, El::TRANSPOSE,
      alpha, Acol, Acol,
      El::TypeTraits<DataType>::Zero(), factor);
}

void kfac::get_matrix_inverse(
    El::Matrix<DataType, El::Device::GPU>& Ainv,
    El::Matrix<DataType, El::Device::GPU>& Linv,
    const El::Matrix<DataType, El::Device::GPU>& A,
    const bool report_time,
    const DataType damping,
    const DataType damping_bn_err,
    const bool is_bn,
    const cudaStream_t& stream) {
  assert(A.Height() == A.Width());
  assert(Ainv.Height() == A.Height());
  assert(Ainv.Width() == A.Height());
  El::Copy(A, Ainv);

  const double t_start = get_time();

  if(damping > 0 || damping_bn_err > 0)
    add_to_diagonal(
        Ainv.Buffer(), Ainv.Height(),
        damping, damping_bn_err,
        is_bn,
        stream);

  const double t_damping = get_time();

  const auto uplo = El::UpperOrLowerNS::LOWER;
  El::Cholesky(
      uplo,
      (El::AbstractMatrix<DataType> &) Ainv);

  const double t_spotrf = get_time();

  assert(Linv.Height() == Ainv.Height());
  assert(Linv.Width() == Ainv.Height());
  identity(Linv.Buffer(), Linv.Height(), stream);
  El::Trsm(
      El::LeftOrRightNS::LEFT,
      uplo,
      El::OrientationNS::NORMAL,
      El::UnitOrNonUnitNS::NON_UNIT,
      El::TypeTraits<DataType>::One(),
      (const El::AbstractMatrix<DataType> &) Ainv,
      (El::AbstractMatrix<DataType> &) Linv,
      true);
  El::Gemm(
      El::TRANSPOSE, El::NORMAL,
      El::TypeTraits<DataType>::One(), Linv, Linv,
      El::TypeTraits<DataType>::Zero(), Ainv);

  const double t_spotri = get_time();

  // TRSM+GEMM is equivalent to POTRI+fill_upper_tri.
  // fill_upper_tri(Ainv.Buffer(), Ainv.Height());

  const double t_fill = get_time();

  if(report_time) {
    std::cout << "K-FAC callback: get_matrix_inverse of"
              << " " << A.Height() << "x" << A.Width()
              << " using Hydrogen"
              << " (damping=" << damping << "): "
              << " t_damping=" << (t_damping-t_start)
              << ", t_spotrf=" << (t_spotrf-t_damping)
              << ", t_spotri=" << (t_spotri-t_spotrf)
              << ", t_fill=" << (t_fill-t_spotri)
              << std::endl;
  }

  // TODO: Check whether this is actually needed.
  CHECK_CUDA(cudaStreamSynchronize(stream));
}

double kfac::compute_pi(const El::Matrix<DataType, El::Device::GPU>& A,
                        const El::Matrix<DataType, El::Device::GPU>& G,
                        El::Matrix<DataType, El::Device::GPU>& ws,
                        const cudaStream_t& stream) {
  assert(ws.Height() >= A.Height()*2+1);
  assert(ws.Height() >= G.Height()*2+1);
  // TODO: Replace with El::Trace once GPU matrices get supported.
  const auto get_trace =
      [](const El::Matrix<DataType, El::Device::GPU>& X,
         El::Matrix<DataType, El::Device::GPU>& w,
         const cudaStream_t& s) {
        auto diag = El::View(w, El::IR(0, X.Height()), El::ALL);
        auto ones = El::View(w, El::IR(X.Height(), X.Height()*2), El::ALL);
        auto ret = El::View(w, El::IR(X.Height()*2, X.Height()*2+1), El::ALL);
        get_diagonal(diag.Buffer(), X.LockedBuffer(), X.Height(), s);
        El::Ones(ones, ones.Height(), ones.Width());
        El::Gemm(
            El::TRANSPOSE, El::NORMAL,
            El::TypeTraits<DataType>::One(), diag, ones,
            El::TypeTraits<DataType>::Zero(), ret);
        return El::Matrix<DataType>(ret)(0, 0);
      };
  return sqrt((get_trace(A, ws, stream)/A.Height())/(get_trace(G, ws, stream)/G.Height()));
}

std::string kfac::get_matrix_stat(const El::Matrix<DataType, El::Device::GPU>& X,
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
}

void kfac::allreduce_lower_tri(El::Matrix<DataType, El::Device::GPU>& A,
                               El::Matrix<DataType, El::Device::GPU>& AL,
                               lbann_comm *comm,
                               const cudaStream_t& stream) {
  assert(A.Height() == A.Width());
  assert(AL.Height() == A.Height()*(A.Height()+1)/2);
  assert(AL.Width() == 1);
  pack_lower_tri(AL.Buffer(), A.LockedBuffer(), A.Height(), stream);
  comm->allreduce((El::AbstractMatrix<DataType>&) AL,
                  comm->get_trainer_comm());
  unpack_lower_tri(A.Buffer(), AL.Buffer(), A.Height(), stream);
}

std::unique_ptr<callback_base>
build_kfac_callback_from_pbuf(
    const google::protobuf::Message& proto_msg,
    const std::shared_ptr<lbann_summary>&) {
  using MsgType = lbann_data::Callback::CallbackKFAC;
  using CallbackType = kfac;
  const auto& params = dynamic_cast<const MsgType&>(proto_msg);

  const auto parse_damping_params =
      [](const std::string str) {
        if(str == "")
          return std::vector<double>({kfac::damping_0_default});
        else {
          const auto ret = parse_list<double>(str);
          if(ret.size() > 2)
            LBANN_ERROR("The length of damping vectors should be 1 or 2.");
          return ret;
        }
      };

  const auto parse_update_intervals =
      [](const std::string str) {
        if(str == "")
          return std::vector<size_t>({1});
        else {
          const auto ret = parse_list<size_t>(str);
          if(ret.size() > 2)
            LBANN_ERROR("The length of update interval vectors should be 1 or 2.");
          return ret;
        }
      };

  const std::vector<double> damping_act_params = parse_damping_params(params.damping_act());
  const std::vector<double> damping_err_params = parse_damping_params(params.damping_err());
  const std::vector<double> damping_bn_act_params = parse_damping_params(params.damping_bn_act());
  const std::vector<double> damping_bn_err_params = parse_damping_params(params.damping_bn_err());
  size_t damping_warmup_steps = params.damping_warmup_steps();
  if(damping_warmup_steps == 0) damping_warmup_steps = kfac::damping_warmup_steps_default;
  double kronecker_decay = params.kronecker_decay();
  if(kronecker_decay == 0.0)
    kronecker_decay = kfac::kronecker_decay_default;
  const bool print_time = params.print_time();
  const bool print_matrix = params.print_matrix();
  const bool print_matrix_summary = params.print_matrix_summary();
  const bool use_pi = params.use_pi();
  const std::vector<size_t> update_intervals = parse_update_intervals(params.update_intervals());
  const size_t update_interval_steps = params.update_interval_steps();

  const std::string inverse_strategy_str = params.inverse_strategy();
  kfac_inverse_strategy inverse_strategy;
  if(inverse_strategy_str == "" || inverse_strategy_str == "all")
    inverse_strategy = ALL;
  else if(inverse_strategy_str == "each")
    inverse_strategy = EACH;
  else if(inverse_strategy_str == "root")
    inverse_strategy = ROOT;
  else {
    std::stringstream err;
    err << "Invalid inverse strategy type: "
        << inverse_strategy_str;
    LBANN_ERROR(err.str());
  }

  return make_unique<CallbackType>(
      damping_act_params,
      damping_err_params,
      damping_bn_act_params,
      damping_bn_err_params,
      damping_warmup_steps,
      kronecker_decay,
      print_time, print_matrix, print_matrix_summary,
      use_pi,
      update_intervals, update_interval_steps,
      inverse_strategy);
}

El::Matrix<DataType, El::Device::GPU>& kfac::get_workspace_matrix(
    const std::string key, const size_t height, const size_t width) {
  if(m_workspace.find(key) == m_workspace.end()) {
    m_workspace.emplace(
        key, El::Matrix<DataType, El::Device::GPU>(height, width));
#ifdef HYDROGEN_HAVE_CUB
    m_workspace[key].SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif // HYDROGEN_HAVE_CUB
  }
  auto& ret = m_workspace[key];
  if((size_t) ret.Height() != height || (size_t) ret.Width() != width) {
    // Make sure that no kernels are using this workspace.
    CHECK_CUDA(cudaDeviceSynchronize());
    ret.Resize(height, width);
  }
  return ret;
}

} // namespace callback
} // namespace lbann
