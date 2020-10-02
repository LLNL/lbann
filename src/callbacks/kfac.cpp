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
  const auto&& stream = hydrogen::cuda::GetDefaultStream();
  const auto& context = static_cast<const sgd_execution_context&>(m->get_execution_context());
  const size_t num_steps = context.get_step();
  const auto layers = m->get_layers();

  // List up layers to be updated
  if(m_learnable_layers.size() == 0){
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

      if(is_fc || is_conv) {
        const struct kfac_layer_metadata metadata =
            {layer_id, l_conv, is_fc, is_conv, false, false, 0, 0,
             proc_rank};
        m_learnable_layers.push_back(metadata);

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
          const auto input_dims = l->get_input_dims(); // CHW
          num_channels = input_dims[0];
          spatial_prod = 1;
          // std::accumulate might overflow for large 3D layers
          for(auto i = input_dims.begin()+1; i != input_dims.end(); i++)
            spatial_prod *= *i;
        }
        assert(num_channels == (size_t) scale_values.Height());
        assert(num_channels == (size_t) scale_values.LocalHeight());
        assert(num_channels == (size_t) bias_values.Height());
        assert(num_channels == (size_t) bias_values.LocalHeight());

        const struct kfac_layer_metadata metadata = {
          layer_id, nullptr,
          false, false, is_bn_after_fc, is_bn_after_conv,
          num_channels, spatial_prod, proc_rank};
        m_learnable_layers.push_back(metadata);
      }
      if(m_inverse_strategy != ROOT)
        proc_rank = (proc_rank+1)%num_procs;
    }

    if(comm->am_trainer_master()) {
      for(const auto metadata : m_learnable_layers) {
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
  for(const auto metadata : m_learnable_layers) {
    if(!(metadata.is_fc || metadata.is_conv))
      continue;
    const auto &l = layers[metadata.layer_id];
    auto& weights = l->get_weights(0);
    if(l->num_weights() != 1) {
      std::stringstream err;
      err << "The K-FAC callback does not currently support biases."
          << " layer: " << l->get_name()
          << ", #weights: " << l->num_weights();
      LBANN_ERROR(err.str());
    }

    const bool is_update_required = (num_steps%m_update_interval_steps) == 0
        || (m_kronecker_average.find(metadata.layer_id) == m_kronecker_average.end()
            && comm->get_rank_in_trainer() == metadata.proc_rank);
    if(!is_update_required)
      continue;

    // Get activations, errors, and gradients
    const auto parent = l->get_parent_layers()[0];
    const auto child = l->get_child_layers()[0];
    const auto& dtl_parent = dynamic_cast<const data_type_layer<DataType>&>(*parent);
    const auto& dtl_child = dynamic_cast<const data_type_layer<DataType>&>(*child);
    const El::AbstractMatrix<DataType>& local_activations = dtl_parent.get_local_activations();
    const El::AbstractMatrix<DataType>& local_errors = dtl_child.get_local_error_signals();
    const auto mini_batch_size = dtl_parent.get_activations().Width();
    assert(mini_batch_size == dtl_child.get_error_signals().Width());
    const auto local_batch_size = local_activations.Width();

    // Compute Kronecker factors, assuming that local_errors are
    // already multiplied by 1/N in the loss layer.
    El::Matrix<DataType, El::Device::GPU> A, G;
    if(!metadata.is_conv) {
      get_kronecker_factor_fc(A, local_activations, 1.0/mini_batch_size);
      get_kronecker_factor_fc(G, local_errors, mini_batch_size);

    } else {
      const auto input_dims = l->get_input_dims(); // CHW
      const auto output_dims = l->get_output_dims(); // KH'W'
      const size_t num_input_channels = input_dims[0];
      const size_t num_output_channels = output_dims[0];
      size_t spatial_input_prod = 1, spatial_output_prod = 1;
      // std::accumulate might overflow for large 3D layers
      std::vector<int> input_spatial_dims, output_spatial_dims;
      for(auto i = input_dims.begin()+1; i != input_dims.end(); i++) {
        spatial_input_prod *= *i;
        input_spatial_dims.push_back(*i);
      }
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
      assert((size_t) local_activations.Height() == num_input_channels*spatial_input_prod);
      assert((size_t) local_errors.Height() == num_output_channels*spatial_output_prod);

      get_kronecker_factor_conv(
          A,
          local_activations, 1.0/mini_batch_size,
          local_batch_size, num_input_channels, input_spatial_dims,
          metadata.l_conv, true, stream);
      get_kronecker_factor_conv(
          G,
          local_errors, DataType(mini_batch_size)/spatial_output_prod,
          local_batch_size, num_output_channels, output_spatial_dims,
          metadata.l_conv, false, stream);
    }

    // Accumulate local Kronecker factors
    allreduce_lower_tri(A, comm, stream);
    allreduce_lower_tri(G, comm, stream);

    // Update average Kronecker factors
    // TODO: Each matrix is used only by one process, but A and G
    // should be consumed within this scope.
    if(m_kronecker_average.find(metadata.layer_id) == m_kronecker_average.end())
      m_kronecker_average.emplace(metadata.layer_id, std::make_pair(A, G));
    auto &Aave = m_kronecker_average[metadata.layer_id].first;
    auto &Gave = m_kronecker_average[metadata.layer_id].second;
    update_kronecker_average(
        Aave.Buffer(), A.Buffer(), A.Height()*A.Width(), m_kronecker_decay, stream);
    update_kronecker_average(
        Gave.Buffer(), G.Buffer(), G.Height()*G.Width(), m_kronecker_decay, stream);

    // Dump matrices for debugging
    if(comm->am_trainer_master() && m_print_matrix) {
      if(comm->am_trainer_master()) {
        std::cout << std::endl; El::Print(A, "A");
        std::cout << std::endl; El::Print(G, "G");
        std::cout << std::endl; El::Print(Aave, "Aave");
        std::cout << std::endl; El::Print(Gave, "Gave");
        std::cout << std::endl;
      }
    }

    // Dump L2 norm of matrices
    if(comm->am_trainer_master() && m_print_matrix_summary) {
      const auto &dtw = dynamic_cast<data_type_weights<DataType>*>(&weights);
      const auto &w_values = dtw->get_values();
      std::ostringstream oss;
      oss << "K-FAC callback: L2 norm @ "<< l->get_name() << ": "
          << get_matrix_stat(w_values.LockedMatrix(), "W")
          << ", " << get_matrix_stat(local_activations, "acts")
          << ", " << get_matrix_stat(local_errors, "errs")
          << ", " << get_matrix_stat(A, "A")
          << ", " << get_matrix_stat(G, "G")
          << ", " << get_matrix_stat(Aave, "Aave")
          << ", " << get_matrix_stat(Gave, "Gave")
          << std::endl;
      std::cout << oss.str();
    }

    // TODO: Use persistent workspace instead of using local
    // matrices to remove synchronization.
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }

  // Step 1 (BN)
  for(const auto metadata : m_learnable_layers) {
    if(!(metadata.is_bn_after_fc || metadata.is_bn_after_conv))
      continue;
    const auto &l = layers[metadata.layer_id];

    // Get activations, errors, and gradients
    const auto parent = l->get_parent_layers()[0];
    const auto child = l->get_child_layers()[0];
    const auto& dtl_parent = dynamic_cast<const data_type_layer<DataType>&>(*parent);
    const auto& dtl_child = dynamic_cast<const data_type_layer<DataType>&>(*child);
    const El::AbstractMatrix<DataType>& local_activations = dtl_parent.get_local_activations();
    const El::AbstractMatrix<DataType>& local_errors = dtl_child.get_local_error_signals();
    const auto mini_batch_size = dtl_parent.get_activations().Width();
    assert(mini_batch_size == dtl_child.get_error_signals().Width());
    const auto local_batch_size = local_activations.Width();

    assert(l->num_weights() == 4); // scale, bias, r_mean, r_var
    auto& scales = l->get_weights(0);
    auto& biases = l->get_weights(1);
    optimizer *s_optimizer = scales.get_optimizer();
    optimizer *b_optimizer = biases.get_optimizer();
    auto* s_dto = dynamic_cast<data_type_optimizer<DataType>*>(s_optimizer);
    auto* b_dto = dynamic_cast<data_type_optimizer<DataType>*>(b_optimizer);
    El::Matrix<DataType, El::Device::GPU> s_gradients = s_dto->get_gradient().Matrix();
    El::Matrix<DataType, El::Device::GPU> b_gradients = b_dto->get_gradient().Matrix();
    const auto &s_dtw = dynamic_cast<data_type_weights<DataType>*>(&scales);
    const auto &b_dtw = dynamic_cast<data_type_weights<DataType>*>(&biases);
    const auto &scale_values = s_dtw->get_values();
    const auto &bias_values = b_dtw->get_values();

    const bool is_update_required = (num_steps%m_update_interval_steps) == 0
        || m_kronecker_inverse.find(metadata.layer_id) == m_kronecker_inverse.end();
    if(is_update_required) {
      El::Matrix<DataType, El::Device::GPU> factor(metadata.bn_num_channels*2, local_batch_size);
      compute_bn_factor(
          local_activations.LockedBuffer(),
          local_errors.LockedBuffer(),
          scale_values.LockedMatrix().LockedBuffer(),
          bias_values.LockedMatrix().LockedBuffer(),
          factor.Buffer(),
          local_batch_size,
          metadata.bn_num_channels,
          metadata.bn_spatial_prod,
          stream);

      El::Matrix<DataType, El::Device::GPU> fisher_block(metadata.bn_num_channels*2, metadata.bn_num_channels*2);
      const DataType alpha = mini_batch_size;
      El::Gemm(
          El::NORMAL, El::TRANSPOSE,
          alpha, factor, factor,
          El::TypeTraits<DataType>::Zero(), fisher_block);

      allreduce_lower_tri(fisher_block, comm, stream);

      // Update average Kronecker factors
      // TODO: Each matrix is used only by one process, but fisher_block
      // should be consumed within this scope.
      if(m_kronecker_average.find(metadata.layer_id) == m_kronecker_average.end())
        m_kronecker_average.emplace(
            metadata.layer_id,
            std::make_pair(fisher_block, El::Matrix<DataType, El::Device::GPU>()));
      auto &Fave = m_kronecker_average[metadata.layer_id].first;
      update_kronecker_average(
          Fave.Buffer(), fisher_block.Buffer(),
          fisher_block.Height()*fisher_block.Width(),
          m_kronecker_decay, stream);

      // TODO: Use persistent workspace instead of using local
      // matrices to remove synchronization.
      CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // dump L2 norm of matrices
    if(comm->am_trainer_master() && m_print_matrix_summary) {
      std::ostringstream oss;
      oss << "K-FAC callback: L2 norm @ "<< l->get_name() << ": "
          << get_matrix_stat(scale_values.LockedMatrix(), "scale")
          << ", " << get_matrix_stat(bias_values.LockedMatrix(), "bias")
          << ", " << get_matrix_stat(local_activations, "acts")
          << ", " << get_matrix_stat(local_errors, "errs")
          << std::endl;
      std::cout << oss.str();
    }

  }

  // Step 2: Model-parallel inverse computation
  for(const auto metadata : m_learnable_layers) {
    if(!(metadata.is_fc || metadata.is_conv))
      continue;
    const auto &l = layers[metadata.layer_id];
    auto& weights = l->get_weights(0);
    optimizer *w_optimizer = weights.get_optimizer();
    auto* w_dto = dynamic_cast<data_type_optimizer<DataType>*>(w_optimizer);
    // w_gradients is already synchronized among processes.
    El::Matrix<DataType, El::Device::GPU>& w_gradients = w_dto->get_gradient().Matrix();

    const bool is_update_required = (num_steps%m_update_interval_steps) == 0
        || m_kronecker_inverse.find(metadata.layer_id) == m_kronecker_inverse.end();
    if(is_update_required && comm->get_rank_in_trainer() == metadata.proc_rank) {
      const auto &Aave = m_kronecker_average[metadata.layer_id].first;
      const auto &Gave = m_kronecker_average[metadata.layer_id].second;
      // Compute the pi constant
      const DataType pi = m_use_pi ? compute_pi(Aave, Gave) : 1.0;
      // Compute the inverse of the factors
      const bool print_time = m_print_time;
      // Since setting different damping constants for A and G is an
      // alternative heuristics to pi, they should be the same if pi is used.
      if(m_use_pi && m_damping_act != m_damping_err) {
        std::stringstream err;
        err << "Damping values for activations and errors are different while the pi constant is used."
            << " layer: " << l->get_name()
            << ", m_damping_act: " << m_damping_act
            << ", m_damping_err: " << m_damping_err;
        LBANN_WARNING(err.str());
      }

      if(m_kronecker_inverse.find(metadata.layer_id) == m_kronecker_inverse.end())
        m_kronecker_inverse.emplace(metadata.layer_id, std::make_pair(
            El::Matrix<DataType, El::Device::GPU>(),
            El::Matrix<DataType, El::Device::GPU>()));
      auto &Ainv = m_kronecker_inverse[metadata.layer_id].first;
      auto &Ginv = m_kronecker_inverse[metadata.layer_id].second;
      get_matrix_inverse(Ainv, Aave, print_time,
                         DataType(m_damping_act*pi), 0,
                         false, stream);
      get_matrix_inverse(Ginv, Gave, print_time,
                         DataType(m_damping_err/pi), 0,
                         false, stream);

      if(m_print_matrix_summary) {
        std::ostringstream oss;
        oss << "K-FAC callback: pi=" << pi << " @ "<< l->get_name() << std::endl;
        std::cout << oss.str();
      }
    }

    DataType dst_scale = El::TypeTraits<DataType>::Zero(),
        gradient_scale = El::TypeTraits<DataType>::One();
    // grad_buffer is already synchronized among processes,
    // and won't be all-reduced later.
    auto& grad_buffer = w_optimizer->get_gradient_buffer(
        dst_scale, gradient_scale, false);

    if(comm->get_rank_in_trainer() == metadata.proc_rank) {
      const auto &Ainv = m_kronecker_inverse[metadata.layer_id].first;
      const auto &Ginv = m_kronecker_inverse[metadata.layer_id].second;
      if(metadata.is_conv) {
        const auto num_output_channels = l->get_output_dims()[0];
        assert(w_gradients.Width() == 1);
        assert((w_gradients.Height()%num_output_channels) == 0);
        const auto height_reshaped = w_gradients.Height()/num_output_channels;
        w_gradients.Attach(height_reshaped,
                           num_output_channels,
                           w_gradients.Buffer(),
                           height_reshaped);
      }

      // Compute preconditioned gradients
      El::Matrix<DataType, El::Device::GPU> Gg(
          Ginv.Height(),
          metadata.is_conv ? w_gradients.Height() : w_gradients.Width());
      El::Gemm(
          El::NORMAL, metadata.is_conv ? El::TRANSPOSE : El::NORMAL,
          El::TypeTraits<DataType>::One(), Ginv, w_gradients,
          El::TypeTraits<DataType>::Zero(), Gg);
      El::Matrix<DataType, El::Device::GPU> Fgrad(Ginv.Height(), Ainv.Width());
      El::Gemm(
          El::NORMAL, El::NORMAL,
          El::TypeTraits<DataType>::One(), Gg, Ainv,
          El::TypeTraits<DataType>::Zero(), Fgrad);

      if(metadata.is_conv) {
        Fgrad.Attach(Fgrad.Width()*Fgrad.Height(), 1,
                     Fgrad.Buffer(),
                     Fgrad.Width()*Fgrad.Height());
      } else {
        assert(Fgrad.Height() == w_gradients.Height());
        assert(Fgrad.Width() == w_gradients.Width());
      }

      // Apply preconditioned grads
      El::Copy(Fgrad, grad_buffer.Matrix());

      // Dump matrices for debugging
      if(m_print_matrix) {
        std::cout << std::endl; El::Print(Ainv, "Ainv");
        std::cout << std::endl; El::Print(Ginv, "Ginv");
        std::cout << std::endl; El::Print(w_gradients, "w_grad");
        std::cout << std::endl; El::Print(Fgrad, "Fgrad");
        std::cout << std::endl;
      }

      // Dump L2 norm of matrices
      if(m_print_matrix_summary) {
        const auto &dtw = dynamic_cast<data_type_weights<DataType>*>(&weights);
        const auto &w_values = dtw->get_values();
        std::ostringstream oss;
        oss << "K-FAC callback: L2 norm @ "<< l->get_name() << ": "
            << get_matrix_stat(w_values.LockedMatrix(), "W")
            << ", " << get_matrix_stat(Ainv, "Ainv")
            << ", " << get_matrix_stat(Ginv, "Ginv")
            << ", " << get_matrix_stat(w_gradients, "grad")
            << ", " << get_matrix_stat(Fgrad, "Finvgrad")
            << std::endl;
        std::cout << oss.str();
      }

      // TODO: Use persistent workspace instead of using local
      // matrices to remove synchronization.
      CHECK_CUDA(cudaStreamSynchronize(stream));

    }
  }

  // Step 2 (BN)
  for(const auto metadata : m_learnable_layers) {
    if(!(metadata.is_bn_after_fc || metadata.is_bn_after_conv))
      continue;
    const auto &l = layers[metadata.layer_id];

    const bool is_update_required = (num_steps%m_update_interval_steps) == 0
        || m_kronecker_inverse.find(metadata.layer_id) == m_kronecker_inverse.end();

    if(!(is_update_required && comm->get_rank_in_trainer() == metadata.proc_rank))
      continue;

    if(m_kronecker_inverse.find(metadata.layer_id) == m_kronecker_inverse.end())
      m_kronecker_inverse.emplace(metadata.layer_id, std::make_pair(
          El::Matrix<DataType, El::Device::GPU>(), // Finv
          El::Matrix<DataType, El::Device::GPU>())); // dummy

    const auto &Fave = m_kronecker_average[metadata.layer_id].first;
    auto& Finv = m_kronecker_inverse[metadata.layer_id].first;
    const bool print_time = comm->am_trainer_master() && m_print_time;
    get_matrix_inverse(
        Finv, Fave, print_time,
        DataType(m_damping_bn_act), DataType(m_damping_bn_err),
        true, stream);

    // dump L2 norm of matrices
    if(comm->am_trainer_master() && m_print_matrix_summary) {
      std::ostringstream oss;
      oss << "K-FAC callback: L2 norm @ "<< l->get_name() << ": "
          << get_matrix_stat(Fave, "Fave")
          << std::endl;
      std::cout << oss.str();
    }

    auto& scales = l->get_weights(0);
    auto& biases = l->get_weights(1);
    optimizer *s_optimizer = scales.get_optimizer();
    optimizer *b_optimizer = biases.get_optimizer();
    auto* s_dto = dynamic_cast<data_type_optimizer<DataType>*>(s_optimizer);
    auto* b_dto = dynamic_cast<data_type_optimizer<DataType>*>(b_optimizer);
    El::Matrix<DataType, El::Device::GPU> s_gradients = s_dto->get_gradient().Matrix();
    El::Matrix<DataType, El::Device::GPU> b_gradients = b_dto->get_gradient().Matrix();

    El::Matrix<DataType, El::Device::GPU> stacked_grads(metadata.bn_num_channels*2, 1);
    CHECK_CUDA(cudaMemcpyAsync(
        stacked_grads.Buffer(), s_gradients.LockedBuffer(),
        metadata.bn_num_channels*sizeof(DataType), cudaMemcpyDeviceToDevice,
        hydrogen::cuda::GetDefaultStream()));
    CHECK_CUDA(cudaMemcpyAsync(
        stacked_grads.Buffer()+metadata.bn_num_channels, b_gradients.LockedBuffer(),
        metadata.bn_num_channels*sizeof(DataType), cudaMemcpyDeviceToDevice,
        hydrogen::cuda::GetDefaultStream()));

    El::Matrix<DataType, El::Device::GPU> Fgrad(metadata.bn_num_channels*2, 1);
    El::Gemm(
        El::NORMAL, El::NORMAL,
        El::TypeTraits<DataType>::One(), Finv, stacked_grads,
        El::TypeTraits<DataType>::Zero(), Fgrad);

    DataType dst_scale = El::TypeTraits<DataType>::Zero(),
        gradient_scale = El::TypeTraits<DataType>::One();
    auto& s_grad_buffer = s_optimizer->get_gradient_buffer(
        dst_scale, gradient_scale, false);
    auto& b_grad_buffer = b_optimizer->get_gradient_buffer(
        dst_scale, gradient_scale, false);
    // TODO: Better way to copy?
    CHECK_CUDA(cudaMemcpy(
        s_grad_buffer.Matrix().Buffer(), Fgrad.LockedBuffer(),
        metadata.bn_num_channels*sizeof(DataType), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(
        b_grad_buffer.Matrix().Buffer(), Fgrad.LockedBuffer()+metadata.bn_num_channels,
        metadata.bn_num_channels*sizeof(DataType), cudaMemcpyDeviceToDevice));

    // dump L2 norm of matrices
    if(comm->am_trainer_master() && m_print_matrix_summary) {
      std::ostringstream oss;
      oss << "K-FAC callback: L2 norm @ "<< l->get_name() << ": "
          << ", " << get_matrix_stat(Finv, "Finv")
          << ", " << get_matrix_stat(Fgrad, "Fgrad")
          << ", " << get_matrix_stat(s_gradients, "scale_grad")
          << ", " << get_matrix_stat(b_gradients, "bias_grad")
          << std::endl;
      std::cout << oss.str();
    }

    // TODO: Use persistent workspace instead of using local
    // matrices to remove synchronization.
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }

  // Step 3: All-gather of each preconditioned gradient tensor
  for(const auto metadata : m_learnable_layers) {
    if(!(metadata.is_fc || metadata.is_conv))
      continue;
    const auto &l = layers[metadata.layer_id];
    auto& weights = l->get_weights(0);
    optimizer *w_optimizer = weights.get_optimizer();
    DataType dst_scale = El::TypeTraits<DataType>::Zero(),
        gradient_scale = El::TypeTraits<DataType>::One();
    // grad_buffer is already synchronized among processes,
    // and won't be all-reduced later.
    auto& grad_buffer = w_optimizer->get_gradient_buffer(
        dst_scale, gradient_scale, false);
    El::Broadcast(
        grad_buffer.Matrix(), comm->get_trainer_comm(),
        metadata.proc_rank);
  }

  // Step 3 (BN)
  for(const auto metadata : m_learnable_layers) {
    if(!(metadata.is_bn_after_fc || metadata.is_bn_after_conv))
      continue;
    const auto &l = layers[metadata.layer_id];
    auto& scales = l->get_weights(0);
    auto& biases = l->get_weights(1);
    optimizer *s_optimizer = scales.get_optimizer();
    optimizer *b_optimizer = biases.get_optimizer();
    DataType dst_scale = El::TypeTraits<DataType>::Zero(),
        gradient_scale = El::TypeTraits<DataType>::One();
    auto& s_grad_buffer = s_optimizer->get_gradient_buffer(
        dst_scale, gradient_scale, false);
    auto& b_grad_buffer = b_optimizer->get_gradient_buffer(
        dst_scale, gradient_scale, false);
    El::Broadcast(
        s_grad_buffer.Matrix(), comm->get_trainer_comm(),
        metadata.proc_rank);
    El::Broadcast(
        b_grad_buffer.Matrix(), comm->get_trainer_comm(),
        metadata.proc_rank);
  }

}

void kfac::get_kronecker_factor_fc(
    El::AbstractMatrix<DataType>& factor,
    const El::AbstractMatrix<DataType>& A,
    const DataType alpha) {
  assert(A.GetDevice() == El::Device::GPU);
  factor.Resize(A.Height(), A.Height());
  El::Gemm(
      El::NORMAL, El::TRANSPOSE,
      alpha, A, A,
      El::TypeTraits<DataType>::Zero(), factor);
}

void kfac::get_kronecker_factor_conv(
    El::Matrix<DataType, El::Device::GPU>& factor,
    const El::Matrix<DataType, El::Device::GPU>& A,
    const DataType alpha,
    const size_t local_batch_size, const size_t num_channels,
    const std::vector<int> spatial_dims,
    const convolution_layer<DataType, data_layout::DATA_PARALLEL, El::Device::GPU> *l_conv,
    const bool use_im2col,
    const cudaStream_t& stream) {
  assert(factor.GetDevice() == El::Device::GPU);
  assert(A.GetDevice() == El::Device::GPU);

  const auto dilations = l_conv->get_dilations();
  for(auto i = dilations.begin(); i != dilations.end(); i++)
    if(*i != 1) {
      std::stringstream err;
      err << "The K-FAC callback onky supports dilation width of 1."
          << " layer: " << l_conv->get_name();
      LBANN_ERROR(err.str());
    }

  // The matrix size will be overwritten later.
  El::Matrix<DataType, El::Device::GPU> Acol;
  if(use_im2col) {
    im2col(A, Acol,
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
    Acol.Resize(num_channels, local_batch_size*spatial_prod);
    conv_transpose(
        A.LockedBuffer(), Acol.Buffer(),
        local_batch_size, num_channels, spatial_prod,
        stream);
  }

  factor.Resize(Acol.Height(), Acol.Height());
  El::Gemm(
      El::NORMAL, El::TRANSPOSE,
      alpha, Acol, Acol,
      El::TypeTraits<DataType>::Zero(), factor);

  CHECK_CUDA(cudaStreamSynchronize(stream));
}

void kfac::get_matrix_inverse(
    El::Matrix<DataType, El::Device::GPU>& Ainv,
    const El::Matrix<DataType, El::Device::GPU>& A,
    const bool report_time,
    const DataType damping,
    const DataType damping_bn_err,
    const bool is_bn,
    const cudaStream_t& stream) {
  assert(A.Width() == A.Height());
  Ainv.Resize(A.Height(), A.Width());
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

  El::Matrix<DataType, El::Device::GPU> Linv(Ainv.Height(), Ainv.Width());
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

  CHECK_CUDA(cudaStreamSynchronize(stream));
}

double kfac::compute_pi(const El::Matrix<DataType, El::Device::GPU>& A,
                        const El::Matrix<DataType, El::Device::GPU>& G) {
  // TODO: El::Trace is defined but not implemented yet.
  const auto get_trace =
      [](const El::Matrix<DataType, El::Device::GPU> X) {
        const El::Matrix<DataType> XCPU(X);
        DataType s = 0.0;
        for(int i = 0; i < XCPU.Height(); i++)
          s += XCPU(i, i);
        return (double) s;
      };
  return sqrt((get_trace(A)/A.Height())/(get_trace(G)/G.Height()));
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
                               lbann_comm *comm,
                               const cudaStream_t& stream) {
  assert(A.Height() == A.Width());
  El::Matrix<DataType, El::Device::GPU> AL(A.Height()*(A.Height()+1)/2, 1);
  pack_lower_tri(AL.Buffer(), A.LockedBuffer(), A.Height(), stream);
  comm->allreduce((El::AbstractMatrix<DataType>&) AL,
                  comm->get_trainer_comm());
  unpack_lower_tri(A.Buffer(), AL.Buffer(), A.Height(), stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));
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

} // namespace callback
} // namespace lbann
