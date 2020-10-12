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
#include "lbann/callbacks/kfac/kfac_block_fc_conv.hpp"
#include "lbann/callbacks/kfac/kfac_block_bn.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/learning/fully_connected.hpp"
#include "lbann/layers/learning/convolution.hpp"
#include "lbann/layers/regularizers/batch_normalization.hpp"

namespace lbann {
namespace callback {

#ifdef LBANN_HAS_GPU

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

      kfac_block* block;
      if(is_fc) {
        block = new kfac_block_fc_conv(
            l, this, layer_id, proc_rank,
            false, 0, 0, {}, {});

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

        block = new kfac_block_fc_conv(
            l, this, layer_id, proc_rank,
            true,
            spatial_input_prod, spatial_output_prod,
            input_spatial_dims, output_spatial_dims);

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

        block = new kfac_block_bn(
            l, this, layer_id, proc_rank,
            is_bn_after_conv,
            num_channels, spatial_prod);
      }

      m_blocks.push_back(std::shared_ptr<kfac_block>(block));
      if(m_inverse_strategy != ROOT)
        proc_rank = (proc_rank+1)%num_procs;
    }

    if(comm->am_trainer_master())
      for(const auto& block : m_blocks)
        std::cout << "K-FAC callback setup: "
                  << block->get_info() << std::endl;
  }

  // Step 1: Ensure that each process has averaged Kronecker factors
  // for the model-parallel part.
  for(auto& block : m_blocks) {
    const bool is_update_required =
        ((num_steps%m_update_interval_steps) == 0
         || !block->has_kronecker_inverse());
    if(!is_update_required)
      continue;
    block->update_kronecker_factors(
        comm,
        m_kronecker_decay,
        m_print_matrix, m_print_matrix_summary);
  }

  // Step 2: Model-parallel inverse computation
  for(auto& block : m_blocks) {
    const bool is_update_required =
        ((num_steps%m_update_interval_steps) == 0
         || !block->has_kronecker_inverse())
        && (size_t) comm->get_rank_in_trainer() == block->get_inverse_proc_rank();
    if(!is_update_required)
      continue;

    // TODO: Add kfac_block::is_bn?
    const bool is_bn = dynamic_cast<kfac_block_bn*>(block.get()) != nullptr;
    block->update_kronecker_inverse(
        comm, m_use_pi,
        is_bn ? m_damping_bn_act : m_damping_act,
        is_bn ? m_damping_bn_err : m_damping_err,
        m_print_matrix, m_print_matrix_summary,
        m_print_time);
  }

  // Step 3: All-gather of each preconditioned gradient tensor
  for(auto& block : m_blocks) {
    block->update_preconditioned_grads(comm);
  }

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

#endif // LBANN_HAS_GPU

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
