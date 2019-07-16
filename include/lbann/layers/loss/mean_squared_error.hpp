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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYERS_LOSS_MEAN_SQUARED_ERROR_HPP_INCLUDED
#define LBANN_LAYERS_LOSS_MEAN_SQUARED_ERROR_HPP_INCLUDED

#include "lbann/layers/layer.hpp"

//#define DISPLAY_INDIVIDUAL_MSE
#ifdef DISPLAY_INDIVIDUAL_MSE
#include "lbann/utils/cublas.hpp"
#include "lbann/models/model.hpp"
#endif

namespace lbann {

/** @brief
 *
 *  Given a prediction @f$y@f$ and ground truth @f$\hat{y}@f$,
 *  @f[
 *    MSE(y,\hat{y})
 *      = \frac{1}{n} \sum\limits_{i=1}^{n} (y_i - \hat{y}_i)^2
 *  @f]
 */
template <data_layout T_layout, El::Device Dev>
class mean_squared_error_layer : public Layer {
public:

  mean_squared_error_layer(lbann_comm *comm) : Layer(comm) {
    this->m_expected_num_parent_layers = 2;
  }

  mean_squared_error_layer(const mean_squared_error_layer& other)
    : Layer(other) {
    m_workspace.reset(other.m_workspace ?
                      other.m_workspace->Copy() :
                      nullptr);
  }

  mean_squared_error_layer& operator=(const mean_squared_error_layer& other) {
    Layer::operator=(other);
    m_workspace.reset(other.m_workspace ?
                      other.m_workspace->Copy() :
                      nullptr);
    return *this;
  }

  mean_squared_error_layer* copy() const override { return new mean_squared_error_layer(*this); }
  std::string get_type() const override { return "mean squared error"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  void setup_dims() override {
    Layer::setup_dims();
    set_output_dims({1});

    // Check that input dimensions match
    if (get_input_dims(0) != get_input_dims(1)) {
      const auto& parents = get_parent_layers();
      std::stringstream err;
      err << get_type() << " layer \"" << get_name() << "\" "
          << "has input tensors with different dimensions (";
      for (int i = 0; i < get_num_parents(); ++i) {
        const auto& dims = get_input_dims(i);
        err << (i > 0 ? ", " : "")
            << "layer \"" << parents[i]->get_name() << "\" outputs ";
        for (size_t j = 0; j < dims.size(); ++j) {
          err << (j > 0 ? " x " : "") << dims[j];
        }
      }
      err << ")";
      LBANN_ERROR(err.str());
    }

  }

  void setup_data() override {
    Layer::setup_data();

    // Initialize workspace
    const auto& input_dist = get_prev_activations(0).DistData();
    m_workspace.reset(AbsDistMat::Instantiate(*input_dist.grid,
                                              input_dist.root,
                                              El::STAR,
                                              input_dist.rowDist,
                                              (input_dist.blockHeight == 1
                                               && input_dist.blockWidth == 1 ?
                                               El::ELEMENT : El::BLOCK),
                                              input_dist.device));
#ifdef DISPLAY_INDIVIDUAL_MSE
    m_workspace_pc.reset(AbsDistMat::Instantiate(*input_dist.grid,
                                                 input_dist.root,
                                                 El::STAR,
                                                 input_dist.rowDist,
                                                 (input_dist.blockHeight == 1
                                                  && input_dist.blockWidth == 1 ?
                                                  El::ELEMENT : El::BLOCK),
                                                 input_dist.device));
#endif
#ifdef HYDROGEN_HAVE_CUB
    if (m_workspace->GetLocalDevice() == El::Device::GPU) {
      m_workspace->Matrix().SetMemoryMode(1); // CUB memory pool
#ifdef DISPLAY_INDIVIDUAL_MSE
      m_workspace_pc->Matrix().SetMemoryMode(1); // CUB memory pool
#endif
    }
#endif // HYDROGEN_HAVE_CUB

  }

  void fp_compute() override {

    // Initialize workspace
    m_workspace->Empty();
    m_workspace->AlignWith(get_prev_activations());
    m_workspace->Resize(1, get_prev_activations().Width());
#ifdef DISPLAY_INDIVIDUAL_MSE
    m_workspace_pc->Empty();
    m_workspace_pc->AlignWith(get_prev_activations());
    m_workspace_pc->Resize(get_prev_activations().Height(),
                           get_prev_activations().Width());
#endif

    // Compute local contributions and accumulate
    /// @todo Consider reduce rather than allreduce
#ifdef DISPLAY_INDIVIDUAL_MSE
    local_fp_compute(get_input_size(),
                     get_local_prev_activations(0),
                     get_local_prev_activations(1),
                     m_workspace->Matrix(),
                     m_workspace_pc->Matrix());
#else
    local_fp_compute(get_input_size(),
                     get_local_prev_activations(0),
                     get_local_prev_activations(1),
                     m_workspace->Matrix());
#endif
    m_comm->allreduce(*m_workspace, m_workspace->RedundantComm());
    El::Copy(*m_workspace, get_activations());

#ifdef DISPLAY_INDIVIDUAL_MSE
    print_mse(*m_workspace_pc);
#endif

   // Clean up
    m_workspace->Empty();
#ifdef DISPLAY_INDIVIDUAL_MSE
    m_workspace_pc->Empty();
#endif

  }

#ifdef DISPLAY_INDIVIDUAL_MSE
  void print_mse(AbsDistMat &mse) {
    //constexpr DataType zero = 0;
    constexpr DataType one = 1;
    const auto& local_input = mse.LockedMatrix();
    const auto& local_height = local_input.Height();
    const auto& local_width = local_input.Width();
    const auto& mini_batch_size = mse.Width();

    auto model = this->get_model();
    int cur_step = model->get_step();
    int num_steps_per_epoch = model->get_num_iterations_per_epoch(
        model->get_execution_mode());

    if ((cur_step % num_steps_per_epoch) == 0) {
      m_per_parameter_mse = std::vector<DataType>(local_height, 0);
      m_num_samples = 0;
    }

    m_num_samples += mini_batch_size;

    GPUMat sum_d, ones_d;
    sum_d.SetMemoryMode(1);
    ones_d.SetMemoryMode(1);
    sum_d.Resize(local_height, 1);

    auto&& handle = El::GPUManager::cuBLASHandle();
    auto&& stream = El::GPUManager::Stream();
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    if (local_input.IsEmpty()) {
      El::Zero(sum_d);
    } else {
      ones_d.Resize(local_width, 1);
      El::Fill(ones_d, one);
      //El::Gemv(El::NORMAL, one, local_input, ones_d, zero, sum_d);
      //El::Zero(sum_d);
      if (local_width != 1) {
        LBANN_ERROR("local_width != 1");
      }
      El::Copy(local_input, sum_d);
    }

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    //El::Scale(one/mini_batch_size, sum_d);
    get_comm()->allreduce(static_cast<AbsMat&>(sum_d),
                          mse.DistComm());
    if (get_comm()->am_world_master()) {
      std::vector<DataType> mse_h(local_height);
      CHECK_CUDA(cudaMemcpyAsync(mse_h.data(),
                                 sum_d.LockedBuffer(),
                                 //local_input.LockedBuffer(),
                                 sizeof(DataType) * local_height,
                                 cudaMemcpyDeviceToHost, stream));
      CHECK_CUDA(cudaStreamSynchronize(stream));
      for (int i = 0; i < local_height; ++i) {
        m_per_parameter_mse[i] += mse_h[i];
      }
      // last iteration
      if (((cur_step + 1) % num_steps_per_epoch) == 0) {
        DataType mse_sum = 0;
        for (auto &x: m_per_parameter_mse) {
          x /= m_num_samples;
          mse_sum += x;
        }
        mse_sum /= m_per_parameter_mse.size();
        std::stringstream ss;
        ss << "Per-parameter MSE:";
        for (auto i: m_per_parameter_mse) {
          ss << " " << i;
        }
        std::cout << ss.str() << "\n" << "Mean MSE: " << mse_sum << std::endl;
      }
    }
  }
#endif // DISPLAY_INDIVIDUAL_MSE
  
  void bp_compute() override {

    // Initialize workspace
    m_workspace->Empty();
    m_workspace->AlignWith(get_prev_activations());
    El::Copy(get_prev_error_signals(), *m_workspace);

    // Compute local gradients
    local_bp_compute(get_input_size(),
                     get_local_prev_activations(0),
                     get_local_prev_activations(1),
                     m_workspace->LockedMatrix(),
                     get_local_error_signals(0),
                     get_local_error_signals(1));

    // Clean up
    m_workspace->Empty();

  }

private:

  /** Compute local contributions to mean squared error loss. */
#ifdef DISPLAY_INDIVIDUAL_MSE
  static void local_fp_compute(El::Int height,
                               const AbsMat& local_prediction,
                               const AbsMat& local_ground_truth,
                               AbsMat& local_contribution,
                               AbsMat& local_contribution_pc);
#else
  static void local_fp_compute(El::Int height,
                               const AbsMat& local_prediction,
                               const AbsMat& local_ground_truth,
                               AbsMat& local_contribution);
#endif

  /** Compute local gradients. */
  static void local_bp_compute(El::Int height,
                               const AbsMat& local_prediction,
                               const AbsMat& local_ground_truth,
                               const AbsMat& local_gradient_wrt_output,
                               AbsMat& local_gradient_wrt_prediction,
                               AbsMat& local_gradient_wrt_ground_truth);

  /** Workspace matrix. */
  std::unique_ptr<AbsDistMat> m_workspace;
#ifdef DISPLAY_INDIVIDUAL_MSE  
  std::unique_ptr<AbsDistMat> m_workspace_pc;
  std::vector<DataType> m_per_parameter_mse;
  int m_num_samples;
#endif
};

} // namespace lbann

#endif // LBANN_LAYERS_LOSS_MEAN_SQUARED_ERROR_HPP_INCLUDED
