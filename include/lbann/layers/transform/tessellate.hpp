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

#ifndef LBANN_LAYERS_TRANSFORM_TESSELLATE_HPP_INCLUDED
#define LBANN_LAYERS_TRANSFORM_TESSELLATE_HPP_INCLUDED

#include "lbann/layers/layer.hpp"

namespace lbann {

/** @brief Repeat a tensor until it matches specified dimensions.
 *
 *  The output dimensions do not need to be integer multiples of the
 *  input dimensions.
 *
 *  As an example, tessellating a @f$ 2 \times 2 @f$ matrix into a
 *  @f$ 3 \times 4 @f$ matrix looks like the following:
 *  @f[
 *    \begin{bmatrix}
 *      1 & 2 \\
 *      3 & 4
 *    \end{bmatrix}
 *    \rightarrow
 *    \begin{bmatrix}
 *      1 & 2 & 1 & 2 \\
 *      3 & 4 & 3 & 4 \\
 *      1 & 2 & 1 & 2
 *    \end{bmatrix}
 *  @f]
 *
 *  Formally, suppose we tessellate an input tensor @f$ X @f$ with
 *  dimensions @f$d_1 \times\cdots\times d_n@f$ to obtain an output
 *  tensor @f$ Y @f$ with dimensions @f$e_1 \times\cdots\times
 *  e_n@f$. Then, denoting the modulo operator with @f$ \% @f$,
 *  @f[ Y_{i_1,\cdots,i_n} = X_{i_1\% d_1,\cdots,i_n\% d_n} @f]
 */
template <data_layout Layout = data_layout::DATA_PARALLEL, El::Device Device = El::Device::CPU>
class tessellate_layer : public Layer {
public:

  tessellate_layer(lbann_comm *comm, std::vector<int> dims = {})
    : Layer(comm) {
    set_output_dims(dims);
  }

  tessellate_layer(const tessellate_layer& other)
    : Layer(other),
      m_input_v(other.m_input_v ? other.m_input_v->Copy() : nullptr) {}
  tessellate_layer& operator=(const tessellate_layer& other) {
    Layer::operator=(other);
    m_input_v.reset(other.m_input_v ? other.m_input_v->Copy() : nullptr);
    return *this;
  }

  tessellate_layer* copy() const override { return new tessellate_layer(*this); }
  std::string get_type() const override { return "tessellate"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  void setup_dims() override {
    Layer::setup_dims();
    std::stringstream err;

    // Check input and output dimensions
    const auto input_dims = get_input_dims();
    const auto& output_dims = get_output_dims();
    if (input_dims.size() != output_dims.size()) {
      err << get_type() << " layer \"" << get_name() << "\" "
          << "attempted to tessellate a ";
      for (size_t i = 0; i < input_dims.size(); ++i) {
        err << (i > 0 ? "x" : "") << input_dims[i];
      }
      err << " tensor into a ";
      for (size_t i = 0; i < output_dims.size(); ++i) {
        err << (i > 0 ? "x" : "") << output_dims[i];
      }
      err << " tensor";
      LBANN_ERROR(err.str());
    }

    /// @todo Support tessellation with >3 dimensions
    if (input_dims.size() > 3) {
      err << get_type() << " layer \"" << get_name() << "\" "
          << "attempted to tessellate a ";
      for (size_t i = 0; i < input_dims.size(); ++i) {
        err << (i > 0 ? "x" : "") << input_dims[i];
      }
      err << " tensor, but tessellation is currently only supported "
          << "with 3 dimensions or less";
    }

  }

  void setup_matrices(const El::Grid& grid) override {
    Layer::setup_matrices(grid);
    auto dist_data = get_prev_activations().DistData();
    dist_data.colDist = El::STAR;
    m_input_v.reset(AbsDistMat::Instantiate(dist_data));
  }

protected:

  void fp_compute() override {

    // Get input and output dimensions
    auto input_dims = get_input_dims();
    auto output_dims = get_output_dims();
    while (input_dims.size() < 3) { input_dims.insert(input_dims.begin(), 1); }
    while (output_dims.size() < 3) { output_dims.insert(output_dims.begin(), 1); }

    // Get input and output data
    auto& output = get_activations();
    const auto& input = get_prev_activations();
    m_input_v->Empty(false);
    m_input_v->AlignWith(output);
    if (m_input_v->DistData() == input.DistData()) {
      El::LockedView(*m_input_v, input);
    } else {
      El::Copy(input, *m_input_v);
    }
    const auto& local_input = m_input_v->LockedMatrix();

    // Apply tessellation
    /// @todo Support >3 dimensions
    if (input_dims.size() > 3) {
      LBANN_ERROR("tessellate layer currently only supports 3D tensors");
    }
    fp_compute_3d(input_dims, output_dims, local_input, output);

  }

  void bp_compute() override {

    // Get input and output dimensions
    auto input_dims = get_input_dims();
    auto output_dims = get_output_dims();
    while (input_dims.size() < 3) { input_dims.insert(input_dims.begin(), 1); }
    while (output_dims.size() < 3) { output_dims.insert(output_dims.begin(), 1); }

    // Get input and output data
    const auto& gradient_wrt_output = get_prev_error_signals();
    auto& gradient_wrt_input = get_error_signals();
    m_input_v->Empty(false);
    m_input_v->AlignWith(gradient_wrt_output);
    if (m_input_v->DistData() == gradient_wrt_input.DistData()) {
      El::View(*m_input_v, gradient_wrt_input);
    } else {
      m_input_v->Resize(gradient_wrt_input.Height(),
                        gradient_wrt_input.Width());
    }
    auto& local_gradient_wrt_input = m_input_v->Matrix();

    // Apply back prop with local data
    /// @todo Support >3 dimensions
    bp_compute_3d(input_dims, output_dims,
                  gradient_wrt_output, local_gradient_wrt_input);

    // Accumulate local error signals, if needed
    if (m_input_v->DistData() != gradient_wrt_input.DistData()) {
      m_comm->allreduce(*m_input_v, m_input_v->RedundantComm());
      El::Copy(*m_input_v, gradient_wrt_input);
    }

  }

private:

  /** View into input tensor. */
  std::unique_ptr<AbsDistMat> m_input_v;

  /** Apply tessellation.
   *  Columns of 'input' should be intact mini-batch samples. If the
   *  data layout is not purely data-parallel, this means input data
   *  is duplicated over the input matrix's column communicator.
   */
  static void fp_compute_3d(const std::vector<int>& input_dims,
                            const std::vector<int>& output_dims,
                            const AbsMat& input,
                            AbsDistMat& output);
  /** Compute local contribution to tessellation back prop
   *  The global gradient w.r.t. input can be obtained by performing
   *  an allreduce over the input matrix's column communicator.
   */
  static void bp_compute_3d(const std::vector<int>& input_dims,
                            const std::vector<int>& output_dims,
                            const AbsDistMat& gradient_wrt_output,
                            AbsMat& gradient_wrt_input);

};

} // namespace lbann

#endif // LBANN_LAYERS_TRANSFORM_TESSELLATE_HPP_INCLUDED
