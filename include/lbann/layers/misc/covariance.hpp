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

#ifndef LBANN_LAYERS_MISC_COVARIANCE_HPP_INCLUDED
#define LBANN_LAYERS_MISC_COVARIANCE_HPP_INCLUDED

#include "lbann/layers/layer.hpp"

namespace lbann {

/** @brief Estimate covariance.
 *
 *  Given inputs @f$x@f$ and @f$y@f$ with empirical means
 *  @f$\bar{x}@f$ and @f$\bar{y}@f$, an unbiased estimator for the
 *  covariance is given by
 *  @f[
 *    \sigma_{xy}^2
 *      \approx \frac{1}{n-1} \sum\limits_{i=1}^{n} (x - \bar{x}) (y - \bar{y})
 *  @f]
 *  Scaling by @f$ 1/n @f$ instead of @f$ 1/(n-1) @f$ is a biased
 *  estimator.
 */
template <data_layout Layout, El::Device Device>
class covariance_layer : public Layer {
public:

  covariance_layer(lbann_comm *comm, bool biased)
    : Layer(comm), m_biased(biased) {
    this->m_expected_num_parent_layers = 2;
  }
  covariance_layer(const covariance_layer& other)
    : Layer(other),
      m_biased(other.m_biased),
      m_means(other.m_means ? other.m_means->Copy() : nullptr),
      m_workspace(other.m_workspace ?
                  other.m_workspace->Copy() : nullptr) {}
  covariance_layer& operator=(const covariance_layer& other) {
    Layer::operator=(other);
    m_biased = other.m_biased;
    m_means.reset(other.m_means ? other.m_means->Copy() : nullptr);
    m_workspace.reset(other.m_workspace ?
                      other.m_workspace->Copy() : nullptr);
    return *this;
  }

  covariance_layer* copy() const override { return new covariance_layer(*this); }
  std::string get_type() const override { return "covariance"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  description get_description() const override {
    auto desc = Layer::get_description();
    desc.add("Biased", m_biased);
    return desc;
  }

protected:

  void setup_matrices(const El::Grid& grid) override {
    Layer::setup_matrices(grid);
    auto dist_data = get_prev_activations().DistData();
    dist_data.colDist = El::STAR;
    m_means.reset(AbsDistMat::Instantiate(dist_data));
    m_workspace.reset(AbsDistMat::Instantiate(dist_data));
  }

  void setup_dims() override {
    Layer::setup_dims();
    set_output_dims({1});
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

  void fp_compute() override;
  void bp_compute() override;

private:

  /** Whether to use biased covariance estimator. */
  bool m_biased;

  /** Means for each mini-batch sample.  */
  std::unique_ptr<AbsDistMat> m_means;
  /** Workspace. */
  std::unique_ptr<AbsDistMat> m_workspace;

};

} // namespace lbann

#endif // LBANN_LAYERS_MISC_COVARIANCE_HPP_INCLUDED
