////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
// weights .hpp .cpp - Layer weights class
////////////////////////////////////////////////////////////////////////////////

#include "lbann/weights/weights.hpp"
#include "lbann/optimizers/optimizer.hpp"
#include <numeric>
#include<sys/types.h>
#include <unistd.h>

namespace lbann {

weights::weights(lbann_comm* comm,
                 cudnn::cudnn_manager* cudnn)
  : m_comm(comm),
    m_cudnn(cudnn),
    m_frozen(false) {

  // Initialize weights name
  static int num_weights = 0;
  m_name = "weights" + std::to_string(num_weights);
  num_weights++;

  // Zero initialization is default
  m_initializer = new constant_initializer(m_comm, DataType(0));

}

weights::weights(const weights& other)
  : m_name(other.m_name),
    m_comm(other.m_comm),
    m_cudnn(other.m_cudnn),
    m_matrix_height_dims(other.m_matrix_height_dims),
    m_matrix_width_dims(other.m_matrix_width_dims),
    m_values(other.m_values),
    m_initializer(other.m_initializer),
    m_optimizer(other.m_optimizer),
    m_frozen(other.m_frozen) {

  // Create deep copy of pointers
  if (m_values != nullptr)      { m_values = m_values->Copy(); }
  if (m_initializer != nullptr) { m_initializer = m_initializer->copy(); }
  if (m_optimizer != nullptr) {
    m_optimizer = m_optimizer->copy();
    m_optimizer->set_weights(*this);
  }

}

weights& weights::operator=(const weights& other) {
  m_name = other.m_name;
  m_comm = other.m_comm;
  m_cudnn = other.m_cudnn;
  m_matrix_height_dims = other.m_matrix_height_dims;
  m_matrix_width_dims = other.m_matrix_width_dims;

  // Deep copies
  if (m_values != nullptr)      { delete m_values; }
  if (m_initializer != nullptr) { delete m_initializer; }
  if (m_optimizer != nullptr)   { delete m_optimizer; }
  m_values = other.m_values;
  m_initializer = other.m_initializer;
  m_optimizer = other.m_optimizer;
  if (m_values != nullptr)      { m_values = m_values->Copy(); }
  if (m_initializer != nullptr) { m_initializer = m_initializer->copy(); }
  if (m_optimizer != nullptr)   { m_optimizer = m_optimizer->copy(); }

  m_frozen = other.m_frozen;

  return *this;
}

weights::~weights() {
  if (m_values != nullptr)      { delete m_values; }
  if (m_initializer != nullptr) { delete m_initializer; }
  if (m_optimizer != nullptr)   { delete m_optimizer; }
}

  void weights::setup(int size, El::Device dev) {
  setup(std::vector<int>(1, size), std::vector<int>(), El::STAR, El::STAR, dev);
}

  void weights::setup(std::vector<int> tensor_dims, El::Device dev) {
  setup(tensor_dims, std::vector<int>(), El::STAR, El::STAR, dev);
}

void weights::setup(int matrix_height,
                    int matrix_width,
                    El::Distribution col_dist,
                    El::Distribution row_dist,
                    El::Device dev) {
  setup(std::vector<int>(1, matrix_height),
        std::vector<int>(1, matrix_width),
        col_dist, row_dist, dev);
}

void weights::setup(std::vector<int> matrix_height_dims,
                    std::vector<int> matrix_width_dims,
                    El::Distribution col_dist,
                    El::Distribution row_dist,
                    El::Device dev) {

  if (m_values != nullptr) {
    // Check that dimensions are unchanged if weights are already
    // initialized
    const El::DistData dist_data(*m_values);
    if (m_matrix_height_dims == matrix_height_dims
        && m_matrix_width_dims == matrix_width_dims
        && dist_data.colDist == col_dist
        && dist_data.rowDist == row_dist) {
      return;
    } else {
      std::stringstream err;
      err << "attempted to setup " << m_name << " as a "
          << get_dims_string(matrix_height_dims, matrix_width_dims) << " "
          << "weights matrix with "
          << "col_dist=" << col_dist << ", "
          << "row_dist=" << row_dist << ", "
          << "but it is already setup as a "
          << get_dims_string(m_matrix_height_dims, m_matrix_width_dims) << " "
          << "matrix with "
          << "col_dist=" << dist_data.colDist << ", "
          << "row_dist=" << dist_data.rowDist;
      LBANN_ERROR(err.str());
    }
  } else {
    // Check that tensor dimensions are valid
    bool dims_are_valid = true;
    for (const auto& d : matrix_height_dims) {
      if (d <= 0) { dims_are_valid = false; }
    }
    for (const auto& d : matrix_width_dims) {
      if (d <= 0) { dims_are_valid = false; }
    }
    if (!dims_are_valid) {
      std::stringstream err;
      err << "attempted to setup " << m_name << " as a "
          << get_dims_string(matrix_height_dims, matrix_width_dims) << " "
          << "weights matrix";
      LBANN_ERROR(err.str());
    }
  }

  // Initialize weights matrix
  m_matrix_height_dims = matrix_height_dims;
  m_matrix_width_dims = matrix_width_dims;
  m_values = m_initializer->construct_matrix(get_matrix_height(),
                                             get_matrix_width(),
                                             col_dist,
                                             row_dist,
                                             dev);

  // Setup optimizer
  if (m_optimizer != nullptr) {
    m_optimizer->setup(*this);
  }

}

std::vector<int> weights::get_dims() const {
  const auto& width_dims = get_matrix_width_dims();
  const auto& height_dims = get_matrix_height_dims();
  std::vector<int> dims;
  dims.reserve(width_dims.size() + height_dims.size());
  for (const auto& d : width_dims)  { dims.push_back(d); }
  for (const auto& d : height_dims) { dims.push_back(d); }
  return dims;
}

int weights::get_matrix_height() const {
  const auto& height_dims = get_matrix_height_dims();
  return std::accumulate(height_dims.begin(), height_dims.end(),
                         1, std::multiplies<int>());
}

int weights::get_matrix_width() const {
  const auto& width_dims = get_matrix_width_dims();
  return std::accumulate(width_dims.begin(), width_dims.end(),
                         1, std::multiplies<int>());
}

void weights::set_initializer(weights_initializer* initializer) {
  if (m_initializer != nullptr) { delete m_initializer; }
  m_initializer = initializer;
}

void weights::set_optimizer(optimizer* opt) {
  if (m_optimizer != nullptr) { delete m_optimizer; }
  m_optimizer = opt;
}

AbsDistMat& weights::get_values() {
  if (m_values == nullptr) {
    LBANN_ERROR("attempted to access values of weights before they are setup");
  }
  return *m_values;
}

const AbsDistMat& weights::get_values() const {
  if (m_values == nullptr) {
    LBANN_ERROR("attempted to access values of weights before they are setup");
  }
  return *m_values;
}

void weights::set_values(const AbsDistMat& values) {

  // Check if weights matrix has been setup
  if (m_values == nullptr) {
    LBANN_ERROR("attempted to set values of weights before they are setup");
  }

  // Copy input to weights matrix
  El::Copy(values, *m_values);
}

void weights::set_value(DataType value, int index) {

#ifdef LBANN_DEBUG
  // Check that tensor position is valid
  const auto& size = get_size();
  if (index < 0 || index >= size) {
    std::stringstream err;
    err << "attempted to set weight value at index " << index << ", "
        << "but there are " << size << " values";
    LBANN_ERROR(err.str());
  }
#endif // LBANN_DEBUG

  // Set matrix entry
  const auto& height = get_matrix_height();
  set_value(value, index % height, index / height);

}

void weights::set_value(DataType value, std::vector<int> pos) {

  // Get tensor dimensions
  const auto& dims = get_dims();

#ifdef LBANN_DEBUG
  // Check that tensor position is valid
  bool pos_is_valid = true;
  if (dims.size() != pos.size()) {
    pos_is_valid = false;
  } else {
    for (size_t i = 0 ; i < dims.size(); ++i) {
      if (pos[i] < 0 || pos[i] >= dims[i]) { pos_is_valid = false;}
    }
  }
  if (!pos_is_valid) {
    std::stringstream err;
    err << "attempted to set weight value at position (";
    for (size_t i = 0 ; i < pos.size(); ++i) {
      err << (i > 0 ? "x" : "") << pos[i];
    }
    err << ") in a tensor with dimensions ";
    for (size_t i = 0 ; i < dims.size(); ++i) {
      err << (i > 0 ? "x" : "") << dims[i];
    }
    LBANN_ERROR(err.str());
  }
#endif // LBANN_DEBUG

  // Get index of weight value and set
  int index = 0;
  for (size_t i = 0; i < dims.size(); ++i) {
    index = index * dims[i] + pos[i];
  }
  set_value(value, index);

}

void weights::set_value(DataType value, int row, int col) {

#ifdef LBANN_DEBUG
  {
    // Check that matrix entry is valid
    const auto& height = get_matrix_height();
    const auto& width = get_matrix_width();
    if (row < 0 || row >= height || col < 0 || col > width ) {
      std::stringstream err;
      err << "attempted to set weights value at entry "
          << "(" << row << "," << col << ") "
          << "in a " << height << "x" << width << " matrix";
      LBANN_ERROR(err.str());
    }
  }
#endif // LBANN_DEBUG

  // Set value if it is local
  if (m_values->IsLocal(row, col)) {
    const El::Int local_row = m_values->LocalRow(row);
    const El::Int local_col = m_values->LocalCol(col);
    m_values->SetLocal(local_row, local_col, value);
  }

}

void weights::get_values_view(AbsDistMat& values_v) {
  const AbsDistMat& values = get_values();
  if (values.DistData() == values_v.DistData()
      && m_cudnn == nullptr) {
    El::LockedView(values_v, values);
  }
  else {
    El::Copy(values, values_v);
  }
}

std::string weights::get_dims_string(const std::vector<int>& matrix_height_dims,
                                     const std::vector<int>& matrix_width_dims) {
  std::stringstream ss;
  ss << "(";
  for (size_t i = 0; i < matrix_height_dims.size(); ++i) {
    ss << (i > 0 ? "x" : "") << matrix_height_dims[i];
  }
  ss << ")x(";
  for (size_t i = 0; i < matrix_width_dims.size(); ++i) {
    ss << (i > 0 ? "x" : "") << matrix_width_dims[i];
  }
  ss << ")";
  return ss.str();
}

bool weights::save_to_checkpoint_shared(lbann::persist& p)
{
  // define name to store weight values
  char l_name[512];
  sprintf(l_name, "weights_%s_%lldx%lld", m_name.c_str(), m_values->Height(), m_values->Width());
  // write weights using persist call -- uses Elemental's write function.
  p.write_distmat(persist_type::model, l_name, m_values);
  // if saving training state, also write out state of optimizer
  if (m_optimizer != nullptr) {
    m_optimizer->save_to_checkpoint_shared(p, m_name);
  }

  return true;
}

void weights::write_proto(lbann_data::WeightsData* proto) const {

  // Set proto properties
  proto->Clear();
  proto->set_name(m_name);
  for (const auto& d : get_dims()) {
    proto->mutable_shape()->add_dim(d);
  }
  proto->set_height(get_matrix_height());
  proto->set_width(get_matrix_width());

  // Write weight values to prototext on world master process
  CircMat<El::Device::CPU> values = *m_values; /// @todo What if weights are on GPU?
  values.SetRoot(0); /// @todo What if world master is not process 0?
  if (m_comm->am_world_master()) {
    const auto& local_values = values.LockedMatrix();
    const El::Int height = local_values.Height();
    const El::Int width = local_values.Width();
    /// @todo OpenMP parallelization
    /** @todo Our matrices are column-major while Numpy expects
     *  row-major matrices. This row-wise iteration is fine for
     *  matrices and column vectors, but it can mess up the order of
     *  the weights if a high-dimensional tensor is represented as a
     *  matrix. This is what we need for quantization on convolution
     *  kernel weights.
     */
    for (El::Int i = 0; i < height; ++i) {
      for (El::Int j = 0; j < width; ++j) {
        proto->add_data(local_values(i,j));
      }
    }
  }

}

bool weights::load_from_checkpoint_shared(lbann::persist& p)
{
  // define filename containing saved weight values
  char l_name[512], f_name[512];
  sprintf(l_name, "weights_%s_%lldx%lld", m_name.c_str(), m_values->Height(), m_values->Width());
  sprintf(f_name, "%s.bin", l_name);
  p.read_distmat(persist_type::model, f_name, m_values);
  if (m_optimizer != nullptr) {
    m_optimizer->load_from_checkpoint_shared(p, m_name);
  }

  return true;
}

bool weights::load_from_save(std::string ckpt_dir, std::vector<std::string> weight_list){
  // create weight file name to match to weight list entry
  char l_name[1024];
  sprintf(l_name, "model_weights_%s_%lldx%lld.bin", m_name.c_str(), m_values->Height(), m_values->Width());
  std::vector<std::string>::iterator it;
  it = find(weight_list.begin(),weight_list.end(),l_name);
  auto pos = std::distance(weight_list.begin(),it);
  // If match is found read in weight values.
  if((unsigned) pos < weight_list.size()){
    std::string full_path = ckpt_dir + weight_list[pos];
    if(m_comm->am_world_master())
      std::cout << "Loading " << m_name <<  "\n";
    El::Read(*m_values,full_path, El::BINARY, true);

  }
  return true;
}

bool weights::save_to_checkpoint_distributed(lbann::persist& p){
  // Functions identically to shared checkpoint except weights and parameters are saved on a per rank basis
  char l_name[512];
  sprintf(l_name, "weights_%s_%lldx%lld", m_name.c_str(), m_values->LocalHeight(), m_values->LocalWidth());
  p.write_rank_distmat(persist_type::model, l_name, *m_values);
  if (m_optimizer != nullptr)
    m_optimizer->save_to_checkpoint_distributed(p, m_name);
  return true;
}

bool weights::load_from_checkpoint_distributed(lbann::persist& p){
  // Functions identically to shared checkpoint except weights and parameters are loaded on a per rank basis
  char l_name[512];
  sprintf(l_name, "weights_%s_%lldx%lld", m_name.c_str(), m_values->LocalHeight(), m_values->LocalWidth());
  p.read_rank_distmat(persist_type::model, l_name, *m_values);
  if (m_optimizer != nullptr)
    m_optimizer->load_from_checkpoint_distributed(p, m_name);
  return true;
}


}  // namespace lbann
