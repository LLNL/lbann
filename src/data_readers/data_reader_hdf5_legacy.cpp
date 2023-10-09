////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
/////////////////////////////////////////////////////////////////////////////////
#include "lbann/data_readers/data_reader_hdf5_legacy.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"
#include "lbann/comm_impl.hpp"
#include "lbann/utils/distconv.hpp"
#include "lbann/utils/profiling.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>

namespace {
inline hid_t check_hdf5(hid_t hid, const char* file, int line)
{
  if (hid < 0) {
    std::cerr << "HDF5 error" << std::endl;
    std::cerr << "Error at " << file << ":" << line << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  return hid;
}
} // namespace

#define CHECK_HDF5(call) check_hdf5(call, __FILE__, __LINE__)

namespace lbann {

template <typename TensorDataType>
hdf5_reader<TensorDataType>::hdf5_reader(const bool shuffle,
                                         const std::string key_data,
                                         const std::string key_labels,
                                         const std::string key_responses,
                                         const bool hyperslab_labels)
  : generic_data_reader(shuffle),
    m_use_data_store(
      global_argument_parser().get<bool>(LBANN_OPTION_USE_DATA_STORE)),
    m_key_data(key_data),
    m_key_labels(key_labels),
    m_key_responses(key_responses),
    m_hyperslab_labels(hyperslab_labels)
{}

template <typename TensorDataType>
hdf5_reader<TensorDataType>::hdf5_reader(const hdf5_reader& rhs)
  : generic_data_reader(rhs)
{
  copy_members(rhs);
}

template <typename TensorDataType>
hdf5_reader<TensorDataType>&
hdf5_reader<TensorDataType>::operator=(const hdf5_reader<TensorDataType>& rhs)
{
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }
  generic_data_reader::operator=(rhs);
  copy_members(rhs);
  return (*this);
}

template <typename TensorDataType>
void hdf5_reader<TensorDataType>::copy_members(const hdf5_reader& rhs)
{
  if (rhs.m_data_store != nullptr) {
    m_data_store = new data_store_conduit(rhs.get_data_store());
  }
  m_data_store->set_data_reader_ptr(this);

  m_num_features = rhs.m_num_features;
  m_data_dims = rhs.m_data_dims;
  m_hyperslab_dims = rhs.m_hyperslab_dims;
  m_comm = rhs.m_comm;
  m_file_paths = rhs.m_file_paths;
  m_use_data_store = rhs.m_use_data_store;
  m_key_data = rhs.m_key_data;
  m_key_labels = rhs.m_key_labels;
  m_key_responses = rhs.m_key_responses;
  m_hyperslab_labels = rhs.m_hyperslab_labels;
  m_all_responses = rhs.m_all_responses;
}

template <typename TensorDataType>
void hdf5_reader<TensorDataType>::read_hdf5_hyperslab(hsize_t h_data,
                                                      hsize_t filespace,
                                                      int rank,
                                                      TensorDataType* sample)
{
  prof_region_begin("read_hdf5_hyperslab", prof_colors[0], false);
  // this is the splits, right now it is hard coded to split along the
  // z axis
  int num_io_parts = dc::get_number_of_io_partitions();

  // how many times the pattern should repeat in the hyperslab
  hsize_t count[4] = {1, 1, 1, 1};

  // necessary for the hdf5 lib
  hid_t memspace = H5Screate_simple(4, m_hyperslab_dims.data(), NULL);
  int spatial_offset = rank % num_io_parts;
  hsize_t offset[4] = {0, m_hyperslab_dims[1] * spatial_offset, 0, 0};

  // from an explanation of the hdf5 select_hyperslab:
  // start -> a starting location for the hyperslab
  // stride -> the number of elements to separate each element or block to be
  // selected count -> the number of elemenets or blocks to select along each
  // dimension block -> the size of the block selected from the dataspace
  // hsize_t status;

  CHECK_HDF5(H5Sselect_hyperslab(filespace,
                                 H5S_SELECT_SET,
                                 offset,
                                 NULL,
                                 count,
                                 m_hyperslab_dims.data()));

  CHECK_HDF5(
    H5Dread(h_data, get_hdf5_data_type(), memspace, filespace, m_dxpl, sample));
  prof_region_end("read_hdf5_hyperslab", false);
}

template <typename TensorDataType>
void hdf5_reader<TensorDataType>::read_hdf5_sample(int data_id,
                                                   TensorDataType* sample,
                                                   TensorDataType* labels)
{
  int world_rank = get_comm()->get_rank_in_trainer();
  auto file = m_file_paths[data_id];
  hid_t h_file = CHECK_HDF5(H5Fopen(file.c_str(), H5F_ACC_RDONLY, m_fapl));

  // load in dataset
  hid_t h_data = CHECK_HDF5(H5Dopen(h_file, m_key_data.c_str(), H5P_DEFAULT));
  hid_t filespace = CHECK_HDF5(H5Dget_space(h_data));
  // get the number of dimensions from the dataset
  int rank1 = H5Sget_simple_extent_ndims(filespace);
  hsize_t dims[rank1];
  // read in what the dimensions are
  CHECK_HDF5(H5Sget_simple_extent_dims(filespace, dims, NULL));

  read_hdf5_hyperslab(h_data, filespace, world_rank, sample);
  // close data set
  CHECK_HDF5(H5Dclose(h_data));

  if (this->has_labels() && labels != nullptr) {
    assert_always(m_hyperslab_labels);
    hid_t h_labels =
      CHECK_HDF5(H5Dopen(h_file, m_key_labels.c_str(), H5P_DEFAULT));
    hid_t filespace_labels = CHECK_HDF5(H5Dget_space(h_labels));
    read_hdf5_hyperslab(h_labels, filespace_labels, world_rank, labels);
    CHECK_HDF5(H5Dclose(h_labels));
  }
  else if (this->has_responses()) {
    assert_always(labels == nullptr);
    h_data = CHECK_HDF5(H5Dopen(h_file, m_key_responses.c_str(), H5P_DEFAULT));
    CHECK_HDF5(H5Dread(h_data,
                       H5T_NATIVE_FLOAT,
                       H5S_ALL,
                       H5S_ALL,
                       H5P_DEFAULT,
                       &m_all_responses[0]));
    CHECK_HDF5(H5Dclose(h_data));
  }
  CHECK_HDF5(H5Fclose(h_file));
  return;
}

template <typename TensorDataType>
void hdf5_reader<TensorDataType>::load()
{
  lbann_comm* l_comm = get_comm();
  MPI_Comm mpi_comm = l_comm->get_trainer_comm().GetMPIComm();
  int world_rank = l_comm->get_rank_in_trainer();
  int color = world_rank / dc::get_number_of_io_partitions();
  MPI_Comm_split(mpi_comm, color, world_rank, &m_comm);
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_file_paths.size());
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  int nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  if ((nprocs % dc::get_number_of_io_partitions()) != 0) {
    LBANN_ERROR("nprocs should be divisible by num of io partitions otherwise "
                "this wont work");
  }

  // Read the dimension size of the first sample,
  // assuming that all of the samples have the same dimension size
  if (m_file_paths.size() > 0) {
    const hid_t h_file =
      CHECK_HDF5(H5Fopen(m_file_paths[0].c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
    const hid_t h_data =
      CHECK_HDF5(H5Dopen(h_file, m_key_data.c_str(), H5P_DEFAULT));
    const hid_t h_space = CHECK_HDF5(H5Dget_space(h_data));
    if (CHECK_HDF5(H5Sget_simple_extent_ndims(h_space)) != 4) {
      LBANN_ERROR("The number of dimensions of HDF5 data samples should be 4");
    }
    hsize_t dims[4];
    CHECK_HDF5(H5Sget_simple_extent_dims(h_space, dims, NULL));
    CHECK_HDF5(H5Dclose(h_data));
    m_data_dims = std::vector<El::Int>(dims, dims + 4);
  }
  else {
    LBANN_ERROR("The number of HDF5 samples should not be zero");
  }

  m_num_features = std::accumulate(m_data_dims.begin(),
                                   m_data_dims.end(),
                                   (size_t)1,
                                   std::multiplies<size_t>());

  for (auto i : m_data_dims) {
    m_hyperslab_dims.push_back(i);
  }
  // Partition the z dimension
  m_hyperslab_dims[1] /= dc::get_number_of_io_partitions();

#define DATA_READER_HDF5_USE_MPI_IO
#ifdef DATA_READER_HDF5_USE_MPI_IO
  m_fapl = CHECK_HDF5(H5Pcreate(H5P_FILE_ACCESS));
  CHECK_HDF5(H5Pset_fapl_mpio(m_fapl, m_comm, MPI_INFO_NULL));
  m_dxpl = CHECK_HDF5(H5Pcreate(H5P_DATASET_XFER));
  CHECK_HDF5(
    H5Pset_dxpl_mpio(m_dxpl, H5FD_MPIO_INDEPENDENT)); // H5FD_MPIO_COLLECTIVE
#else
  m_fapl = H5P_DEFAULT;
  m_dxpl = H5P_DEFAULT;
#endif
  std::vector<int> local_list_sizes;
  auto& arg_parser = global_argument_parser();
  if (arg_parser.get<bool>(LBANN_OPTION_PRELOAD_DATA_STORE)) {
    LBANN_ERROR("preload_data_store not supported on HDF5 data reader");
  }
  if (m_use_data_store) {
    instantiate_data_store();
  }

  select_subset_of_data();
  MPI_Comm_dup(dc::get_mpi_comm(), &m_response_gather_comm);
}

template <typename TensorDataType>
void hdf5_reader<TensorDataType>::load_sample(conduit::Node& node, int data_id)
{
  const std::string conduit_key = LBANN_DATA_ID_STR(data_id);
  auto& conduit_obj = node[conduit_key + "/slab"];
  conduit_obj.set(
    get_conduit_data_type(m_num_features / dc::get_number_of_io_partitions()));
  TensorDataType* sample_buf = conduit_obj.value();
  if (this->has_labels()) {
    assert_always(m_hyperslab_labels);
    auto& conduit_labels_obj = node[conduit_key + "/labels_slab"];
    conduit_labels_obj.set(get_conduit_data_type(
      m_num_features / dc::get_number_of_io_partitions()));
    TensorDataType* labels_buf = conduit_labels_obj.value();
    read_hdf5_sample(data_id, sample_buf, labels_buf);
  }
  else {
    read_hdf5_sample(data_id, sample_buf, nullptr);
  }
  if (this->has_responses()) {
    node[conduit_key + "/responses"].set(&m_all_responses[0],
                                         m_all_responses.size());
  }
  if (priming_data_store()) {
    // Once the node has been populated save it in the data store
    m_data_store->set_conduit_node(data_id, node);
  }
}

template <typename TensorDataType>
bool hdf5_reader<TensorDataType>::fetch_label(Mat& Y, int data_id, int mb_idx)
{
  if (!this->has_labels()) {
    return generic_data_reader::fetch_label(Y, data_id, mb_idx);
  }

  prof_region_begin("fetch_label", prof_colors[0], false);
  assert_always(m_hyperslab_labels);
  assert_always(m_use_data_store);
  TensorDataType* buf = nullptr;
  assert_eq((unsigned long)Y.Height(), m_num_features);
  conduit::Node node;
  const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
  node.set_external(ds_node);
  const std::string conduit_obj = LBANN_DATA_ID_STR(data_id);
  buf = node[conduit_obj + "/labels_slab"].value();
  auto Y_v = create_datum_view(Y, mb_idx);
  std::memcpy(Y_v.Buffer(),
              buf,
              m_num_features / dc::get_number_of_io_partitions() *
                sizeof(TensorDataType));
  prof_region_end("fetch_label", false);
  return true;
}

template <typename TensorDataType>
bool hdf5_reader<TensorDataType>::fetch_data_field(data_field_type data_field,
                                                   CPUMat& Y,
                                                   int data_id,
                                                   int mb_idx)
{
  if (data_field != INPUT_DATA_TYPE_LABEL_RECONSTRUCTION) {
    NOT_IMPLEMENTED(data_field);
  }

  prof_region_begin("fetch_label_reconstruction", prof_colors[0], false);
  assert_always(m_hyperslab_labels);
  assert_always(m_use_data_store);
  TensorDataType* buf = nullptr;
  assert_eq((unsigned long)Y.Height(), m_num_features);
  conduit::Node node;
  if (m_use_data_store &&
      (data_store_active() || m_data_store->has_conduit_node(data_id))) {
    prof_region_begin("get_conduit_node", prof_colors[0], false);
    const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
    node.set_external(ds_node);
    prof_region_end("get_conduit_node", false);
  }
  else {
    load_sample(node, data_id);
  }

  //  const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
  //  node.set_external(ds_node);
  const std::string conduit_key = LBANN_DATA_ID_STR(data_id);
  buf = node[conduit_key + "/labels_slab"].value();
  auto Y_v = create_datum_view(Y, mb_idx);
  std::memcpy(Y_v.Buffer(),
              buf,
              m_num_features / dc::get_number_of_io_partitions() *
                sizeof(TensorDataType));
  prof_region_end("fetch_label_reconstruction", false);
  return true;
}

template <typename TensorDataType>
bool hdf5_reader<TensorDataType>::fetch_datum(Mat& X, int data_id, int mb_idx)
{
  prof_region_begin("fetch_datum", prof_colors[0], false);

  // Note (trb 02/06/2023): By making this a constexpr check, the
  // compiler can deduce that the following "assert_eq" will not
  // divide by zero in any surviving case. A necessary condition for
  // sizeof(DataType) % sizeof(TensorDataType) == 0ul is that
  // sizeof(DataType) >= sizeof(TensorDataType), so (sizeof(DataType)
  // / sizeof(TensorDataType)) >= 1 in all surviving cases.
  if constexpr (sizeof(DataType) % sizeof(TensorDataType) > 0ul) {
    LBANN_ERROR("Invalid configuration.");
    return false;
  }
  else {
    assert_eq((unsigned long)X.Height(),
              m_num_features / dc::get_number_of_io_partitions() /
                (sizeof(DataType) / sizeof(TensorDataType)));
  }

  auto X_v = create_datum_view(X, mb_idx);
  if (m_use_data_store) {
    fetch_datum_conduit(X_v, data_id);
  }
  else {
    read_hdf5_sample(data_id, (TensorDataType*)X_v.Buffer(), nullptr);
  }
  prof_region_end("fetch_datum", false);
  return true;
}

template <typename TensorDataType>
void hdf5_reader<TensorDataType>::fetch_datum_conduit(Mat& X, int data_id)
{
  const std::string conduit_key = LBANN_DATA_ID_STR(data_id);
  // Create a node to hold all of the data
  conduit::Node node;
  if (m_use_data_store &&
      (data_store_active() || m_data_store->has_conduit_node(data_id))) {
    prof_region_begin("get_conduit_node", prof_colors[0], false);
    const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
    node.set_external(ds_node);
    prof_region_end("get_conduit_node", false);
  }
  else {
    load_sample(node, data_id);
  }
  prof_region_begin("set_external", prof_colors[0], false);
  conduit::Node slab;
  slab.set_external(node[conduit_key + "/slab"]);
  prof_region_end("set_external", false);
  TensorDataType* data = slab.value();
  prof_region_begin("copy_to_buffer", prof_colors[0], false);
  std::memcpy(X.Buffer(),
              data,
              slab.dtype().number_of_elements() * slab.dtype().element_bytes());
  prof_region_end("copy_to_buffer", false);
}

// get from a cached response
template <typename TensorDataType>
bool hdf5_reader<TensorDataType>::fetch_response(Mat& Y,
                                                 int data_id,
                                                 int mb_idx)
{
  if (!this->has_responses()) {
    return generic_data_reader::fetch_response(Y, data_id, mb_idx);
  }

  prof_region_begin("fetch_response", prof_colors[0], false);
  float* buf = nullptr;
  if (m_hyperslab_labels) {
    assert_eq((unsigned long)Y.Height(), m_num_features);
    const std::string conduit_key = LBANN_DATA_ID_STR(data_id);
    conduit::Node node;
    const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
    node.set_external(ds_node);
    conduit::Node slab;
    slab.set_external(node[conduit_key + "/responses_slab"]);
    prof_region_end("set_external", false);
    buf = slab.value();
    auto Y_v = create_datum_view(Y, mb_idx);
    std::memcpy(Y_v.Buffer(), buf, m_num_features * sizeof(TensorDataType));
  }
  else {
    assert_eq((unsigned long)Y.Height(), m_all_responses.size());
    conduit::Node node;
    if (m_use_data_store &&
        (data_store_active() || m_data_store->has_conduit_node(data_id))) {
      const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
      node.set_external(ds_node);
    }
    else {
      load_sample(node, data_id);
    }
    const std::string conduit_obj = LBANN_DATA_ID_STR(data_id);
    buf = node[conduit_obj + "/responses"].value();
    auto Y_v = create_datum_view(Y, mb_idx);
    std::memcpy(Y_v.Buffer(), buf, m_all_responses.size() * sizeof(DataType));
  }
  prof_region_end("fetch_response", false);
  return true;
}

template <>
hid_t hdf5_reader<float>::get_hdf5_data_type() const
{
  return H5T_NATIVE_FLOAT;
}
template <>
hid_t hdf5_reader<double>::get_hdf5_data_type() const
{
  return H5T_NATIVE_DOUBLE;
}
template <>
hid_t hdf5_reader<short>::get_hdf5_data_type() const
{
  return H5T_NATIVE_SHORT;
}

template <>
conduit::DataType
hdf5_reader<float>::get_conduit_data_type(conduit::index_t num_elements) const
{
  return conduit::DataType::float32(num_elements);
}
template <>
conduit::DataType
hdf5_reader<double>::get_conduit_data_type(conduit::index_t num_elements) const
{
  return conduit::DataType::float64(num_elements);
}
template <>
conduit::DataType
hdf5_reader<short>::get_conduit_data_type(conduit::index_t num_elements) const
{
  return conduit::DataType::int16(num_elements);
}

// TODO (oyamay): Instantiate hdf5_reader<short> for large samples
#define PROTO(T) template class hdf5_reader<T>;

#include "lbann/macros/instantiate.hpp"

} // namespace lbann
