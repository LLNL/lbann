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
/////////////////////////////////////////////////////////////////////////////////
#include "lbann/data_readers/data_reader_hdf5.hpp"
#include "lbann/utils/profiling.hpp"
#include "lbann/utils/distconv.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"

#include <cstdio>
#include <string>
#include <fstream>
#include <unordered_set>
#include <iostream>
#include <cstring>

namespace {
inline hid_t check_hdf5(hid_t hid, const char *file, int line) {
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

const std::string hdf5_reader::HDF5_KEY_DATA = "full";
const std::string hdf5_reader::HDF5_KEY_RESPONSES = "unitPar";

hdf5_reader::hdf5_reader(const bool shuffle)
    : generic_data_reader(shuffle),
      m_use_data_store(options::get()->get_bool("use_data_store")) {
}

hdf5_reader::hdf5_reader(const hdf5_reader& rhs)  : generic_data_reader(rhs) {
  copy_members(rhs);
}

hdf5_reader& hdf5_reader::operator=(const hdf5_reader& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }
  generic_data_reader::operator=(rhs);
  copy_members(rhs);
  return (*this);
}

void hdf5_reader::copy_members(const hdf5_reader &rhs) {
  if(rhs.m_data_store != nullptr) {
    m_data_store = new data_store_conduit(rhs.get_data_store());
  }
  m_data_store->set_data_reader_ptr(this);

  m_has_labels = rhs.m_has_labels;
  m_has_responses = rhs.m_has_responses;
  m_num_features = rhs.m_num_features;
  m_data_dims = rhs.m_data_dims;
  m_hyperslab_dims = rhs.m_hyperslab_dims;
  m_comm = rhs.m_comm;
  m_file_paths = rhs.m_file_paths;
  m_use_data_store = rhs.m_use_data_store;

  for(size_t i = 0; i < m_num_response_features; i++) {
    m_all_responses[i] = rhs.m_all_responses[i];
  }
}

void hdf5_reader::read_hdf5_hyperslab(hsize_t h_data, hsize_t filespace,
                                      int rank, short *sample) {
  prof_region_begin("read_hdf5_hyperslab", prof_colors[0], false);
  // this is the splits, right now it is hard coded to split along the
  // z axis
  int num_io_parts = dc::get_number_of_io_partitions();

  // how many times the pattern should repeat in the hyperslab
  hsize_t count[4] = {1,1,1,1};

  // necessary for the hdf5 lib
  hid_t memspace = H5Screate_simple(4, m_hyperslab_dims.data(), NULL);
  int spatial_offset = rank % num_io_parts;
  hsize_t offset[4] = {0, m_hyperslab_dims[1] * spatial_offset, 0, 0};

  // from an explanation of the hdf5 select_hyperslab:
  // start -> a starting location for the hyperslab
  // stride -> the number of elements to separate each element or block to be selected
  // count -> the number of elemenets or blocks to select along each dimension
  // block -> the size of the block selected from the dataspace
  //hsize_t status;

  CHECK_HDF5(H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                                 offset, NULL, count,
                                 m_hyperslab_dims.data()));

  CHECK_HDF5(H5Dread(h_data, H5T_NATIVE_SHORT, memspace,
                     filespace, m_dxpl, sample));
  prof_region_end("read_hdf5_hyperslab", false);
}

void hdf5_reader::read_hdf5_sample(int data_id, short *sample) {
  int world_rank = get_comm()->get_rank_in_trainer();
  auto file = m_file_paths[data_id];
  hid_t h_file = CHECK_HDF5(H5Fopen(file.c_str(), H5F_ACC_RDONLY, m_fapl));

  // load in dataset
  hid_t h_data = CHECK_HDF5(
      H5Dopen(h_file, HDF5_KEY_DATA.c_str(), H5P_DEFAULT));
  hid_t filespace = CHECK_HDF5(H5Dget_space(h_data));
  //get the number of dimesnionse from the dataset
  int rank1 = H5Sget_simple_extent_ndims(filespace);
  hsize_t dims[rank1];
  // read in what the dimensions are
  CHECK_HDF5(H5Sget_simple_extent_dims(filespace, dims, NULL));

  read_hdf5_hyperslab(h_data, filespace, world_rank, sample);
  //close data set
  CHECK_HDF5(H5Dclose(h_data));

  if (m_has_responses) {
    h_data = CHECK_HDF5(H5Dopen(h_file, HDF5_KEY_RESPONSES.c_str(), H5P_DEFAULT));
    CHECK_HDF5(H5Dread(h_data, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_all_responses));
    CHECK_HDF5(H5Dclose(h_data));
  }
  CHECK_HDF5(H5Fclose(h_file));
  return;
}

void hdf5_reader::load() {
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
  if ((nprocs % dc::get_number_of_io_partitions()) !=0) {
    LBANN_ERROR("nprocs should be divisible by num of io partitions otherwise this wont work");
  }

  // Read the dimension size of the first sample,
  // assuming that all of the samples have the same dimension size
  if (m_file_paths.size() > 0) {
    const hid_t h_file = CHECK_HDF5(H5Fopen(m_file_paths[0].c_str(),
                                            H5F_ACC_RDONLY, H5P_DEFAULT));
    const hid_t h_data = CHECK_HDF5(H5Dopen(h_file, HDF5_KEY_DATA.c_str(),
                                            H5P_DEFAULT));
    const hid_t h_space = CHECK_HDF5(H5Dget_space(h_data));
    if (CHECK_HDF5(H5Sget_simple_extent_ndims(h_space)) != 4) {
      LBANN_ERROR("The number of dimensions of HDF5 data samples should be 4");
    }
    hsize_t dims[4];
    CHECK_HDF5(H5Sget_simple_extent_dims(h_space, dims, NULL));
    CHECK_HDF5(H5Dclose(h_data));
    m_data_dims = std::vector<int>(dims, dims+4);
  } else {
    LBANN_ERROR("The number of HDF5 samples should not be zero");
  }

  m_num_features = std::accumulate(m_data_dims.begin(),
                                   m_data_dims.end(),
                                   (size_t) 1,
                                   std::multiplies<size_t>());


  for (auto i: m_data_dims) {
    m_hyperslab_dims.push_back(i);
  }
  // Partition the z dimension
  m_hyperslab_dims[1] /= dc::get_number_of_io_partitions();

#define DATA_READER_HDF5_USE_MPI_IO
#ifdef DATA_READER_HDF5_USE_MPI_IO
  m_fapl = CHECK_HDF5(H5Pcreate(H5P_FILE_ACCESS));
  CHECK_HDF5(H5Pset_fapl_mpio(m_fapl, m_comm, MPI_INFO_NULL));
  m_dxpl = CHECK_HDF5(H5Pcreate(H5P_DATASET_XFER));
  CHECK_HDF5(H5Pset_dxpl_mpio(m_dxpl, H5FD_MPIO_INDEPENDENT));  // H5FD_MPIO_COLLECTIVE
#else
  m_fapl = H5P_DEFAULT;
  m_dxpl = H5P_DEFAULT;
#endif
  std::vector<int> local_list_sizes;
  options *opts = options::get();
  if (opts->get_bool("preload_data_store")) {
    LBANN_ERROR("preload_data_store not supported on HDF5 data reader");
  }
  if (m_use_data_store) {
    instantiate_data_store();
  }

  select_subset_of_data();
  MPI_Comm_dup(dc::get_mpi_comm(), &m_response_gather_comm);
}

bool hdf5_reader::fetch_label(Mat& Y, int data_id, int mb_idx) {
  return true;
}

bool hdf5_reader::fetch_datum(Mat& X, int data_id, int mb_idx) {
  prof_region_begin("fetch_datum", prof_colors[0], false);

  // In the Cosmoflow case, each minibatch should have only one
  // sample per rank.
  assert_eq(X.Width(), 1);
  assert_eq(X.Height(),
            m_num_features / dc::get_number_of_io_partitions()
            / (sizeof(DataType) / sizeof(short)));

  if (m_use_data_store) {
    fetch_datum_conduit(X, data_id);
  } else {
    read_hdf5_sample(data_id, (short*)X.Buffer());
  }
  prof_region_end("fetch_datum", false);
  return true;
}

void hdf5_reader::fetch_datum_conduit(Mat& X, int data_id) {
  const std::string conduit_key = LBANN_DATA_ID_STR(data_id);
  // Create a node to hold all of the data
  conduit::Node node;
  if (data_store_active()) {
    prof_region_begin("get_conduit_node", prof_colors[0], false);
    const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
    node.set_external(ds_node);
    prof_region_end("get_conduit_node", false);
  } else {
    auto &conduit_obj = node[conduit_key + "/slab"];
    conduit_obj.set(conduit::DataType::int16(
        m_num_features / dc::get_number_of_io_partitions()));
    short *sample_buf = conduit_obj.value();
    read_hdf5_sample(data_id, sample_buf);
    node[conduit_key + "/responses"].set(m_all_responses, 4);
    if (priming_data_store()) {
      // Once the node has been populated save it in the data store
      m_data_store->set_conduit_node(data_id, node);
    }
  }
  prof_region_begin("set_external", prof_colors[0], false);
  conduit::Node slab;
  slab.set_external(node[conduit_key + "/slab"]);
  prof_region_end("set_external", false);
  short *data = slab.value();
  prof_region_begin("copy_to_buffer", prof_colors[0], false);
  std::memcpy(X.Buffer(), data, slab.dtype().number_of_elements()*slab.dtype().element_bytes());
  prof_region_end("copy_to_buffer", false);
}

//get from a cached response
bool hdf5_reader::fetch_response(Mat& Y, int data_id, int mb_idx) {
  prof_region_begin("fetch_response", prof_colors[0], false);
  assert_eq(Y.Height(), m_num_response_features);
  float *buf = nullptr;
  if (data_store_active()) {
    conduit::Node node;
    const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
    node.set_external(ds_node);
    const std::string conduit_obj = LBANN_DATA_ID_STR(data_id);
    buf = node[conduit_obj+"/responses"].value();
  }else {
    buf = m_all_responses;
  }
  std::memcpy(Y.Buffer(), buf,
              m_num_response_features*sizeof(DataType));
  gather_responses(Y.Buffer());
  prof_region_end("fetch_response", false);
  return true;
}

// Gather scattered responses to the first N ranks, where N is the
// mini-batch size. This is not necessary when the rank reordering
// is used.
void hdf5_reader::gather_responses(float *responses) {
  float recv_buf[m_num_response_features];
  const int rank = dc::get_mpi_rank();
  const int num_part = dc::get_number_of_io_partitions();
  const int mini_batch_size = this->get_loaded_mini_batch_size();
  const int src_rank = rank * num_part;
  const int dst_rank = rank / num_part;
  const int tag = 0;
  int req_idx = 0;
  MPI_Request req[2];

  // send
  if (rank % num_part == 0) {
    MPI_Isend(responses, m_num_response_features, MPI_FLOAT, dst_rank,
              tag, m_response_gather_comm, &req[req_idx]);
    ++req_idx;
  }

  // recv
  if (rank < mini_batch_size) {
    MPI_Irecv(recv_buf, m_num_response_features, MPI_FLOAT, src_rank, tag,
              m_response_gather_comm, &req[req_idx]);
    ++req_idx;
  }

  if (req_idx > 0) {
    MPI_Waitall(req_idx, req, MPI_STATUS_IGNORE);
  }

  std::memcpy(responses, recv_buf, sizeof(float) * m_num_response_features);
}

} // namespace lbann
