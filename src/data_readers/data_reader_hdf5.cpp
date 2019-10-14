////////////////////////////////////////////////////////////////////////////////
//// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
//// Produced at the Lawrence Livermore National Laboratory.
//// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
//// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
////
//// LLNL-CODE-697807.
//// All rights reserved.
////
//// This file is part of LBANN: Livermore Big Artificial Neural Network
//// Toolkit. For details, see http://software.llnl.gov/LBANN or
//// https://github.com/LLNL/LBANN.
////
//// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
//// may not use this file except in compliance with the License.  You may
//// obtain a copy of the License at:
////
//// http://www.apache.org/licenses/LICENSE-2.0
////
//// Unless required by applicable law or agreed to in writing, software
//// distributed under the License is distributed on an "AS IS" BASIS,
//// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
//// implied. See the License for the specific language governing
//// permissions and limitations under the license.
////
///////////////////////////////////////////////////////////////////////////////////
#include "lbann/data_readers/data_reader_hdf5.hpp"
#include "lbann/utils/profiling.hpp"
#include <cstdio>
#include <string>
#include <fstream>
#include <unordered_set>
#include <iostream>
#include <cstring>
#include "lbann/utils/distconv.hpp"

namespace {
inline hid_t check_hdf5(hid_t hid, const char *file, int line) {
  if (hid < 0) {
    std::cerr << "HDF5 error" << std::endl;
    std::cerr << "Error at " << file << ":" << line << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  return hid;
}
}

#define CHECK_HDF5(call) check_hdf5(call, __FILE__, __LINE__)

namespace lbann {
  const std::string hdf5_reader::HDF5_KEY_DATA = "full";
  const std::string hdf5_reader::HDF5_KEY_RESPONSES = "unitPar";

  hdf5_reader::hdf5_reader(const bool shuffle)
    : generic_data_reader(shuffle) {}

  void hdf5_reader::read_hdf5(hsize_t h_data, hsize_t filespace, int rank, std::string key, hsize_t* dims, DataType * data_out) {
    // this is the splits, right now it is hard coded to split along the z axis
    int num_io_parts = dc::get_number_of_io_partitions();
    int ylines = 1;
    int xlines = 1;
    int zlines = num_io_parts;
    int channellines = 1;

    hsize_t xPerNode = dims[3]/xlines;
    hsize_t yPerNode = dims[2]/ylines;
    hsize_t zPerNode = dims[1]/zlines;
    hsize_t cPerNode = dims[0]/channellines;
    // how many times the pattern should repeat in the hyperslab
    hsize_t count[4] = {1,1,1,1};
    // local dimensions aka the dimensions of the slab we will read in
    hsize_t dims_local[4] = {cPerNode, zPerNode, yPerNode, xPerNode};

    // necessary for the hdf5 lib
    hid_t memspace = H5Screate_simple(4, dims_local, NULL);
    int spatial_offset = rank%num_io_parts;

    hsize_t offset[4] = {0, zPerNode*spatial_offset, 0, 0};

    // from an explanation of the hdf5 select_hyperslab:
    // start -> a starting location for the hyperslab
    // stride -> the number of elements to separate each element or block to be selected
    // count -> the number of elemenets or blocks to select along each dimension
    // block -> the size of the block selected from the dataspace
    //hsize_t status;

    //todo add error checking
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, dims_local);
    H5Dread(h_data, H5T_NATIVE_SHORT, memspace, filespace, H5P_DEFAULT, data_out);
  }

  void hdf5_reader::load() {
    lbann_comm* l_comm = get_comm();
    MPI_Comm mpi_comm = dc::get_input_comm(*l_comm);
    int world_rank = dc::get_input_rank(*l_comm);
    int color = world_rank/dc::get_number_of_io_partitions();
    MPI_Comm_split(mpi_comm, color, world_rank, &m_comm);
    m_shuffled_indices.clear();
    m_shuffled_indices.resize(m_file_paths.size());
    std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    if ((nprocs%dc::get_number_of_io_partitions()) !=0) {
      std::cerr<<"nprocs should be divisible by num of io partitions otherwise this wont work \n";
    }
    m_data_dims = {4, 512, 512, 512};
    m_num_features = std::accumulate(m_data_dims.begin(),
                                     m_data_dims.end(),
                                     (size_t) 1,
                                     std::multiplies<size_t>());
#ifdef DATA_READER_HDF5_USE_MPI_IO
    m_fapl = H5Pcreate(H5P_FILE_ACCESS);
    CHECK_HDF5(H5Pset_fapl_mpio(m_fapl, m_comm, MPI_INFO_NULL));
    m_dxpl = H5Pcreate(H5P_DATASET_XFER);
    CHECK_HDF5(H5Pset_dxpl_mpio(m_dxpl, H5FD_MPIO_COLLECTIVE));
#else
    m_fapl = H5P_DEFAULT;
    m_dxpl = H5P_DEFAULT;
#endif
    select_subset_of_data();
  }
  bool hdf5_reader::fetch_label(Mat& Y, int data_id, int mb_idx) {
    return true;
  }
  bool hdf5_reader::fetch_datum(Mat& X, int data_id, int mb_idx) {
    prof_region_begin("fetch_datum", prof_colors[0], false);
    int world_rank = dc::get_input_rank(*get_comm());

    auto file = m_file_paths[data_id];
    hid_t h_file = H5Fopen(file.c_str(), H5F_ACC_RDONLY, m_fapl);

#if 1
    dc::MPIPrintStreamInfo() << "HDF5 file opened: "
                             << file;
#endif

    if (h_file < 0) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
          " hdf5_reader::load() - can't open file : " + file);
    }

    // load in dataset
    hid_t h_data = CHECK_HDF5(
        H5Dopen(h_file, HDF5_KEY_DATA.c_str(), H5P_DEFAULT));
    hid_t filespace = CHECK_HDF5(H5Dget_space(h_data));
    //get the number of dimesnionse from the dataset
    int rank1 = H5Sget_simple_extent_ndims(filespace);
    hsize_t dims[rank1];
    // read in what the dimensions are
    CHECK_HDF5(H5Sget_simple_extent_dims(filespace, dims, NULL));

    if (h_data < 0) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
          " hdf5_reader::load() - can't find hdf5 key : " + HDF5_KEY_DATA);
    }

    // In the Cosmoflow case, each minibatch should have only one
    // sample per rank.
    assert_eq(X.Width(), 1);
    // Assuming 512^3 samples
    assert_eq(X.Height(),
              512 * 512 * 512 * 4 / dc::get_number_of_io_partitions()
              / (sizeof(DataType) / sizeof(short)));

    DataType *dest = X.Buffer();
    read_hdf5(h_data, filespace, world_rank, HDF5_KEY_DATA, dims, dest);
    //close data set
    H5Dclose(h_data);
    if (m_has_responses) {
      h_data = H5Dopen(h_file, HDF5_KEY_RESPONSES.c_str(), H5P_DEFAULT);
      H5Dread(h_data, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_all_responses);
      H5Dclose(h_data);
    }
    H5Fclose(h_file);

    //TODO do i need this?
    // not if I pass a ref to X I dont think
    //this should be equal to num_nuerons/LBANN_NUM_IO_PARTITIONS
    //unsigned long int pixelcount = m_image_width*m_image_height*m_image_depth*m_image_num_channels;
    // #ifdef LBANN_DISTCONV_COSMOFLOW_KEEP_INT16
    //    std::memcpy(dest,data, sizeof(short)*pixelcount);
    // #else
    //    LBANN_OMP_PARALLEL_FOR
    //       for(int p = 0; p<pixelcount; p++) {
    //TODO what is m_scaling_factor_int16
    //           dest[p] = tmp[p] * m_scaling_factor_int16;
    // mash this with above
    //X.Set(p, mb_idx,*tmp++);
    //       }
    // #endif
    //auto pixel_col = X(El::IR(0, X.Height()), El::IR(mb_idx, mb_idx+1));
    //std::vector<size_t> dims = {
    //  1ull,
    //  static_cast<size_t>(m_image_height),
    //  static_cast<size_t>(m_image_width)};
    //m_transform_pipeline.apply(pixel_col, dims);
    prof_region_end("fetch_datum", false);
    return true;
  }
  //get from a cached response
  bool hdf5_reader::fetch_response(Mat& Y, int data_id, int mb_idx) {
    prof_region_begin("fetch_response", prof_colors[0], false);
    assert_eq(Y.Height(), 4);
    std::memcpy(Y.Buffer(), &m_all_responses,
                m_num_response_features*sizeof(DataType));
    prof_region_end("fetch_response", false);
    return true;
  }

};
