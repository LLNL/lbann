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
namespace lbann {
  const std::string hdf5_reader::HDF5_KEY_DATA = "full";
  const std::string hdf5_reader::HDF5_KEY_RESPONSES = "physPar";

  hdf5_reader::hdf5_reader(const bool shuffle)
    : generic_data_reader(shuffle) {}

  void hdf5_reader::read_hdf5(hsize_t h_data, hsize_t filespace, int rank, std::string key, hsize_t* dims, DataType * data_out) {
    // this is the splits, right now it is hard coded to split along the z axis
    int num_io_parts = dc::get_number_of_io_partitions();
    int ylines = 1;
    int xlines = 1;
    int zlines = num_io_parts;
    int channellines = 1;

    // todo: when taking care of the odd case this cant be an int
    int xPerNode = dims[3]/xlines;
    int yPerNode = dims[2]/ylines;
    int zPerNode = dims[1]/zlines;
    int cPerNode = dims[0]/channellines;
    // offset in each dimension
    hsize_t offset[4];
    // how many times the pattern should repeat in the hyperslab
    hsize_t count[4] = {1,1,1,1};
    // local dimensions aka the dimensions of the slab we will read in
    hsize_t dims_local[4] = {cPerNode, zPerNode, yPerNode, xPerNode};

    // necessary for the hdf5 lib
    hid_t memspace = H5Screate_simple(4, dims_local, NULL);
    int spatial_offset = rank%num_io_parts

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
    const El::mpi::Comm & w_comm = l_comm->get_world_comm();
    MPI_Comm mpi_comm = w_comm.GetMPIComm();
    int world_rank = get_rank_in_world();
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
    select_subset_of_data();
  }
  bool hdf5_reader::fetch_label(Mat& Y, int data_id, int mb_idx) {
    return true;
  }
  bool hdf5_reader::fetch_datum(Mat& X, int data_id, int mb_idx) {
    prof_region_begin("fetch_datum", prof_colors[0], false); 
    int world_rank = get_rank_in_world();

    auto file = m_file_paths[data_id];

    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id, m_comm, MPI_INFO_NULL);
    hid_t dxpl_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE); 
    hid_t h_file = H5Fopen(file.c_str(), H5F_ACC_RDONLY, fapl_id);
    
    if (h_file < 0) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + 
          " hdf5_reader::load() - can't open file : " + file);
    }

    // load in dataset
    hid_t h_data =  H5Dopen(h_file, HDF5_KEY_DATA.c_str(), dxpl_id);
    hid_t filespace = H5Dget_space(h_data);
    //get the number of dimesnionse from the dataset
    int rank1 = H5Sget_simple_extent_ndims(filespace);
    hsize_t dims[rank1];
    // read in what the dimensions are
    H5Sget_simple_extent_dims(filespace, dims, NULL);

    if (h_data < 0) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
          " hdf5_reader::load() - can't find hdf5 key : " + HDF5_KEY_DATA);
    }

    //TODO: add the int 16 stuff
    // check if mb_idx needs to be changed to not be hard coded
    //int adj_mb_idx = mb_idx+(world_rank%dc::get_number_of_io_partitions());
    Mat X_v = El::View(X, El::IR(0,X.Height()), El::IR(mb_idx, adj_mb_idx+1));

    DataType *dest = X_v.Buffer();
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
    Mat Y_v = El::View(Y, El::IR(0, Y.Height()), El::IR(mb_idx, mb_idx+1));
    //TODO: possibly 4 tho, python tells me its float64
    std::memcpy(Y_v.Buffer(), &m_all_responses,
       m_num_responses_features*4);
    prof_region_end("fetch_response", false);
    return true;
  } 

};

