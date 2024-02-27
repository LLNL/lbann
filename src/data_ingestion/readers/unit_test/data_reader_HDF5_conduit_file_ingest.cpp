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
////////////////////////////////////////////////////////////////////////////////
#include "Catch2BasicSupport.hpp"

#include "MPITestHelpers.hpp"
#include "TestHelpers.hpp"
#include "lbann/proto/lbann.pb.h"
#include "lbann/proto/proto_common.hpp"
#include <google/protobuf/text_format.h>
#include "lbann/utils/conduit_extensions.hpp"

#include <conduit/conduit.hpp>
#include <conduit/conduit_relay_mpi.hpp>
// #include <conduit/conduit_relay_io_hdf5_api.hpp>
// #include "conduit_relay.hpp"
#include <conduit/conduit_relay_io_hdf5.hpp>
//#include "hdf5.h"
#include <cstdlib>
#include <errno.h>
#include <string.h>

#include "./data_reader_common_HDF5_test_utils.hpp"
#include "./data_reader_common_catch2.hpp"

#include "./test_data/hdf5_c3fd_test_data_and_schemas.yaml"
#include "lbann/data_ingestion/readers/data_reader_HDF5.hpp"

//-----------------------------------------------------------------------------
// helper to create an HDF5 dataset
herr_t
create_hdf5_nd_dataset(std::string fname, std::string path, int rank, int const * dims,
    hid_t mem_type, hid_t file_type, void * to_write)
{
    hid_t file;
    herr_t status = 0;

    // initialize count and dimensions
    std::vector<hsize_t> hdims(rank);
    for (int d = 0; d < rank; ++d)
    {
        hdims[d] = dims[d];
    }

    // create the file
    file = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // Create, init a dataspace for the dataset
    hid_t    dataset, dataspace;
    dataspace = H5Screate_simple(rank, hdims.data(), NULL);

    // Create, init the dataset.  Element type is double.
    dataset = H5Dcreate(file, path.c_str(), file_type, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, to_write);
    status = H5Dclose(dataset);

    // close the dataspace and file
    status = H5Sclose(dataspace);
    status = H5Fclose(file);

    return status;
}

TEST_CASE("HDF5 Conduit Hyperslab  data reader file ingest tests",
          "[.filesystem][data_reader][hdf5][conduit][file_ingest]")
{
  // initialize stuff (boilerplate)
  auto& comm = unit_test::utilities::current_world_comm();
  lbann::init_random(0, 2);
  lbann::init_data_seq_random(42);

  conduit::Node node;
  node.parse(hdf5_c3fd_data_sample, "yaml");

  auto hdf5_dr = std::make_unique<lbann::hdf5_data_reader>();
  hdf5_dr->set_comm(&comm);
  DataReaderHDF5WhiteboxTester white_box_tester;

#if 1
  SECTION("HDF5 Parallel I/O Conduit read from HDF5 file")
  {
    // create working directory
    std::string work_dir = create_test_directory("hdf5_reader");

    // Example from https://llnl-conduit.readthedocs.io/en/latest/relay_io.html#hdf5-hyperslabs
    // ------------------------------------------------------------------
    // Create a 2D array and show it off.
    int constexpr rank = 2;
    int constexpr nrows = 3;
    int constexpr ncols = 4;
    int constexpr eltcount = nrows * ncols;
    double data[eltcount];
    for (int i = 0; i < eltcount; ++i)
    {
        data[i] = i;
    }

    std::cout << "Array, in memory:\n";
    for (int j = 0; j < nrows; ++j)
    {
        for (int i = 0; i < ncols; ++i)
        {
            std::cout << std::right << std::setw(4) << data[j * ncols + i];
        }
        std::cout << std::endl;
    }

    // Create an HDF5 file with a 2D array.
    herr_t status = 0;
    // HDF5 dimensions are ordered from slowest- to fastest-varying.
    // This is the same as C and C++ nested arrays and opposite from
    // many people's geometric intuition.
    hsize_t hdims[rank]{ nrows, ncols };

    const char* fname = "t_relay_io_hdf5_read_ndarray.hdf5";
    hid_t file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // Create, initialize a dataspace for the dataset
    hid_t    dataset, dataspace;
    dataspace = H5Screate_simple(rank, hdims, NULL);

    // Create, initialize the dataset.  Element type is double.
    const char* dsname = "twoDarray";
    dataset = H5Dcreate(file, dsname, H5T_NATIVE_DOUBLE, dataspace,
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    status = H5Dclose(dataset);


// herr_t H5Sset_extent_simple	(	hid_t 	space_id,
// int 	rank,
// const hsize_t 	dims[],
// const hsize_t 	max[]
// )

    // close the dataspace and file
    status = H5Sclose(dataspace);
    status = H5Fclose(file);

    std::cout << "\nsaved array to '" << fname << ":" << dsname << "'" << std::endl;



    // ------------------------------------------------------------------
    // Now read a subset of that 2D array from the HDF5 file.
    // Two rows, two columns; total of four elements.
    int constexpr rnrows = 2;
    int constexpr rncols = 2;
    int constexpr reltcount = rnrows * rncols;
    // As noted earlier, HDF5 orders all dimensions from slowest- to
    // fastest-varying.  In this two-dimensional example, row (or y-index)
    // always comes before column (or x-index).  If working with a 3D
    // dataset, level (or z-index) would come before row.
    int p_sizes[rank]{ rnrows, rncols };
    // offset to row 0, column 1
    int p_offsets[rank]{ 0, 1 };
    // read every row, every other column
    int p_strides[rank]{ 1, 2 };
    // Store pointers to these parameters in the read_opts Node
    conduit::Node read_opts;
    read_opts["sizes"].set_external(p_sizes, rank);
    read_opts["offsets"].set_external(p_offsets, rank);
    read_opts["strides"].set_external(p_strides, rank);

    std::cout << "\nHDF5 Options for reading the array:" << std::endl;
    read_opts.print();

    // Read some of the 2D array in the HDF5 file into an array of doubles
    conduit::Node read_data;
    double p_data_out[reltcount]{42, 42, 42, 42};
    read_data.set_external(p_data_out, reltcount);
    std::cout << "Check the info befor ereading "<< std::endl;
    read_data.info().print();
    std::string in_path;
    in_path.append(fname).append(":").append(dsname);
    conduit::relay::io::hdf5_read(in_path.c_str(), read_opts, read_data);
    std::cout << "Check the info after reading "<< std::endl;
    // if(err) {
    //   std::cout << "I think that is failed." << std::endl;
    // }
    //    CONDUIT_CHECK_HDF5_ERROR(conduit::relay::io::hdf5_read(in_path.c_str(), read_opts, read_data), "Error opening hyperslab file.");

    read_data.info().print();
    read_data.print_detailed();

    // Show what we read
    std::cout << "Subset of array, read from '" << in_path << "'" << std::endl;
    double *foo = read_data.value();

    for (int j = 0; j < rnrows; ++j)
    {
        for (int i = 0; i < rncols; ++i)
        {
            std::cout << std::right << std::setw(8) << foo[j * rncols + i];
        }
        std::cout << std::endl;
    }
    for (int j = 0; j < rnrows; ++j)
    {
        for (int i = 0; i < rncols; ++i)
        {
            std::cout << std::right << std::setw(8) << p_data_out[j * rncols + i];
        }
        std::cout << std::endl;
    }

#if 0
    // open hdf5 file and obtain a handle
    hid_t h5_id =
      conduit::relay::io::hdf5_create_file(work_dir + "/C3FD_test_sample.hdf5");
    // write data
    conduit::relay::io::hdf5_write(node, h5_id);
    // close our file
    conduit::relay::io::hdf5_close_file(h5_id);

    hid_t h5_fid = conduit::relay::io::hdf5_open_file_for_read(
      work_dir + "/C3FD_test_sample.hdf5");
    const std::string original_path = "/RUN_ID/000000001";
    const std::string new_pathname = "000000001";

    // Setup the data schema for this C3FD data set
    conduit::Node& data_schema = white_box_tester.get_data_schema(*hdf5_dr);
    data_schema.parse(hdf5_c3fd_data_schema, "yaml");

    // Read in the experiment schema and setup the data reader
    conduit::Node& experiment_schema =
      white_box_tester.get_experiment_schema(*hdf5_dr);
    experiment_schema.parse(hdf5_c3fd_experiment_parallel_io_schema, "yaml");
    // experiment_schema.print();
    white_box_tester.parse_schemas(*hdf5_dr);

    white_box_tester.print_metadata(*hdf5_dr);

    conduit::Node test_node;
    white_box_tester.load_sample(*hdf5_dr,
                                 test_node[new_pathname],
                                 h5_fid,
                                 original_path);

    // Check to see if the HRRL sample can be read via the data
    // reader's load_sample method.  Note that this will coerce and
    // normalize all data fields as specified in the data set and
    // experiment schemas.
    std::vector<std::string> fields = {"NodeFeatures",
                                       "EdgeFeatures",
                                       "COOList"};
    check_node_fields(node,
                      test_node,
                      data_schema,
                      fields,
                      original_path,
                      new_pathname);
#endif
  }

  SECTION("HDF5 Parallel I/O Conduit read from HDF5 file - part 2")
  {
    // create working directory
    std::string work_dir = create_test_directory("hdf5_reader");

// create a simple buffer of doubles
    conduit::Node n;

    n["full_data"].set(conduit::DataType::c_double(20));

    double *vin = n["full_data"].value();

    for(int i=0;i<20;i++)
    {
        vin[i] = i;
    }

    std::cout << "Example Full Data" << std::endl;

    n.print();
    conduit::relay::io::hdf5_write(n,"tout_hdf5_slab_opts");
    //    conduit::relay::io::hdf5_write(n,"tout_hdf5_slab_opts.hdf5");

    // read 10 [1->11) entries (as above test, but using hdf5 read options)

    conduit::Node n_res;
    conduit::Node opts;
    opts["offset"] = 1;
    opts["stride"] = 2;
    opts["size"]   = 10;

    conduit::Node nload;
    conduit::relay::io::hdf5_read("tout_hdf5_slab_opts:full_data",opts,nload);
    //    conduit::relay::io::hdf5_read("tout_hdf5_slab_opts.hdf5:full_data",opts,nload);
    nload.print();


    nload.info().print();
    nload.print_detailed();

    std::cout << "Load Result" << std::endl;
    nload.print();

    double *vload = nload.value();
    for(int i=0;i<10;i++)
    {
      std::cout << vload[i] << " =?= " << (1.0 + i * 2.0) << std::endl;
        //        CHECK(vload[i],1.0 + i * 2.0,1e-3);
    }
  }
#endif

  SECTION("HDF5 Parallel I/O Conduit read from HDF5 file - part 3")
  {
    int constexpr rank = 2;
    int constexpr ncols = 5;
    int constexpr nrows = 3;
    int constexpr dset_size[rank] = { nrows, ncols };
    int constexpr nelts = ncols * nrows;

    conduit::Node n_in(conduit::DataType::float64(nelts));

    conduit::float64_array val_in = n_in.value();

    for(conduit::index_t i=0;i<nelts;i++)
    {
        val_in[i] = i;
    }

    std::cout << "Here is the original data" << std::endl;
    n_in.print_detailed();

    // Create an HDF5 data set in an HDF5 file
    hid_t mem_type = H5T_NATIVE_DOUBLE;
    hid_t file_type = H5T_NATIVE_DOUBLE;
    herr_t status = create_hdf5_nd_dataset("tout_hdf5_r_2D_array.hdf5",
        "myobj", rank, dset_size, mem_type, file_type, val_in.data_ptr());

    // Assert (not expect) status >= 0, to crash if the test fails
    //    ASSERT_GE(status, 0) << "Error creating the HDF5 test dataset.";

    // read in the whole thing
    conduit::Node n_whole_out;

    conduit::relay::io::hdf5_read("tout_hdf5_r_2D_array.hdf5:myobj",n_whole_out);

    std::cout << "Read the whole data set (doubles from 0 through 14):\n";
    n_whole_out.print();

    // should contain ncols x nrows elements
    CHECK(nelts == n_whole_out.dtype().number_of_elements());

    conduit::float64_array val_whole_out = n_whole_out.value();

    for(conduit::index_t i=0;i<nelts;i++)
    {
      if(val_in[i] != val_whole_out[i]) {
        std::cout << "These should match " << val_in[i] << " != " << val_whole_out[i] << std::endl;
      }
    }

    // // now read in the options of the hdf5 node
    // conduit::Node read_hdf5_options;
    // conduit::relay::io::hdf5_options(read_hdf5_options);
    // read_hdf5_options.print_detailed();

    // now read in part of the array
    conduit::Node read_opts;
    int constexpr rncols = 3;
    int constexpr rnrows = 2;
    int constexpr rnelts = rncols * rnrows;
    std::vector<int> size_ary;
    size_ary.push_back(rnrows);
    size_ary.push_back(rncols);
    read_opts["sizes"].set_external(size_ary);
    int constexpr rcoloff = 1;
    int constexpr rrowoff = 1;
    std::vector<int> offset_ary;
    offset_ary.push_back(rrowoff);
    offset_ary.push_back(rcoloff);
    read_opts["offsets"].set(offset_ary);

    conduit::Node n_out;

    conduit::relay::io::hdf5_read("tout_hdf5_r_2D_array.hdf5:myobj",read_opts,n_out);

    std::cout << "Read partial data set (2 rows, 3 cols, starting at (1, 1)):\n";
    n_out.print_detailed();

    // should contain ncols x nrows elements
    if(rnelts != n_out.dtype().number_of_elements()) {
      std::cout << "Number of elements is wrong " << rnelts << " != " << n_out.dtype().number_of_elements() << std::endl;
    }

    conduit::float64_array val_out = n_out.value();

    conduit::index_t offset = ncols * rrowoff;
    conduit::index_t linear_idx = 0;
    for (conduit::index_t j = 0; j < rnrows; j++)
    {
        for (conduit::index_t i = 0; i < rncols; i++)
        {
            CHECK(val_in[offset + rcoloff + i] == val_out[linear_idx]);
            linear_idx += 1;
        }
        offset += ncols;
    }

    // make sure we aren't leaking
    //    EXPECT_EQ(check_h5_open_ids(),DO_NO_HARM);
  }

  SECTION("HDF5 Parallel I/O Conduit read from HDF5 file - part 4")
  {
    // create a simple buffer of doubles
    conduit::Node n;

    n["full_data"].set(conduit::DataType::c_double(20));

    double *vin = n["full_data"].value();

    for(int i=0;i<20;i++)
    {
        vin[i] = i;
    }

    std::cout << "T4 Example Full Data" << std::endl;

    n.print();

    conduit::relay::io::hdf5_write(n,"tout_hdf5_slab.hdf5");

    conduit::Node nload;
    nload.set(conduit::DataType::c_double(10));

    double *vload = nload.value();

    // stride to read every other entry into compact storage
    conduit::relay::io::hdf5_read_dset_slab("tout_hdf5_slab.hdf5",
                        "full_data",
                        conduit::DataType::c_double(10,
                                           sizeof(double),   // offset 1 double
                                           sizeof(double)*2, //stride 2 doubles
                                           sizeof(double)),
                                           vload);
    std::cout << "T4 Load Result" << std::endl;;
    nload.print();

    for(int i=0;i<10;i++)
    {
        CHECK(vload[i] == 1.0 + i * 2.0);
    }
  }

  SECTION("HDF5 Parallel I/O Conduit read from HDF5 file - part 5")
  {
    // create a simple buffer of doubles
    conduit::Node n;

    n["full_data"].set(conduit::DataType::c_double(20));

    double *vin = n["full_data"].value();

    for(int i=0;i<20;i++)
    {
        vin[i] = i;
    }

    std::cout << "Example Full Data" << std::endl;

    n.print();
    conduit::relay::io::hdf5_write(n,"tout_hdf5_slab_opts_take2");
    //    conduit::relay::io::hdf5_write(n,"tout_hdf5_slab_opts_take2.hdf5");

    // read 10 [1->11) entries (as above test, but using hdf5 read options)

    conduit::Node n_res;
    conduit::Node opts;
    opts["offset"] = 1;
    opts["stride"] = 2;
    opts["size"]   = 10;

    conduit::Node nload;
    conduit::relay::io::hdf5_read("tout_hdf5_slab_opts_take2:full_data",opts,nload);
    //    conduit::relay::io::hdf5_read("tout_hdf5_slab_opts_take2.hdf5:full_data",opts,nload);
    nload.print();

    std::cout << "Load Result"<< std::endl;
    nload.print();

    double *vload = nload.value();
    for(int i=0;i<10;i++)
    {
        CHECK(vload[i] == 1.0 + i * 2.0);
        //        EXPECT_NEAR(vload[i],1.0 + i * 2.0,1e-3);
    }

  }
}
