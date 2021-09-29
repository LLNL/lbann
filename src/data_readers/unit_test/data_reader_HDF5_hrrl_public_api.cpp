////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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

#include <catch2/catch.hpp>

#include "MPITestHelpers.hpp"
#include "TestHelpers.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/threads/thread_pool.hpp"
#include "lbann/utils/threads/thread_utils.hpp"
#include <google/protobuf/text_format.h>
#include <lbann.pb.h>

#include <conduit/conduit.hpp>
#include <cstdlib>
#include <errno.h>
#include <string.h>

#include "lbann/data_readers/data_reader_HDF5.hpp"
#include "./test_data/hdf5_hrrl_experiment_schema.yaml"
#include "./test_data/hdf5_hrrl_test_data_and_schema.yaml"

class DataReaderHDF5WhiteboxTester
{
public:
  void normalize(lbann::hdf5_data_reader& x,
                 conduit::Node& node,
                 const std::string& path,
                 const conduit::Node& metadata)
  { x.normalize(node, path, metadata); }
  void repack_image(lbann::hdf5_data_reader& x,
                    conduit::Node& node,
                    const std::string& path,
                    const conduit::Node& metadata)
  { x.repack_image(node, path, metadata); }

  void pack(lbann::hdf5_data_reader& x,
            conduit::Node& node,
            size_t index)
  { x.pack(node, index); }

  void parse_schemas(lbann::hdf5_data_reader& x) {
    return x.parse_schemas();
  }

  conduit::Node& get_data_schema(lbann::hdf5_data_reader& x) {
    return x.m_data_schema;
  }

  conduit::Node& get_experiment_schema(lbann::hdf5_data_reader& x) {
    return x.m_experiment_schema;
  }

  void set_data_schema(lbann::hdf5_data_reader& x,
                       const conduit::Node& s) {
    x.set_data_schema(s);
  }

  void set_experiment_schema(lbann::hdf5_data_reader& x,
                             const conduit::Node& s) {
    x.set_experiment_schema(s);
  }

  bool fetch_data_field(lbann::hdf5_data_reader& dr,
                        lbann::data_field_type data_field,
                        lbann::CPUMat& X,
                        int data_id,
                        int mb_idx)
  {
    return dr.fetch_data_field(data_field, X, data_id, mb_idx);
  }

  int get_linearized_size(lbann::hdf5_data_reader& dr,
                          lbann::data_field_type const& data_field)
  {
    return dr.get_linearized_size(data_field);
  }
  void construct_linearized_size_lookup_tables(lbann::hdf5_data_reader& dr,
                                              conduit::Node& node)
  {
    return dr.construct_linearized_size_lookup_tables(node);
  }

};

TEST_CASE("hdf5 data reader data field fetch tests",
          "[data_reader][hdf5][hrrl][data_field]")
{
  // initialize stuff (boilerplate)
  auto& comm = unit_test::utilities::current_world_comm();
  lbann::init_random(0, 2);
  lbann::init_data_seq_random(42);

  conduit::Node ref_node;
  ref_node.parse(hdf5_hrrl_data_sample_id, "yaml");

  lbann::hdf5_data_reader* hdf5_dr = new lbann::hdf5_data_reader();
  DataReaderHDF5WhiteboxTester white_box_tester;

  // Setup the data schema for this HRRL data set
  conduit::Node& data_schema = white_box_tester.get_data_schema(*hdf5_dr);
  data_schema.parse(hdf5_hrrl_data_schema_test, "yaml");
  conduit::Node& experiment_schema = white_box_tester.get_experiment_schema(*hdf5_dr);
  experiment_schema.parse(hdf5_hrrl_experiment_schema, "yaml");
  white_box_tester.parse_schemas(*hdf5_dr);
  // Manually tell the data reader to extract all of the data fields
  white_box_tester.construct_linearized_size_lookup_tables(*hdf5_dr, ref_node);

  hdf5_dr->set_rank(0);
  hdf5_dr->set_comm(&comm);

  El::Int num_samples = 1;

  auto data_store = new lbann::data_store_conduit(hdf5_dr);
  hdf5_dr->set_data_store(data_store);
  // Take the sample and place it into the data store
  int index = 0;
  auto& ds = hdf5_dr->get_data_store();
  conduit::Node& ds_node = ds.get_empty_node(index);
  ds_node.parse(hdf5_hrrl_data_sample_id, "yaml");
  ds.set_preloaded_conduit_node(index, ds_node);

  // Initalize a per-trainer I/O thread pool
  auto io_thread_pool = lbann::make_unique<lbann::thread_pool>();
  io_thread_pool->launch_pinned_threads(1, 1);
  hdf5_dr->setup(io_thread_pool->get_num_threads(), io_thread_pool.get());
  hdf5_dr->set_num_parallel_readers(1);

  SECTION("fetch data field")
  {
    lbann::CPUMat X;
    std::vector<std::string> fields = {"Epmax", "Etot", "Image", "N", "T", "alpha"};
    for (auto& data_field : fields) {
      X.Resize(white_box_tester.get_linearized_size(*hdf5_dr, data_field), num_samples);

      auto io_rng = lbann::set_io_generators_local_index(0);
      for (auto j = 0; j < num_samples; j++) {
        white_box_tester.fetch_data_field(*hdf5_dr, data_field, X, 0, j);
      }

      const std::string test_pathname("000000334/" + data_field);
      for (El::Int j = 0; j < num_samples; j++) {
        // Check to make sure that each element in the transformed field are properly normalized
        size_t num_elements = ref_node[test_pathname].dtype().number_of_elements();
        if(num_elements > 1) {
          for(size_t i = 0; i < num_elements; i++) {
            double check = ref_node[test_pathname].as_double_array()[i];
            CHECK(X(i,0) == Approx(check));
          }
        }
        else {
          double check = ref_node[test_pathname].as_double();
          CHECK(X(0,0) == Approx(check));
        }
      }
    }
  }

  SECTION("fetch invalid data field")
  {
    lbann::CPUMat X;
    std::vector<std::string> fields = {"foo"};
    for (auto& data_field : fields) {
      CHECK_THROWS(X.Resize(white_box_tester.get_linearized_size(*hdf5_dr, data_field), num_samples));

      auto io_rng = lbann::set_io_generators_local_index(0);
      for (auto j = 0; j < num_samples; j++) {
        CHECK_THROWS(white_box_tester.fetch_data_field(*hdf5_dr, data_field, X, 0, j));
      }
    }
  }
}
