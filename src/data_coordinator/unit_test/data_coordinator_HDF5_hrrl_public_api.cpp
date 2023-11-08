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
#include "lbann/data_coordinator/buffered_data_coordinator.hpp"
#include "lbann/proto/lbann.pb.h"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/threads/thread_pool.hpp"
#include "lbann/utils/threads/thread_utils.hpp"
#include <google/protobuf/text_format.h>

#include <conduit/conduit.hpp>
#include <cstdlib>
#include <errno.h>
#include <ostream>
#include <string.h>

#include "../src/data_readers/unit_test/test_data/hdf5_hrrl_test_data_and_schemas.yaml" // THIS IS NOT A YAML FILE
#include "lbann/data_readers/data_reader_HDF5.hpp"

class DataReaderHDF5WhiteboxTester
{
public:
  void normalize(lbann::hdf5_data_reader& x,
                 conduit::Node& node,
                 const std::string& path,
                 const conduit::Node& metadata)
  {
    x.normalize(node, path, metadata);
  }
  void repack_image(lbann::hdf5_data_reader& x,
                    conduit::Node& node,
                    const std::string& path,
                    const conduit::Node& metadata)
  {
    x.repack_image(node, path, metadata);
  }

  void pack(lbann::hdf5_data_reader& x, conduit::Node& node, size_t index)
  {
    x.pack(node, index);
  }

  void parse_schemas(lbann::hdf5_data_reader& x) { return x.parse_schemas(); }

  conduit::Node& get_data_schema(lbann::hdf5_data_reader& x)
  {
    return x.m_data_schema;
  }

  conduit::Node& get_experiment_schema(lbann::hdf5_data_reader& x)
  {
    return x.m_experiment_schema;
  }

  void set_data_schema(lbann::hdf5_data_reader& x, const conduit::Node& s)
  {
    x.set_data_schema(s);
  }

  void set_experiment_schema(lbann::hdf5_data_reader& x, const conduit::Node& s)
  {
    x.set_experiment_schema(s);
  }

  void print_metadata(lbann::hdf5_data_reader& x, std::ostream& os = std::cout)
  {
    x.print_metadata(os);
  }

  void set_delete_packed_fields(lbann::hdf5_data_reader& x, const bool flag)
  {
    x.set_delete_packed_fields(flag);
  }

  bool fetch_data_field(lbann::hdf5_data_reader& dr,
                        lbann::data_field_type data_field,
                        lbann::CPUMat& X,
                        int data_id,
                        int mb_idx)
  {
    return dr.fetch_data_field(data_field, X, data_id, mb_idx);
  }

  bool fetch_datum(lbann::hdf5_data_reader& dr,
                   lbann::CPUMat& X,
                   int data_id,
                   int mb_idx)
  {
    return dr.fetch_datum(X, data_id, mb_idx);
  }

  bool fetch_response(lbann::hdf5_data_reader& dr,
                      lbann::CPUMat& X,
                      int data_id,
                      int mb_idx)
  {
    return dr.fetch_response(X, data_id, mb_idx);
  }

  bool fetch_label(lbann::hdf5_data_reader& dr,
                   lbann::CPUMat& X,
                   int data_id,
                   int mb_idx)
  {
    return dr.fetch_label(X, data_id, mb_idx);
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

TEST_CASE("Data Coordinator hdf5 conduit fetch tests",
          "[data_coordinator][hdf5][hrrl][conduit]")
{
  // initialize stuff (boilerplate)
  auto& comm = unit_test::utilities::current_world_comm();
  lbann::init_random(0, 2);
  lbann::init_data_seq_random(42);

  conduit::Node ref_node;
  ref_node.parse(hdf5_hrrl_data_sample_id, "yaml");

  // std::unique_ptr<lbann::data_coordinator> dc =
  // lbann::make_unique<lbann::buffered_data_coordinator<float>>(&comm);
  //   dc->set_trainer(*this);

  lbann::hdf5_data_reader* hdf5_dr = new lbann::hdf5_data_reader();
  DataReaderHDF5WhiteboxTester white_box_tester;

  // Setup the data schema for this HRRL data set
  conduit::Node& data_schema = white_box_tester.get_data_schema(*hdf5_dr);
  data_schema.parse(hdf5_hrrl_data_schema, "yaml");
  conduit::Node& experiment_schema =
    white_box_tester.get_experiment_schema(*hdf5_dr);
  experiment_schema.parse(hdf5_hrrl_experiment_schema, "yaml");
  white_box_tester.parse_schemas(*hdf5_dr);
  // Manually tell the data reader to extract all of the data fields
  white_box_tester.construct_linearized_size_lookup_tables(*hdf5_dr, ref_node);

  hdf5_dr->set_comm(&comm);

  El::Int num_samples = 1;

  lbann::dataset dataset;
  dataset.setup(num_samples, "training");

  auto data_store = new lbann::data_store_conduit(hdf5_dr);
  hdf5_dr->set_data_store(data_store);
  // Take the sample and place it into the data store
  int index = 0;
  auto& ds = hdf5_dr->get_data_store();
  conduit::Node& ds_node = ds.get_empty_node(index);
  ds_node.parse(hdf5_hrrl_data_sample_id, "yaml");

  // Once the node is constructed pack the requested fields into the node
  size_t sample_index = 334;
  white_box_tester.set_delete_packed_fields(*hdf5_dr, false);
  white_box_tester.pack(*hdf5_dr, ds_node, sample_index);
  white_box_tester.construct_linearized_size_lookup_tables(*hdf5_dr, ds_node);

  ds.set_preloaded_conduit_node(index, ds_node);

  // Initalize a per-trainer I/O thread pool
  auto io_thread_pool = lbann::make_unique<lbann::thread_pool>();
  io_thread_pool->launch_pinned_threads(1, 1);
  hdf5_dr->setup(io_thread_pool->get_num_threads(), io_thread_pool.get());

  hdf5_dr->m_shuffled_indices.emplace_back(0);

  SECTION("fetch conduit node - data field")
  {
    std::vector<conduit::Node> samples(1);
    El::Matrix<El::Int> indices_fetched;
    indices_fetched.Resize(1, 1);
    hdf5_dr->fetch(samples,
                   indices_fetched,
                   dataset.get_position(),
                   dataset.get_sample_stride(),
                   1);
    //    auto valid = hdf5_dr->fetch(samples, indices_fetched, 2);

    //    lbann::CPUMat X;
    // Check the primary data fields
    std::vector<std::string> fields =
      {"Epmax", "Etot", "Image", "N", "T", "alpha"};
    for (auto& data_field : fields) {
      const std::string test_pathname("000000334/" + data_field);
      for (El::Int j = 0; j < num_samples; j++) {
        conduit::Node& sample = samples[j];
        // Check to make sure that each element in the transformed field are
        // properly normalized
        size_t num_elements =
          ref_node[test_pathname].dtype().number_of_elements();
        if (num_elements > 1) {
          for (size_t i = 0; i < num_elements; i++) {
            double test = sample[test_pathname].as_double_array()[i];
            double check = ref_node[test_pathname].as_double_array()[i];
            CHECK(test == Approx(check));
          }
        }
        else {
          double test = sample[test_pathname].as_double();
          double check = ref_node[test_pathname].as_double();
          CHECK(test == Approx(check));
        }
      }
    }

    // Check the packed fields
    // Get the reference packed node
    conduit::Node packed_ref_node;
    packed_ref_node.parse(packed_hdf5_hrrl_data_sample_id, "yaml");

    std::vector<std::string> packed_fields = {"samples", "responses"};
    for (auto& data_field : packed_fields) {
      const std::string test_pathname("000000334/" + data_field);
      for (El::Int j = 0; j < num_samples; j++) {
        conduit::Node& sample = samples[j];
        // Check to make sure that each element in the transformed field are
        // properly normalized
        size_t num_elements =
          packed_ref_node[test_pathname].dtype().number_of_elements();
        if (num_elements > 1) {
          for (size_t i = 0; i < num_elements; i++) {
            double test = sample[test_pathname].as_double_array()[i];
            double check = packed_ref_node[test_pathname].as_double_array()[i];
            CHECK(test == Approx(check));
          }
        }
        else {
          double test = sample[test_pathname].as_double();
          double check = packed_ref_node[test_pathname].as_double();
          CHECK(test == Approx(check));
        }
      }
    }
  }

  SECTION("fetch conduit node - invalid vector")
  {
    std::vector<conduit::Node> samples;
    El::Matrix<El::Int> indices_fetched;
    indices_fetched.Resize(1, 1);
    CHECK_THROWS(hdf5_dr->fetch(samples,
                                indices_fetched,
                                dataset.get_position(),
                                dataset.get_sample_stride(),
                                1));
  }

  SECTION("fetch conduit node - mini-batch too large")
  {
    std::vector<conduit::Node> samples(1);
    El::Matrix<El::Int> indices_fetched;
    indices_fetched.Resize(1, 1);
    CHECK_THROWS(hdf5_dr->fetch(samples,
                                indices_fetched,
                                dataset.get_position(),
                                dataset.get_sample_stride(),
                                2));
  }
}
