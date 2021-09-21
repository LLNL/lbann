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

// It feels like we should be able to pack this node, but with the additional
// level of hierarchy in the sample name, it fails
const std::string hdf5_hrrl_data_sample =R"FOO(RUN_ID:
  000000334:
    Epmax: 15.2486634101312
    Etot: 0.0426354341969429
    Image: [456.288777930614, 231.340700217946, 113.528447010204, 115.115911382861, 116.716861149023, 118.331222098325, 120.52874207647, 122.175220756304, 123.834871115725, 125.507597035081, 126.011234474661, 123.587537036166]
    N: 64037572840.4818
    T: 5.34505173275895
    alpha: 32.6826031770453
)FOO";

// Use this version of the sample for the packing test
const std::string hdf5_hrrl_data_sample_id =R"FOO(000000334:
    Epmax: 15.2486634101312
    Etot: 0.0426354341969429
    Image: [456.288777930614, 231.340700217946, 113.528447010204, 115.115911382861, 116.716861149023, 118.331222098325, 120.52874207647, 122.175220756304, 123.834871115725, 125.507597035081, 126.011234474661, 123.587537036166]
    N: 64037572840.4818
    T: 5.34505173275895
    alpha: 32.6826031770453
)FOO";

// Here is how the HRRL data expects its sample to be packed for this experiment schema
const std::string packed_hdf5_hrrl_data_sample_id =R"FOO(000000334:
    datum: [456.288777930614, 231.340700217946, 113.528447010204, 115.115911382861, 116.716861149023, 118.331222098325, 120.52874207647, 122.175220756304, 123.834871115725, 125.507597035081, 126.011234474661, 123.587537036166]
    response: [15.2486634101312, 0.0426354341969429, 64037572840.4818, 5.34505173275895, 32.6826031770453]
)FOO";

// Use a different schema to create a different packing
const std::string packed_hdf5_hrrl_data_sample_id_foobar =R"FOO(000000334:
    datum: [456.288777930614, 231.340700217946, 113.528447010204, 115.115911382861, 116.716861149023, 118.331222098325, 120.52874207647, 122.175220756304, 123.834871115725, 125.507597035081, 126.011234474661, 123.587537036166]
    foo: [15.2486634101312, 0.0426354341969429]
    bar: [64037572840.4818, 5.34505173275895]
    baz: [32.6826031770453]
)FOO";

// Now change the ordering fields in the experiment schema to change the field order
const std::string packed_hdf5_hrrl_data_sample_id_foobar_permute =R"FOO(000000334:
    datum: [456.288777930614, 231.340700217946, 113.528447010204, 115.115911382861, 116.716861149023, 118.331222098325, 120.52874207647, 122.175220756304, 123.834871115725, 125.507597035081, 126.011234474661, 123.587537036166]
    foo: [0.0426354341969429, 15.2486634101312]
    bar: [5.34505173275895, 64037572840.4818]
    baz: [32.6826031770453]
)FOO";

const std::string hdf5_hrrl_data_schema_test = R"AurthurDent(
# Re, the "ordering" fields: ordering is relative and need not be unique;
# it specifies, e.g, the order in which a set of scalars
# would be appended to a vector.
#
# metadata values in the below schema can be over-ridden by values in
# the experiment_schema.yaml
#
# For reference: the metadata nodes may contain additional info,
# e.g, scale and bias for normalization.
#
# The intent is that the the schema and metadata values below should
# be reasonably static, while the experiment_schema species the
# subset of values to use in an experiment
#
#
Image:
  metadata:
    dims: [4,3]
    channels: 1
    ordering: 0
    scale: [1.5259021896696422e-05]
    bias: [-1.5259021896696422e-05]
Epmax:
  metadata:
    ordering: 10
    scale: 0.1
    bias: -1.0
Etot:
  metadata:
    ordering: 20
    scale: 0.3916485873519399
    bias: -0.00039973613068075743
T:
  metadata:
    ordering: 50
    scale: 0.125
    bias: -0.25
alpha:
  metadata:
    ordering: 60
    scale: 0.1
    bias: -2.5

N:
  metadata:
    ordering: 40
    scale: 3.1662826662374707e-13
    bias: -0.001001267234978943
Xshift:
  metadata:
    ordering: 70
Yshift:
  metadata:
    ordering: 80
)AurthurDent";

const std::string hdf5_hrrl_experiment_schema_test = R"AurthurDent(
Image:
  metadata:
    pack: "datum"
    coerce: "float"
Epmax:
  metadata:
    pack: "response"
Etot:
  metadata:
    pack: "response"
N:
  metadata:
    pack: "response"
T:
  metadata:
    pack: "response"
alpha:
  metadata:
    pack: "response"
)AurthurDent";

const std::string hdf5_hrrl_experiment_schema_test_foobar = R"AurthurDent(
Image:
  metadata:
    pack: "datum"
    coerce: "float"
Epmax:
  metadata:
    pack: "foo"
Etot:
  metadata:
    pack: "foo"
N:
  metadata:
    pack: "bar"
T:
  metadata:
    pack: "bar"
alpha:
  metadata:
    pack: "baz"
)AurthurDent";

// Change how the experiment data should be packed and ordered within each field
const std::string hdf5_hrrl_experiment_schema_test_foobar_permute = R"AurthurDent(
Image:
  metadata:
    pack: "datum"
    coerce: "float"
Epmax:
  metadata:
    pack: "foo"
    ordering: 5
Etot:
  metadata:
    pack: "foo"
    ordering: 4
N:
  metadata:
    pack: "bar"
    ordering: 3
T:
  metadata:
    pack: "bar"
    ordering: 2
alpha:
  metadata:
    pack: "baz"
    ordering: 1
)AurthurDent";

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

  // LBANN_MSG("I think that the initial node name is ", node.name(), " nested with ", node.child(0).name());
  // conduit::Node compact_node;
  // compact_node["data"].update(node);

  lbann::hdf5_data_reader* hdf5_dr = new lbann::hdf5_data_reader();
  DataReaderHDF5WhiteboxTester white_box_tester;

  // Setup the data schema for this HRRL data set
  conduit::Node& data_schema = white_box_tester.get_data_schema(*hdf5_dr);
  data_schema.parse(hdf5_hrrl_data_schema_test, "yaml");
  conduit::Node& experiment_schema = white_box_tester.get_experiment_schema(*hdf5_dr);
  experiment_schema.parse(hdf5_hrrl_experiment_schema_test, "yaml");
  //  hdf5_dr->set_shuffled_indices({0});
  white_box_tester.parse_schemas(*hdf5_dr);
  //  hdf5_dr->construct_linearized_size_lookup_tables(node);

  hdf5_dr->set_rank(0);
  hdf5_dr->set_comm(&comm);
  // conduit::Node schema;
  // schema.parse(hdf5_hrrl_data_schema_test, "yaml");

  El::Int num_samples = 1;

  auto data_store = new lbann::data_store_conduit(hdf5_dr);  // *data_store_conduit
  hdf5_dr->set_data_store(data_store);
  //  hdf5_dr->instantiate_data_store();
  // Take the sample and place it into the data store
  int index = 0;
  auto& ds = hdf5_dr->get_data_store();
  conduit::Node& ds_node = ds.get_empty_node(index);
  LBANN_MSG("Here is the empty node");
  ds_node.print();
  //  ds_node.update(compact_node);
  ds_node.parse(hdf5_hrrl_data_sample_id, "yaml");
  LBANN_MSG("Now it is full");
  ds_node.print();
  ds.set_preloaded_conduit_node(index, ds_node);
  //  ds.compact_nodes();
  LBANN_MSG("Here is the reference to the node that was put into the DS");
  ds_node.print();

  //  node.print();

  //  white_box_tester.parse_schemas(*hdf5_dr);

  // Initalize a per-trainer I/O thread pool
  auto io_thread_pool = lbann::make_unique<lbann::thread_pool>();
  io_thread_pool->launch_pinned_threads(1, 1);
  hdf5_dr->setup(io_thread_pool->get_num_threads(), io_thread_pool.get());
  hdf5_dr->set_num_parallel_readers(1);

  //  white_box_tester.parse_schemas(*hdf5_dr);

  SECTION("fetch data field")
  {
    lbann::CPUMat X;
    //std::vector<std::string> fields = {"Epmax", "Etot"};
    std::vector<std::string> fields = {"Epmax", "Etot", "N", "T", "alpha"};
    //    std::vector<std::string> fields = {"Epmax", "Etot", "Image", "N", "T", "alpha"};
    for (auto& data_field : fields) {
      std::cout << "Fetching " << data_field << std::endl;
      //      X.Resize(white_box_tester.get_linearized_size(*hdf5_dr, data_field), num_samples);
      X.Resize(1, num_samples);

      auto io_rng = lbann::set_io_generators_local_index(0);
      for (auto j = 0; j < num_samples; j++) {
        white_box_tester.fetch_data_field(*hdf5_dr, data_field, X, 0, j);
      }


      El::Print(X);;
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
  //    CHECK_THROWS(white_box_tester.fetch_data_field(*dr, "foobar", X, 0, 0));
  }
}
