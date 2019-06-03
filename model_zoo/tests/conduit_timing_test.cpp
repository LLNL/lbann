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
////////////////////////////////////////////////////////////////////////////////

#include "lbann_config.hpp"

#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"
#include <vector>
#include <string>
#include "lbann/lbann.hpp"
#include "lbann/utils/jag_utils.hpp"

using namespace lbann;

void test_conduit(int from, int to, std::vector<std::string> filenames);
void test_conduit_2(int from, int to, std::vector<std::string> filenames);
void test_conduit_3(int from, int to, std::vector<std::string> filenames);

std::vector<std::string>  emi_v = {
  "(0.0, 0.0)/0.0",
  "(90.0, 0.0)/0.0",
  "(90.0, 78.0)/0.0"
};

std::vector<std::string> jag_scalar_keys = {
        "BWx",
        "BT",
        "tMAXt",
        "BWn",
        "MAXpressure",
        "BAte",
        "MAXtion",
        "tMAXpressure",
        "BAt",
        "Yn",
        "Ye",
        "Yx",
        "tMAXte",
        "BAtion",
        "MAXte",
        "tMAXtion",
        "BTx",
        "MAXt",
        "BTn",
        "BApressure",
        "tMINradius",
        "MINradius"
};

int main(int argc, char *argv[]) {
  int random_seed = lbann_default_random_seed;
  lbann_comm *comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();
  int np = comm->get_procs_in_world();

  options *opts = options::get();
  opts->init(argc, argv);

  if (argc == 1 || np != 1) {
    if (master) {
      std::cerr << "\nusage: " << argv[0] << " --filelist=<string> \n"
        "where: filelist contains a list of conduit filenames;\n"
        "\nPlease run with a single processor; you are running with "
        << np << " procs\n\n";
    }
    finalize(comm);
    return EXIT_SUCCESS;
  }

  const std::string input_fn = opts->get_string("filelist");
  std::vector<std::string> filenames;
  read_filelist(comm, input_fn, filenames);

  int nn = filenames.size() / 3;

  //=======================================================================
  // notes: each test gets a different set of conduit files to alleviate
  //        caching effects for the second test; since all files/samples
  //        contain the same amount of data, this should be fair. Note that
  //        the "test_total" outputs whould differ

  for (int i = 0; i < 3; i++) {
    std::cout << "========== Starting test round " << i << " ==========" << std::endl;
    // 1st test: load each image from the sample root
    test_conduit(0, nn, filenames);

    // 2nd test: load outputs/images into a node, then load each image
    // from that node
    test_conduit_2(nn, nn*2, filenames);

    // 3nd test: load outputs/images into a node, then load each image
    // from that node
    test_conduit_3(nn*2, nn*3, filenames);
  }

  // sanity test: run both tests on the same data. Times should be faster
  //              due to cach effects. More important. the "test_total"
  //              outputs should be identical
  std::cout << "========== Starting sanity checks ==========" << std::endl;
  test_conduit(0, nn, filenames);
  test_conduit_2(0, nn, filenames);
  test_conduit_3(0, nn, filenames);


  //=======================================================================

  finalize(comm);
  return EXIT_SUCCESS;
}

// load each image from the sample base
void test_conduit(int from, int to, std::vector<std::string> filenames) {
  double tm1 = get_time();
  double total = 0;
  double n1 = 0;
  double n2 = 0;
  double n3 = 0;
  double n4 = 0;
  double tm3;
  conduit::Node leaf;
  for (int j=from; j<to; j++) {
    std::cerr << "loading: " << filenames[j] << "\n";
    hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( filenames[j].c_str() );
    std::vector<std::string> cnames;
    conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
    for (size_t i=0; i<cnames.size(); i++) {
      for (size_t k=0; k<emi_v.size(); k++) {
        const std::string key = cnames[i] + "/outputs/images/" + emi_v[k] + "/emi";
        tm3 = get_time();
        conduit::relay::io::hdf5_read(hdf5_file_hnd, key, leaf);
        n1 += (get_time() - tm3);
        tm3 = get_time();
        conduit::float32_array emi = leaf.value();
        n2 += (get_time() - tm3);
        const size_t image_size = emi.number_of_elements();
        for (size_t x=0; x<image_size; x++) {
          total += emi[x];
        }
      }
      for (size_t k=0; k<jag_scalar_keys.size(); k++) {
        const std::string key = cnames[i] + "/outputs/scalars/" + jag_scalar_keys[k];
        tm3 = get_time();
        conduit::relay::io::hdf5_read(hdf5_file_hnd, key, leaf);
        n3 += (get_time() - tm3);
        tm3 = get_time();
        const double val = leaf.to_value();
        n4 += (get_time() - tm3);
        total += sizeof(val);
      }
    }
  }
  double tm2 = get_time();
  std::cerr << "baseline hdf5_read per field; time: " << tm2-tm1 << "  test_total: " << total << "\n";
  std::cerr << "  time to load individual images from samples: " << n1 << "\n";
  std::cerr << "  time to access images values:        " << n2 << "\n";
  std::cerr << "  time to load individual scalars from samples: " << n3 << "\n";
  std::cerr << "  time to access scalar values:        " << n4 << "\n\n";
}

// load each outputs/images for each sample, then load each image from that node
void test_conduit_2(int from, int to, std::vector<std::string> filenames) {
  double tm1 = get_time();
  double total = 0;
  double n1 = 0;
  double n2 = 0;
  double n3 = 0;
  double n4 = 0;
  double tm3;
  conduit::Node inode;
  for (int j=from; j<to; j++) {
    std::cerr << "loading: " << filenames[j] << "\n";
    hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( filenames[j].c_str() );
    std::vector<std::string> cnames;
    conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
    for (size_t i=0; i<cnames.size(); i++) {
      tm3 = get_time();
      const std::string key2 = cnames[i] + "/outputs/scalars/";
      conduit::relay::io::hdf5_read(hdf5_file_hnd, key2, inode);
      n4 += (get_time() - tm3);
      const std::string key = cnames[i] + "/outputs/images/";
      tm3 = get_time();
      conduit::relay::io::hdf5_read(hdf5_file_hnd, key, inode);
      n1 += (get_time() - tm3);
      for (size_t k=0; k<emi_v.size(); k++) {
        tm3 = get_time();
        const conduit::Node& leaf = inode[emi_v[k] + "/emi"];
        n2 += (get_time() - tm3);
        tm3 = get_time();
        conduit::float32_array emi = leaf.value();
        n3 += (get_time() - tm3);
        const size_t image_size = emi.number_of_elements();
        for (size_t x=0; x<image_size; x++) {
          total += emi[x];
        }
      }
    }
  }
  double tm2 = get_time();
  std::cerr << "multi-conduit node access; time: " << tm2-tm1 << "  test_total: " << total << "\n";
  std::cerr << "  time to load block of images from samples: " << n1 << "\n";
  std::cerr << "  time to access images node:        " << n2 << "\n";
  std::cerr << "  time to access images value:        " << n3 << "\n";
  std::cerr << "  time to load the scalar samples: " << n4 << "\n\n";
}

// load each outputs/images for each sample, then load each image from that node
void test_conduit_3(int from, int to, std::vector<std::string> filenames) {
  double tm1 = get_time();
  double total = 0;
  double n1 = 0;
  double n2 = 0;
  double tm3;
  conduit::Node inode;
  for (int j=from; j<to; j++) {
    std::cerr << "loading: " << filenames[j] << "\n";

    hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( filenames[j].c_str() );
    std::vector<std::string> cnames;
    conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
    for (size_t i=0; i<cnames.size(); i++) {

      tm3 = get_time();
      const std::string key = cnames[i] + "/outputs/";
      conduit::relay::io::hdf5_read(hdf5_file_hnd, key, inode);
      n1 += (get_time() - tm3);
      // std::cout << "total bytes: " << inode.total_strided_bytes() << std::endl;
      // conduit::Node ninfo;
      // inode.info(ninfo);
      // ninfo.print();

      for (size_t k=0; k<emi_v.size(); k++) {

        tm3 = get_time();
        const conduit::float32_array &emi = inode["/images/" + emi_v[k] + "/emi"].as_float32_array();
        //conduit::float32_array emi = inode["/outputs/images/" + emi_v[k] + "/emi"].value();
        n2 += (get_time() - tm3);

        const size_t image_size = emi.number_of_elements();
        for (size_t x=0; x<image_size; x++) {
          total += emi[x];
        }
      }
    }
  }
  double tm2 = get_time();
  std::cerr << "direct access via operator[]; time: " << tm2-tm1 << "  test_total: " << total << "\n";
  std::cerr << "  time to load entire (images) sample: " << n1 << "\n";
  std::cerr << "  time to access image values: " << n2 << "\n";
}
