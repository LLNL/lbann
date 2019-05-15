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

#ifdef LBANN_HAS_CONDUIT

#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "lbann/lbann.hpp"
#include <time.h>
#include "lbann/utils/jag_utils.hpp"

using namespace lbann;

void get_input_names(std::unordered_set<std::string> &s);
void get_scalar_names(std::unordered_set<std::string> &s);
void get_image_names(std::unordered_set<std::string> &s);
//==========================================================================
int main(int argc, char *argv[]) {
  int random_seed = lbann_default_random_seed;
  world_comm_ptr comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();
  const int np = comm->get_procs_in_world();

  if (np > 1) {
    if (master) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: please run with a single processor\n");
    }
  }

  options *opts = options::get();
  opts->init(argc, argv);

  // sanity check invocation
  if (!opts->has_string("filename")) {
    if (master) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: usage: " + argv[0] + " --filename=<string>\ne.g: --filename=/p/lscratchh/brainusr/datasets/conduit_test/from_100M.bundle");
    }
  }

    const std::string filename = opts->get_string("filename");

    // get lists of inputs and scalars to read from file
    std::unordered_set<std::string> input_names;
    std::unordered_set<std::string> scalar_names;
    std::unordered_set<std::string> image_names;
    get_input_names(input_names);
    get_scalar_names(scalar_names);
    get_image_names(image_names);

    hid_t hdf5_file_hnd;
    std::string key;
    conduit::Node n_ok;
    conduit::Node tmp;
    std::cerr << "opening for read: " << filename << "\n";
    hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( filename.c_str() );

    std::vector<std::string> cnames;
    std::cerr << "calling: hdf5_group_list_child_names\n";
    conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
    std::cerr << "file contains " << cnames.size() << " samples\n";

    for (size_t i=0; i<cnames.size(); i++) {

      key = "/" + cnames[i] + "/performance/success";
      std::cerr << "calling: hdf5_read for key: " << key << "\n";
      conduit::relay::io::hdf5_read(hdf5_file_hnd, key, n_ok);

      int success = n_ok.to_int64();
      if (success == 1) {
        for (auto t : input_names) {
            key = cnames[i] + "/inputs/" + t;
            std::cerr << "calling: hdf5_read for key: " << key << "\n";
            conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
        }

        for (auto t : scalar_names) {
            key = cnames[i] + "/outputs/scalars/" + t;
            std::cerr << "calling: hdf5_read for key: " << key << "\n";
            conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
        }

        for (auto t : image_names) {
            key = cnames[i] + "/outputs/images/" + t;
            std::cerr << "calling: hdf5_read for key: " << key << "\n";
            conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
        }
      }
    }

  return 0;
}

void get_input_names(std::unordered_set<std::string> &s) {
  s.insert("shape_model_initial_modes:(4,3)");
  s.insert("betti_prl15_trans_u");
  s.insert("betti_prl15_trans_v");
  s.insert("shape_model_initial_modes:(2,1)");
  s.insert("shape_model_initial_modes:(1,0)");
}

void get_scalar_names(std::unordered_set<std::string> &s) {
  s.insert("BWx");
  s.insert("BT");
  s.insert("tMAXt");
  s.insert("BWn");
  s.insert("MAXpressure");
  s.insert("BAte");
  s.insert("MAXtion");
  s.insert("tMAXpressure");
  s.insert("BAt");
  s.insert("Yn");
  s.insert("Ye");
  s.insert("Yx");
  s.insert("tMAXte");
  s.insert("BAtion");
  s.insert("MAXte");
  s.insert("tMAXtion");
  s.insert("BTx");
  s.insert("MAXt");
  s.insert("BTn");
  s.insert("BApressure");
  s.insert("tMINradius");
  s.insert("MINradius");
}

void get_image_names(std::unordered_set<std::string> &s) {
  s.insert("(0.0, 0.0)//0.0/emi");
  s.insert("(90.0, 0.0)//0.0/emi");
  s.insert("(90.0, 78.0)//0.0/emi");
}

#endif //#ifdef LBANN_HAS_CONDUIT
