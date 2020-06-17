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
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "lbann/lbann.hpp"
#include "lbann/utils/jag_utils.hpp"
#include <time.h>
#include <cfloat>

using namespace lbann;
using namespace std;

vector<string> get_input_names_jag();
vector<string> get_scalar_names_jag();
vector<string> get_image_names_jag();
vector<string> get_input_names_hydra();
vector<string> get_scalar_names_hydra();
vector<string> get_image_names_hydra();
void test_hydra(string filename);
void test_jag(string filename);

//==========================================================================
#define MAX_SAMPLES 10000

int main(int argc, char *argv[]) {
  int random_seed = lbann_default_random_seed;
  world_comm_ptr comm = initialize(argc, argv, random_seed);

  options *opts = options::get();
  opts->init(argc, argv);

  if (!(opts->has_string("filelist") && opts->has_string("output_dir") && opts->has_string("format"))) {
    LBANN_ERROR("usage: test_speed_hydra_ --filelist=<string> --output_dir=<string> --format=<hdf5|conduit_bin>");
  }

  string filelist = opts->get_string("filelist");
  string format = opts->get_string("format");
  string output_dir = opts->get_string("output_dir");
  stringstream s;
  s << "mkdir -p " << output_dir;
  system(s.str().c_str());

    hid_t hdf5_file_hnd;
    std::string key;
    conduit::Node n_ok;
    conduit::Node tmp;

    vector<string> input_names = get_input_names_hydra();
    vector<string> scalar_names = get_scalar_names_hydra();
    vector<string> image_names = get_image_names_hydra();

    int num_samples = 0;
    int num_files = 0;
    ifstream in(filelist.c_str());
    int sample_id = 0;
    string filename;
    while (!in.eof()) {
      getline(in, filename);
      if (filename.size() < 2) {
        continue;
      }
      ++num_files;
      conduit::Node node;
      hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( filename.c_str() );
      cout << "reading: " << filename << endl;

      size_t k = filename.rfind("/");
      stringstream s2;
      s2 << output_dir << "/" << filename.substr(k+1);

      std::vector<std::string> cnames;
      conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
      cout << "samples per file: " << cnames.size() << endl;

      for (size_t i=0; i<cnames.size(); i++) {
        key = "/" + cnames[i] + "/performance/success";
        try {
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key, n_ok);
        } catch (...) {
          cout << "failed to read success flag for file: " + filename + " and key: " + key << "; if the key contains 'META' (for HYDRA data) this is OK\n";
          continue;
        }

        int success = n_ok.to_int64();
        if (success == 1) {
          conduit::Node node2;
          node2["/performance/success"] = 1;

          for (size_t h=0; h<input_names.size(); h++) {
            key = cnames[i] + "/inputs/" + input_names[h];
            tmp.reset();
            conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
            node2[ "/inputs/" + input_names[h]] = tmp;
          }  

          for (size_t h=0; h<scalar_names.size(); h++) {
            tmp.reset();
            key = cnames[i] + "/scalars/" + scalar_names[h];
            conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
            node2[ "/scalars/" + scalar_names[h]] = tmp;
          }  

          for (size_t h=0; h<image_names.size(); h++) {
            tmp.reset();
            key = cnames[i] + "/images/" + image_names[h];
            conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
            node2["/images/" + image_names[h]] = tmp;
          }

          ++num_samples;
          node[to_string(sample_id)] = node2;
          ++sample_id;
        }
      }

      cout << "calling save: " << s2.str() << " format: " << format << endl;
      conduit::relay::io::save(node, s2.str(), format);
    }
    in.close();

}

vector<string> get_input_names_hydra() {
  vector<string> f;
  f.push_back("p_preheat");
  f.push_back("sc_peak");
  f.push_back("t_3rd");
  f.push_back("t_end");
  return f;
}

vector<string> get_scalar_names_hydra() {
  vector<string> f;
  f.push_back("avg_rhor");
  f.push_back("peak_eprod");
  f.push_back("peak_tion_bw_DT");
  f.push_back("bt_tion_bw_DT");
  f.push_back("avg_tion_bw_DT");
  f.push_back("adiabat");
  f.push_back("bangt");
  f.push_back("burnwidth");
  f.push_back("bt_rhor");
  f.push_back("bt_eprodr");
  f.push_back("peak_eprodr");
  return f;
}

vector<string> get_image_names_hydra() {
  vector<string> f;
  f.push_back("(90,0)/bang/image/data");
  f.push_back("(0,0)/bang/image/data");
  return f;
}



