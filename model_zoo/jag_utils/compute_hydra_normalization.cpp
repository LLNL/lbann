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

vector<string> get_input_names();
vector<string> get_scalar_names();

//==========================================================================
int main(int argc, char *argv[]) {
  int random_seed = lbann_default_random_seed;
  world_comm_ptr comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();
  const int rank = comm->get_rank_in_world();

  //try {
    options *opts = options::get();
    opts->init(argc, argv);

    if (!(opts->has_string("filelist"))) {
      if (master) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: usage: " + argv[0] + " --filelist=<string>");
      }
    }

    //=======================================================================

    hid_t hdf5_file_hnd;
    std::string key;
    conduit::Node n_ok;
    conduit::Node tmp;

    int num_samples = 0;
    vector<string> input_names = get_input_names();
    size_t sz = input_names.size();
    std::vector<double> inputs_v_max(sz, DBL_MIN);
    std::vector<double> inputs_v_min(sz, DBL_MAX);
    std::vector<double> inputs_sum(sz, 0.0);

    ifstream in(opts->get_string("filelist").c_str());
    if (!in) {
      LBANN_ERROR("failed to open " + opts->get_string("filelist") + " for reading");
    }

    size_t hh = 0;
    string filename;
    while (!in.eof()) {
      getline(in, filename);
      if (filename.size() < 2) {
        continue;
      }
      hh += 1;
      if (hh % 10 == 0) std::cout << rank << " :: processed " << hh << " filenames\n";

      try {
        hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( filename.c_str() );
      } catch (...) {
        LBANN_ERROR("failed to open " + filename + " for reading");
      }

      std::vector<std::string> cnames;
      try {
        conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
      } catch (...) {
        LBANN_ERROR("exception hdf5_group_list_child_names; " + filename);
      }

      for (size_t i=0; i<cnames.size(); i++) {
        key = "/" + cnames[i] + "/performance/success";
        try {
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key, n_ok);
        } catch (...) {
          cout << "exception reading success flag for file: " + filename + " and key: " + key << endl;
          continue;
        }

        int success = n_ok.to_int64();
        if (success == 1) {
          try {
            for (size_t h=0; h<input_names.size(); h++) {
              key = cnames[i] + "/inputs/" + input_names[h];
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
              double v = tmp.value();
              if (v < inputs_v_min[h]) inputs_v_min[h] = v;
              if (v > inputs_v_max[h]) inputs_v_max[h] = v;
              inputs_sum[h] += v;
            }  
          } catch (...) {
            LBANN_ERROR("error reading " + key + " from file " + filename);
          }
        }
        ++num_samples;
      }
    }

    for (size_t j=0; j<input_names.size(); j++) {
      cout << input_names[j] << " " << inputs_v_min[j] << " " << inputs_v_max[j] << " " << inputs_sum[j]/num_samples << endl;
    }


/*
  } catch (exception &e) {
    El::ReportException(e);
    return EXIT_FAILURE;
  } catch (std::exception const &e) {
    El::ReportException(e);
    return EXIT_FAILURE;
  }
*/

  // Clean up
  return EXIT_SUCCESS;
}

vector<string> get_input_names() {
  vector<string> f;
  f.push_back("p_preheat");
  f.push_back("sc_peak");
  f.push_back("t_3rd");
  f.push_back("t_end");
  return f;
}

vector<string> get_scalar_names() {
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
