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

  if (!(opts->has_string("filelist") && opts->has_int("jag"))) {
    LBANN_ERROR("usage: test_speed_hydra_ --filelist=<string> --jag=<0|1>");
  }

  if (opts->get_int("jag")) {
    test_jag(opts->get_string("filelist"));
  } else {
    test_hydra(opts->get_string("filelist"));
  }  
  return EXIT_SUCCESS;
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

vector<string> get_input_names_jag() {
  vector<string> f;
  f.push_back("shape_model_initial_modes:(4,3)");
  f.push_back("betti_prl15_trans_u");
  f.push_back("betti_prl15_trans_v");
  f.push_back("shape_model_initial_modes:(2,1)");
  f.push_back("shape_model_initial_modes:(1,0)");
  return f;
}

vector<string> get_scalar_names_jag() {
  vector<string> f;
  f.push_back("BWx");
  f.push_back("BT");
  f.push_back("tMAXt");
  f.push_back("BWn");
  f.push_back("MAXpressure");
  f.push_back("BAte");
  f.push_back("MAXtion");
  f.push_back("tMAXpressure");
  f.push_back("BAt");
  f.push_back("Yn");
  f.push_back("Ye");
  f.push_back("Yx");
  f.push_back("tMAXte");
  f.push_back("BAtion");
  f.push_back("MAXte");
  f.push_back("tMAXtion");
  f.push_back("BTx");
  f.push_back("MAXt");
  f.push_back("BTn");
  f.push_back("BApressure");
  f.push_back("tMINradius");
  f.push_back("MINradius");
  return f;
}

vector<string> get_image_names_jag() {
  vector<string> f;
  f.push_back("(0.0, 0.0)/0.0/emi");
  f.push_back("(90.0, 0.0)/0.0/emi");
  f.push_back("(90.0, 78.0)/0.0/emi");
  return f;
}

void test_hydra(string filename) {
    double tm1 = get_time();
    hid_t hdf5_file_hnd;
    std::string key;
    conduit::Node n_ok;
    conduit::Node tmp;

    vector<string> input_names = get_input_names_hydra();
    vector<string> scalar_names = get_scalar_names_hydra();
    vector<string> image_names = get_image_names_hydra();

    int num_samples = 0;
    int num_files = 0;
    double total = 0;
    double bytes = 0;
    ifstream in(filename.c_str());
    long sample_size = 0;
    while (!in.eof()) {
      getline(in, filename);
      if (filename.size() < 2) {
        continue;
      }
      ++num_files;

      try {
        hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( filename.c_str() );
      } catch (...) {
        LBANN_ERROR("failed to open " + filename + " for reading");
      }
      cout << "reading: " << filename << endl;

      std::vector<std::string> cnames;
      try {
        conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
      } catch (...) {
        LBANN_ERROR("exception hdf5_group_list_child_names; " + filename);
      }
      cout << "samples per file: " << cnames.size() << endl;

      for (size_t i=0; i<cnames.size(); i++) {
        key = "/" + cnames[i] + "/performance/success";
        try {
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key, n_ok);
        } catch (...) {
          cout << "failed to read success flag for file: " + filename + " and key: " + key << endl;
          continue;
        }

        int success = n_ok.to_int64();
        if (success == 1) {
          try {

            sample_size = 0;
            sample_size += sizeof(double) + input_names.size();
            for (size_t h=0; h<input_names.size(); h++) {
              key = cnames[i] + "/inputs/" + input_names[h];
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
              double v = tmp.value();
              total += v;
              bytes += sizeof(double);
            }  

            sample_size += sizeof(double) + scalar_names.size();
            for (size_t h=0; h<scalar_names.size(); h++) {
              key = cnames[i] + "/scalars/" + scalar_names[h];
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
              double v = tmp.value();
              total += v;
              bytes += sizeof(double);
            }  

            for (size_t h=0; h<image_names.size(); h++) {
              key = cnames[i] + "/images/" + image_names[h];
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
              conduit::float64_array emi = tmp.value();
              const size_t image_size = emi.number_of_elements();
              if (image_size != 3*3*64*64) {
                LBANN_ERROR("image_size != 3*3*64*64");
              }
              for (int g=0; g<3*3*64*64; g++) {
                total += emi[g];
                bytes += sizeof(double);
                sample_size += sizeof(double);
              }
            }

            ++num_samples;
            if (num_samples >= MAX_SAMPLES) {
              goto FINISHED;
            }
          } catch (...) {
            LBANN_ERROR("error reading " + key + " from file " + filename);
          }
        }
      }
    }

FINISHED:

    double tm2 = get_time();
    cout << "========================================================\n"
         << "hydra test:\n";
    cout << "bytes per sample: " << sample_size << endl;
    cout << "time: " << tm2 - tm1 << " num samples: " << num_samples << " num files: " << num_files << "\n"
         << "num inputs: " << input_names.size() << " scalars: " << scalar_names.size() << endl;
    cout << "num bytes: " << bytes << " time to read 1M bytes: " << (tm2 - tm1)/(bytes/1000000) << endl;

}

void test_jag(string filename) {
cout << "starting test_jag; filename: " << filename << endl;
    double tm1 = get_time();
    hid_t hdf5_file_hnd;
    std::string key;
    conduit::Node n_ok;
    conduit::Node tmp;

    vector<string> input_names = get_input_names_jag();
    vector<string> scalar_names = get_scalar_names_jag();
    vector<string> image_names = get_image_names_jag();

    int num_samples = 0;
    int num_files = 0;
    double total = 0;
    double bytes = 0;
    ifstream in(filename.c_str());
    if (!in) {
      LBANN_ERROR("failed to open " + filename + " for reading\n");
    }
    long sample_size = 0;
    int bad_samples = 0;
    while (!in.eof()) {
      getline(in, filename);
      if (filename.size() < 2) {
        continue;
      }
      ++num_files;
      cout << "reading: " << filename << endl;

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
      cout << "samples per file: " << cnames.size() << " num samples: " << num_samples << endl;

      for (size_t i=0; i<cnames.size(); i++) {
        key = "/" + cnames[i] + "/performance/success";
        try {
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key, n_ok);
        } catch (...) {
          cout << "failed to read success flag for file: " + filename + " and key: " + key << endl;
          continue;
        }

        int success = n_ok.to_int64();
        if (success == 1) {
          try {
            sample_size = 0;
            sample_size += sizeof(double)*input_names.size();
            for (size_t h=0; h<input_names.size(); h++) {
              key = cnames[i] + "/inputs/" + input_names[h];
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
              double v = tmp.value();
              total += v;
              bytes += sizeof(double);
            }  

            sample_size += sizeof(double)*scalar_names.size();
            for (size_t h=0; h<scalar_names.size(); h++) {
              key = cnames[i] + "/outputs/scalars/" + scalar_names[h];
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
              double v = tmp.value();
              total += v;
              bytes += sizeof(double);
            }  

            for (size_t h=0; h<image_names.size(); h++) {
              key = cnames[i] + "/outputs/images/" + image_names[h];
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
              conduit::float32_array emi = tmp.value();
              const size_t image_size = emi.number_of_elements();
              for (size_t g=0; g<image_size; g++) {
                total += emi[g];
                bytes += sizeof(double);
                sample_size += sizeof(double);
              }
            }

            ++num_samples;
            if (num_samples >= MAX_SAMPLES) {
              goto FINISHED;
            }

          } catch (...) {
            conduit::Node node;
            conduit::relay::io::load(filename, "hdf5", node);
            const conduit::Schema *s = node.schema_ptr();
            cerr << "KEY: " << key << endl;
            s->print();
            LBANN_ERROR("error reading " + key + " from file " + filename);
          }
        } else {
          ++bad_samples;
        }
      }
    }

FINISHED: 

    double tm2 = get_time();
    cout << "========================================================\n"
         << "jag test:\n";
    cout << "bytes per sample: " << sample_size << endl;
    cout << "num bad samples: " << bad_samples << endl;
    cout << "time: " << tm2 - tm1 << " num samples: " << num_samples << " num files: " << num_files << "\n"
         << "num inputs: " << input_names.size() << " scalars: " << scalar_names.size() << endl;
    cout << "num bytes: " << bytes << " time to read 1M bytes: " << (tm2 - tm1)/(bytes/1000000) << endl;

}
