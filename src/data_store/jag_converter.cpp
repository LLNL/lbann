////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#include "lbann/lbann.hpp"
#include "lbann/data_store/jag_io.hpp"
#include "lbann/utils/options.hpp"
#include <cstdint>

std::string usage("\n\nusage: jag_converter --mode=<convert|test|both>  --bundle=<filename> --dir=<directory for converted *.bundle>");

using namespace lbann;

void convert(std::string bundle_fn, std::string dir);
void test(std::string bundle_fn, std::string dir);

int main(int argc, char *argv[]) {

#ifndef LBANN_HAS_CONDUIT
  std::cerr << "ERROR: lbann was not compiled with conduit support\n"
               "(LBANN_HAS_CONDUIT was not defined at compile time)\n";
  exit(9);
#else

  lbann_comm *comm = initialize(argc, argv, 42);

  try {
    options *opts = options::get();
    opts->init(argc, argv);
  
    std::stringstream err;

    std::string bundle_fn;
    std::string convert_dir;
    std::string mode;

    if (!opts->has_string("mode")) {
      err << __FILE__ << " " << __LINE__ << " :: "
          << "you must pass the option: --mode=<string>,\n"
             "where <string> is \"convert\" or \"test\" or \"both\""
             << usage;
      throw lbann_exception(err.str());
    }
    mode = opts->get_string("mode");
    
    if (!opts->has_string("bundle")) {
      err << __FILE__ << " " << __LINE__ << " :: "
          << "you must pass the option: --bundle=<pathname>,\n"
             "which is the input filename"
          << usage;
      throw lbann_exception(err.str());
    }
    bundle_fn= opts->get_string("bundle");

    if (!opts->has_string("dir")) {
      err << __FILE__ << " " << __LINE__ << " :: "
          << "you must pass the option: --dir=<string>,\n"
             "which is the directory for the converted file"
          << usage;
      throw lbann_exception(err.str());
    }
    convert_dir = opts->get_string("dir");

    if (mode == "convert") {
      convert(bundle_fn, convert_dir);
    } else if (mode == "test") {
      test(bundle_fn, convert_dir);
    } else if (mode == "both") {
      convert(bundle_fn, convert_dir);
      test(bundle_fn, convert_dir);
    } else {
      err << __FILE__ << " " << __LINE__ << " :: "
          << "bad value for option: --mode=<string>;\n"
             "must be 'convert' or 'test' or 'both'"
          << usage;
      throw lbann_exception(err.str());
    }

  } catch (lbann_exception& e) {
    lbann_report_exception(e, comm);
  }  

#endif //ifdef LBANN_HAS_CONDUIT

  return 0;
}

#ifdef LBANN_HAS_CONDUIT
void convert(std::string bundle_fn, std::string dir) {
  jag_io io;
  io.convert(bundle_fn, dir);
}

void test(std::string bundle_fn, std::string dir) {
  std::cerr << "\nstarting test ...\n";
  std::cerr << "loading conduit node...\n";
  conduit::Node head;
  conduit::relay::io::load(bundle_fn, "hdf5", head);

  std::cerr << "calling jag.load("<<dir<<")\n";
  jag_io jag;
  jag.load(dir);
  size_t num_samples = jag.get_num_samples();
  size_t sample_id = num_samples > 1 ? 1 : 0;
  const std::vector<std::string> &keys = jag.get_keys();

  std::cerr << "using sample " << sample_id << " of " << num_samples << "\n";
  std::cerr << "num keys: " << keys.size() << "\n";

  size_t num_elts;
  size_t bytes_per_elt;
  size_t total_bytes;
  std::string type;
  std::vector<char> data;
  size_t pass = 0;
  size_t skipped = 0;
  size_t warnings = 0;
  size_t total = 0;

  //=========================================================================\n;
  // test #1: 
  //   loop over all keys; test that what we get from the jag_io is identical
  //   to what we get directly from the conduit node
  //
  //=========================================================================\n;
  for (auto key : keys) {
    ++total;

    // get data from jag_io
    jag.get_metadata(key, num_elts, bytes_per_elt, total_bytes, type);
    if (total_bytes == 0) {
      ++skipped;
      continue;
    }
    data.resize(total_bytes);
    std::string key2 = std::to_string(sample_id) + '/' + key;
    jag.get_data(key2, data.data(), total_bytes);

    // get data directly from conduit
    conduit::Node truth = head[key2];

    char *f2 = 0;
    if (type == "int64") {
      long *f = truth.as_int64_ptr();
      f2 = (char*)f;
    } else if (type == "float64") {
      double *f = truth.as_float64_ptr();
      f2 = (char*)f;
    } else if (type == "uint64") {
      uint64 *f = truth.as_uint64_ptr();
      f2 = (char*)f;
    } else if (type == "char8_str") {
      char *f = truth.as_char8_str();
      f2 = (char*)f;
    } else {
      std::cerr << "WARNING: unhandled type: " << type << "\n";
      ++warnings;
    }
    if (f2) {
      for (size_t i=0; i<total_bytes; i++) {
        if (data[i] != f2[i]) {
          std::cerr << "ERROR: data from jag_io doesn't match data from conuit Node\n"
                    << "key: " << key2 << "\n";
          exit(9);
        }
      }
      ++pass;
    }
  }
  size_t sanity = skipped + warnings + pass;
  std::cerr << "\n\n"
            << "total keys tested:  " << total << "\n"
            << "total keys skipped: " << skipped << " (due to zero length data)\n"
            << "total warnings:     " << warnings << " (handling a data type that's not yet supported)\n"
            << "number that passed: " << pass << "\n"
            << "sanity:             " << sanity << " (should be same as total keys tested)\n";

  //=========================================================================\n;
  // test #2: 
  //   test, for each key, that type, num elts, num_bytes identical
  //=========================================================================\n;
  for (auto key : keys) {
    jag.get_metadata(key, num_elts, bytes_per_elt, total_bytes, type);
    for (size_t j=0; j<num_samples; j++) {
      std::string key2 = std::to_string(sample_id) + '/' + key;
      conduit::Node truth = head[key2];
      conduit::DataType dType = truth.dtype();
      if (num_elts != (size_t)dType.number_of_elements()) {
        std::cerr << "ERROR 1!\n";
        exit(9);
      }
      if (bytes_per_elt != (size_t)dType.element_bytes()) {
        std::cerr << "ERROR 2!\n";
        exit(9);
      }
      if (type != dType.name()) {
        std::cerr << "ERROR 3!\n";
        exit(9);
      }
    }
  }
}
#endif //LBANN_HAS_CONDUIT
