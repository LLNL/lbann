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
#include "conduit/conduit_relay_io_handle.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "lbann/lbann.hpp"
#include "lbann/utils/jag_utils.hpp"

using namespace lbann;

int main(int argc, char *argv[]) {
  int random_seed = lbann_default_random_seed;
  world_comm_ptr comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();
  const int rank = comm->get_rank_in_world();
  const int np = comm->get_procs_in_world();

  // check that we're running with a single CPU
  if (np != 1) {
    LBANN_ERROR("apologies, this is a sequential code; please run with a single processor. Thanks for playing!");
  }

  std::stringstream err;
  options *opts = options::get();
  opts->init(argc, argv);

  // sanity check invocation
  if (!opts->has_string("filelist")) {
    if (master) {
      err << " :: usage: " << argv[0] << " --filelist=<string>\n"
          << "WARNING: this driver deletes the directory 'corrupt_jag_samples' if it exists "
          << "then creates a new directory with that name";
      LBANN_ERROR(err.str());
    }
  }

  // read list of conduit filenames
  std::vector<std::string> files;
  const std::string fn = opts->get_string("filelist");
  read_filelist(comm.get(), fn, files);

  int ee = system("rm -rf corrupt_jag_samples");
  ee = system("mkdir corrupt_jag_samples");
  if (ee) {
    LBANN_ERROR("system call: 'mkdir corrupt_jag_samples' failed");
  }

  std::ofstream out("corrupt_jag_samples/README.txt");
  if (! out) {
    LBANN_ERROR("failed to open corrupt_jag_samples/README.txt for reading");
  }
  out << "#This file contains information for the samples in the file: 'corrupt.bundle'\n";

  conduit::relay::io::IOHandle hndl;
  std::string key;
  conduit::Node node;
  conduit::Node output;
  for (size_t j=rank; j<files.size(); ++j) {
    std::cerr << "processing: " << j << " of " << files.size() << " files\n";

    // open the next conduit file
    try {
      hndl.open(files[j], "hdf5");
    } catch (...) {
      err << "failed to open: " << files[j];
      LBANN_ERROR(err.str());
    }

    // get list of samples in this file
    std::vector<std::string> cnames;
    try {
      hndl.list_child_names(cnames);
    } catch (const std::exception&) {
      err << "list_child_names failed for this file: " << files[j];
      LBANN_ERROR(err.str());
    }

    // loop over the samples in the current file
    for (size_t i=0; i<cnames.size(); i++) {
      try {
        hndl.read(cnames[i], node);
      } catch (...) {
        err << "exception reading from file: " + files[j]<< " this key: " << key;
        LBANN_ERROR(err.str());
      }

      if (i < 1) {
        output[cnames[i]] = node;
        out << cnames[i] << " no corruption\n";
      } else if (i == 1) {
        out << cnames[i] << " missing inputs\n";
        node.remove("inputs");
        output[cnames[i]] = node;
      } else if (i == 2) {
        out << cnames[i] << " missing outputs/scalars\n";
        node.remove("outputs/scalars");
        output[cnames[i]] = node;
      } else if (i == 3) {
        out << cnames[i] << " missing ouputs/images\n";
        node.remove("outputs/images");
        output[cnames[i]] = node;
      } else if (i == 4) {
        out << cnames[i] << " missing outputs\n";
        node.remove("outputs");
        output[cnames[i]] = node;
      } else if (i == 5) {
        out << cnames[i] << " missing outputs/images/(90.0, 0.0)/0.0/emi\n";
        node.remove("outputs/images/(90.0, 0.0)/0.0/emi");
        output[cnames[i]] = node;
      } else if (i == 6) {
        out << cnames[i] << " missing outputs/scalars/MAXpressure";
        node.remove("outputs/scalars/MAXpressure");
        output[cnames[i]] = node;
      } else {
        break;
      }
    }
  }
  std::stringstream output_fn;
  output_fn << "corrupt_jag_samples/corrupt.bundle";
  try {
    conduit::relay::io::save(output, output_fn.str(), "hdf5");
  } catch (...) {
    err << "failed to write " << output_fn.str();
    LBANN_ERROR(err.str());
  }

  out.close();
  std::cout << "\nMade directory 'corrupt_jag_samples/' and wrote files in that directory\n\n";
}
