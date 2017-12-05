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
// prototext .hpp .cpp 
////////////////////////////////////////////////////////////////////////////////

#include "lbann/utils/protobuf_utils.hpp"
#include "lbann/proto/proto_common.hpp"

/** 
 * all methods in protobuf_utils are static
 */

namespace lbann {

void protobuf_utils::parse_prototext_filenames_from_command_line(
               bool master,
               int argc,
               char **argv,
               std::vector<prototext_fn_triple> &names) {
  std::vector<std::string> models;
  std::vector<std::string> optimizers;
  std::vector<std::string> readers;
  for (int k=1; k<argc; k++) {
    std::string s(argv[k]);
    size_t equal_sign = s.find("=");
    if (equal_sign == std::string::npos) {
      std::cerr << "badly formed cmd line param; missing '=': " << s << std::endl;
      exit(1);
    }
    if (s[0] != '-' or s[1] != '-') {
      std::cerr << "badly formed cmd line param; must begin with '--': " << s << std::endl;
      exit(1);
    }
    if (s.find(',') != std::string::npos) {
      std::stringstream err;
      err << __FILE__ << __LINE__ << " :: "
          << " badly formed param; contains ','; " << s << "\n"
          << "possibly you left out '{' or '}' or both ??\n";
      throw lbann_exception(err.str());    
    }
    std::string which = s.substr(2, equal_sign-2);
    std::string fn = s.substr(equal_sign+1);
    if (which == "loadme") {
      models.push_back(fn);
    }
    if (which == "model") {
      models.push_back(fn);
    }
    if (which == "reader") {
      readers.push_back(fn);
    }
    if (which == "optimizer") {
      optimizers.push_back(fn);
    }
  }

  size_t n = models.size();
  if (! (optimizers.size() == 1 || optimizers.size() == n)) {
    std::stringstream err;   
    err << __FILE__ << " " << __LINE__ << " :: "
        << " you specified " << n << " model filenames, and " << optimizers.size()
        << " optimizer filenames; you must specify either one or "<< n
        << " optimizer filenames";
    throw lbann_exception(err.str());    
  }
  if (! (readers.size() == 1 || readers.size() == n)) {
    std::stringstream err;   
    err << __FILE__ << " " << __LINE__ << " :: "
        << " you specified " << n << " model filenames, and " << readers.size()
        << " reader filenames; you must specify either one or "<< n
        << " reader filenames";
    throw lbann_exception(err.str());    
  }

  names.clear();
  for (size_t i=0; i<models.size(); i++) {
    prototext_fn_triple t;
    t.model = models[i];
    if (readers.size() == 1) {
      t.reader = readers[0];  
    } else {
      t.reader = readers[i];  
    }
    if (optimizers.size() == 1) {
      t.optimizer = optimizers[0];  
    } else {
      t.optimizer = optimizers[i];  
    }
    names.push_back(t);
  }
}


void protobuf_utils::read_in_prototext_files(
                bool master,
                std::vector<prototext_fn_triple> &names,
                std::vector<lbann_data::LbannPB*> &models_out) {
  models_out.clear();
  for (auto t : names) {
    lbann_data::LbannPB *pb = new lbann_data::LbannPB;
    if (t.model != "none")
      read_prototext_file(t.model.c_str(), *pb, master);
    if (t.reader != "none") {
      lbann_data::LbannPB p;
      read_prototext_file(t.reader.c_str(), p, master);
      pb->MergeFrom(p);
    }
    if (t.optimizer != "none") {
      lbann_data::LbannPB p;
      read_prototext_file(t.optimizer.c_str(), p, master);
      pb->MergeFrom(p);
    }
    models_out.push_back(pb);
  }
}

void protobuf_utils::load_prototext(
                const bool master,
                const int argc, 
                char **argv,
                std::vector<lbann_data::LbannPB *> &models_out) {
    std::vector<prototext_fn_triple> names;
    parse_prototext_filenames_from_command_line(master, argc, argv, names);
    read_in_prototext_files(master, names, models_out);
    if (models_out.size() == 0) {
      if (master) {
        std::stringstream err;
        err << __FILE__ << __LINE__ << " :: "
            << " failed to load any prototext files";
        throw lbann_exception(err.str());    
      }
    }
    //verify_prototext(master, models);
}

void protobuf_utils::verify_prototext(bool master, const std::vector<lbann_data::LbannPB *> &models) {
  if (master) {
    std::cout << "protobuf_utils::verify_prototext; starting verify for " << models.size() << " models\n";
  }
  for (auto t : models) {
    const lbann_data::DataReader &d_reader = t->data_reader();
    int num_readers = d_reader.reader_size();
    if (master) std::cerr << "num readers: " << num_readers << " has_reader? " << t->has_data_reader() << std::endl;
    if (! t->has_optimizer()) {
      if (master) {
        std::stringstream err;
        err << __FILE__ << __LINE__ << " :: "
            << " loaded model is missing an Optimizer";
        throw lbann_exception(err.str());    
      }
    }
    //const lbann_data::Optimizer &opt_params = t.optimizer();
    //int num_optimizers = 0;

  }
}


}  // namespace lbann
