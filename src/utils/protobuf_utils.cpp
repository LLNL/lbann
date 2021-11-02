////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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

#include <lbann.pb.h> // Actually use LbannPB here

/**
 * all methods in protobuf_utils are static
 */

namespace lbann {
namespace protobuf_utils {

std::vector<prototext_fn_triple>
parse_prototext_filenames_from_command_line(
               const bool master,
               const int trainer_rank) {
  auto& arg_parser = global_argument_parser();
  std::vector<std::string> models;
  std::vector<std::string> optimizers;
  std::vector<std::string> readers;
  std::vector<std::string> data_set_metadata;
  bool single_file_load = false;

  std::string params[] = {LBANN_OPTION_PROTOTEXT, LBANN_OPTION_MODEL, LBANN_OPTION_READER, LBANN_OPTION_METADATA, LBANN_OPTION_OPTIMIZER};
  for (auto& which : params) {
    std::string fn = arg_parser.get<std::string>(which);
    if (fn != "") {
      size_t t_pos = fn.find("trainer");
      if(t_pos != std::string::npos) {
        // append appropriate trainer id to prototext filename
        std::string fname =
          fn.substr(0, t_pos + 7) + std::to_string(trainer_rank);
        fn = fname;
      }
      if (which == LBANN_OPTION_PROTOTEXT) {
        models.push_back(fn);
        single_file_load = true;
      }
      if (which == LBANN_OPTION_MODEL) {
        models.push_back(fn);
      }
      if (which == LBANN_OPTION_READER) {
        readers.push_back(fn);
      }
      if (which == LBANN_OPTION_METADATA) {
        data_set_metadata.push_back(fn);
      }
      if (which == LBANN_OPTION_OPTIMIZER) {
        optimizers.push_back(fn);
      }
    }
  }

  if(!single_file_load) {
    size_t n = models.size();
    if (! (optimizers.size() == 1 || optimizers.size() == n)) {
      LBANN_ERROR(
        "you specified ",
        n,
        " model filenames, and ",
        optimizers.size(),
        " optimizer filenames; you must specify 1 optimizer filenames");
    }
    if (! (readers.size() == 1 || readers.size() == n)) {
      LBANN_ERROR("you specified ",
                  n,
                  " model filenames, and ",
                  readers.size(),
                  " reader filenames; you must specify 1 reader filenames");
    }
    if (! (data_set_metadata.size() == 0 || data_set_metadata.size() == 1 || data_set_metadata.size() == n)) {
      LBANN_ERROR("you specified ",
                  n,
                  " model filenames, and ",
                  data_set_metadata.size(),
                  " data set metadata filenames; you must specify 1 data set "
                  "metadata filenames");
    }
  }

  std::vector<prototext_fn_triple> names;
  for (size_t i=0; i<models.size(); i++) {
    prototext_fn_triple t;
    t.model = models[i];
    if (readers.size() == 0) {
      t.reader = "none";
    }else if (readers.size() == 1) {
      t.reader = readers[0];
    } else {
      t.reader = readers[i];
    }
    if (data_set_metadata.size() == 0) {
      t.data_set_metadata = "none";
    }else if (data_set_metadata.size() == 1) {
      t.data_set_metadata = data_set_metadata[0];
    } else {
      t.data_set_metadata = data_set_metadata[i];
    }
    if (optimizers.size() == 0) {
      t.optimizer = "none";
    }else if (optimizers.size() == 1) {
      t.optimizer = optimizers[0];
    } else {
      t.optimizer = optimizers[i];
    }
    names.push_back(t);
  }
  return names;
}

std::vector<std::unique_ptr<lbann_data::LbannPB>>
read_in_prototext_files(
  const bool master,
  const std::vector<prototext_fn_triple> &names)
{
  std::vector<std::unique_ptr<lbann_data::LbannPB>> models_out;
  for (auto const& t : names) {
    auto pb = make_unique<lbann_data::LbannPB>();
    if (t.model != "none")
      read_prototext_file(t.model, *pb, master);
    if (t.reader != "none") {
      lbann_data::LbannPB p;
      read_prototext_file(t.reader, p, master);
      pb->MergeFrom(p);
    }
    if (t.data_set_metadata != "none") {
      lbann_data::LbannPB p;
      read_prototext_file(t.data_set_metadata, p, master);
      pb->MergeFrom(p);
    }
    if (t.optimizer != "none") {
      lbann_data::LbannPB p;
      read_prototext_file(t.optimizer, p, master);
      pb->MergeFrom(p);
    }
    models_out.emplace_back(std::move(pb));
  }
  return models_out;
}

std::vector<std::unique_ptr<lbann_data::LbannPB>>
load_prototext(
  const bool master,
  const int trainer_rank)
{
  auto names =
    parse_prototext_filenames_from_command_line(master, trainer_rank);
  auto models_out = read_in_prototext_files(master, names);
  if (models_out.size() == 0 && master) {
    LBANN_ERROR("Failed to load any prototext files");
  }
  verify_prototext(master, models_out);
  return models_out;
}

void verify_prototext(
  const bool master,
  const std::vector<std::unique_ptr<lbann_data::LbannPB>> &models) {
  std::stringstream err;
  if (master) {
    std::cout << "protobuf_utils::verify_prototext; starting verify for " << models.size() << " models\n";
  }
  for (size_t j=0; j<models.size(); j++) {
    bool is_good = true;
    lbann_data::LbannPB *t = models[j].get();
    if (! t->has_data_reader()) {
      is_good = false;
      err << "model #" << j << " is missing data_reader\n";
    } else {
      if (t->data_reader().requires_data_set_metadata() && (! t->has_data_set_metadata())) {
        is_good = false;
        err << "model #" << j << " is missing metadata (cmd line flag: --metadata=<string>)\n";
      }
    }
    if (! t->has_model()) {
      is_good = false;
      err << "model #" << j << " is missing model\n";
    }
    if (! t->has_optimizer()) {
      is_good = false;
      err << "model #" << j << " is missing optimizer\n";
    }

    if (! is_good) {
      LBANN_ERROR("please check your command line and/or prototext files:\n", err.str());
    }
  }
}

}// namespace protobuf_utils
}// namespace lbann
