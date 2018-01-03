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
// lbann_callback_save_model .hpp .cpp - Callbacks to save models, currently as binary proto 
////////////////////////////////////////////////////////////////////////////////

#include <string>
#include "lbann/callbacks/callback_save_model.hpp"
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <fstream>

namespace lbann {


//@todo change to on training end
void lbann_callback_save_model::on_epoch_end(model *m) {
  lbann_data::Model model_param;
  m->write_proto(&model_param);
  std::string filename = m_dir + "." + m_extension;
  //@todo flag to save as either binary or text
  write_proto_text(model_param,filename);
}

void lbann_callback_save_model::write_proto_binary(const lbann_data::Model& proto,
                                                   const std::string filename) {
  std::fstream output(filename.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
  proto.SerializeToOstream(&output);
}

void lbann_callback_save_model::write_proto_text(const lbann_data::Model& proto,
                                                 const std::string filename) {

  int fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  auto output = new google::protobuf::io::FileOutputStream(fd);
  google::protobuf::TextFormat::Print(proto, output);
  delete output;
  close(fd);
}

}  // namespace lbann
