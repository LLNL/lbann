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
// lbann_callback_save_model .hpp .cpp - Callbacks to save a models description and weights
////////////////////////////////////////////////////////////////////////////////

#include <string>
#include "lbann/callbacks/callback_save_model.hpp"
#include "lbann/callbacks/callback_checkpoint.hpp" // Reuse the checkpoint naming scheme
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <fstream>
#include <unistd.h>

namespace lbann {


/// Save the model's prototext and weights
void lbann_callback_save_model::on_train_end(model *m) {
  save_model(m);
}

void lbann_callback_save_model::on_test_end(model *m) {
  save_model(m);
}

void lbann_callback_save_model::write_proto_binary(const lbann_data::Model& proto,
                                                   const std::string filename) {
  std::fstream output(filename.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
  proto.SerializeToOstream(&output);
}

void lbann_callback_save_model::write_proto_text(const lbann_data::Model& proto,
                                                 const std::string filename) {
  int fd = openwrite(filename.c_str());
  auto output = new google::protobuf::io::FileOutputStream(fd);
  google::protobuf::TextFormat::Print(proto, output);
  delete output;
  close(fd);
}

bool lbann_callback_save_model::save_model(model *m) {
  lbann_data::Model model_param;

  p.set_cb_type(callback_type::inference);
  save_model_weights(m);
  p.set_cb_type(callback_type::invalid);

#if 0 /// @todo BVE FIXME this method for writing out the prototext does not seem to work
  m->write_proto(&model_param);
  std::string filename = m->get_name() + "." + m_extension;
  std::string fullpath = m_dir + "/" + filename;
  //@todo flag to save as either binary or text
  if(m_extension == "bin") write_proto_binary(model_param,fullpath);
  else write_proto_text(model_param,fullpath);
#endif

  return true;
}

// Save model weights
bool lbann_callback_save_model::save_model_weights(model *m) {
  // if the checkpoint directory is not defined, bail
  if (m_dir.length() == 0) {
    return false;
  }
  // time how long this takes
  // read current epoch and step counters from model
  El::Timer timer;
  lbann_comm *comm = m->get_comm();
  comm->model_barrier();
  // let user know we're saving the weights
  int epoch = m->get_cur_epoch();
  int step = m->get_cur_step();
  if (comm->am_model_master()) {
    timer.Start();
    printf("[%s.%d] Saving model weights: epoch %d step %d ...\n", m->get_name().c_str(), comm->get_model_rank(), epoch, step);
    fflush(stdout);
  }

  // Shared checkpoint, logic identical to Distributed.i
  makedir(m_dir.c_str());
  std::string epochdir = get_shared_checkpoint_dirname(m, m_dir.c_str(), epoch, step);
  if (comm->am_model_master()) {
    p.open_checkpoint(epochdir.c_str());
  }
  // Need to give other ranks knowledge of checkpoint dir for writing of rank specific rng state
  comm->model_broadcast(0, &(p.m_checkpoint_dir[0]), sizeof(p.m_checkpoint_dir));
  m->save_weights(p);
  // close our checkpoint
  p.close_checkpoint();
  if (comm->am_model_master()) {
    std::string latest_file = get_last_shared_checkpoint_filename(m, m_dir.c_str());
    write_latest(latest_file, epoch, step);
  }

  uint64_t bytes_count = p.get_bytes();

  if (comm->am_model_master()) {
    EvalType secs = timer.Stop();
    EvalType bw = 0;
    if (secs > 0.0) {
      bw = EvalType(bytes_count) / (secs * 1024.0 * 1024.0);
    }
    printf("[%s.%d] Saving model weights complete: Epoch=%d Step=%d (%f secs, %llu bytes, %f MB/sec)\n",
           m->get_name().c_str(), comm->get_model_rank(), epoch, step, secs, (unsigned long long) bytes_count, bw);
    fflush(stdout);
  }
  p.reset_bytes();
  return true;
}

}  // namespace lbann
