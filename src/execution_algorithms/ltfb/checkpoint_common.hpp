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
////////////////////////////////////////////////////////////////////////////////
#ifndef LBANN_SRC_EXECUTION_ALGORITHMS_LTFB_CHECKPOINT_COMMON_HPP_INCLUDED
#define LBANN_SRC_EXECUTION_ALGORITHMS_LTFB_CHECKPOINT_COMMON_HPP_INCLUDED

#include "lbann/models/model.hpp"
#include "lbann/weights/data_type_weights_impl.hpp"

#include <unordered_set>

namespace lbann {
namespace ltfb {
namespace {

// Pack model to ship off
std::string pack(model const& m)
{
  std::ostringstream oss;
  {
    RootedBinaryOutputArchive ar(oss, m.get_comm()->get_trainer_grid());
    ar(m);
  }
  return oss.str();
}

// Send a string to the root of the destination trainer
void send_string(lbann_comm const& comm,
                 std::string const& str,
                 int destination_trainer)
{
  size_t size = str.length();
  comm.send(&size, 1, destination_trainer, /*rank=*/0);
  comm.send(reinterpret_cast<El::byte const*>(str.data()),
            size,
            destination_trainer,
            /*rank=*/0);
}

// Receive a string from the root of src_trainer
std::string recv_string(lbann_comm const& comm, int src_trainer)
{
  size_t size = 0;
  comm.recv(&size, 1, src_trainer);
  std::string buf;
  buf.resize(size);
  comm.recv(reinterpret_cast<El::byte*>(buf.data()), size, src_trainer);
  return buf;
}

// Unpack received model
void unpack(model& m, std::string const& str)
{
  std::istringstream iss(str);
  {
    RootedBinaryInputArchive ar(iss, m.get_comm()->get_trainer_grid());
    ar(m);
  }
}

} // namespace

inline static void restore_model_weights(
  model& m,
  std::unordered_map<std::string, std::unique_ptr<weights>>& restore_weights)
{
  // Restore weights that shouldn't be exchanged
  if (restore_weights.empty())
    return;

  // FIXME: Generalize this; enable ptr move??
  for (auto w : m.get_weights()) {
    if (restore_weights.count(w->get_name()) > 0) {
      using TensorDataType = DataType;
      using WeightsType = data_type_weights<TensorDataType>;
      dynamic_cast<WeightsType&>(*w) =
        dynamic_cast<WeightsType&>(*restore_weights[w->get_name()]);
    }
  }
}

inline static std::string sendrecv_string(lbann_comm const& c,
                                          std::string const& src,
                                          El::Int partner_trainer)
{
  if (!c.am_trainer_master())
    return "";

  // Exchange sizes
  size_t my_size = src.size();
  size_t other_size = src.max_size() + 1;
  c.sendrecv(&my_size,
             1,
             partner_trainer,
             0,
             &other_size,
             1,
             partner_trainer,
             0,
             El::SyncInfo<El::Device::CPU>{});

  // Exchange strings
  std::string tgt(other_size, '\0');

  auto const* send_buf = reinterpret_cast<El::byte const*>(src.data());
  auto* recv_buf = reinterpret_cast<El::byte*>(tgt.data());

  // Get the max blk size
  int constexpr max_blk_size_int = std::numeric_limits<int>::max();
  std::size_t constexpr max_blk_size_size_t = max_blk_size_int;

  while (my_size || other_size) {
    int const this_blk_send_size =
      (my_size > max_blk_size_size_t ? max_blk_size_int : my_size);
    int const this_blk_recv_size =
      (other_size > max_blk_size_size_t ? max_blk_size_int : other_size);

    c.sendrecv(send_buf,
               this_blk_send_size,
               partner_trainer,
               0,
               recv_buf,
               this_blk_recv_size,
               partner_trainer,
               0,
               El::SyncInfo<El::Device::CPU>{});

    send_buf += this_blk_send_size;
    recv_buf += this_blk_recv_size;
    my_size =
      (my_size > max_blk_size_size_t ? my_size - max_blk_size_size_t : 0);
    other_size =
      (other_size > max_blk_size_size_t ? other_size - max_blk_size_size_t : 0);
  }
  return tgt;
}

template <typename T>
static void exchange(lbann_comm const& c, T& object, El::Int partner_trainer)
{
  std::ostringstream oss;
  {
    RootedBinaryOutputArchive ar(oss, c.get_trainer_grid());
    c.trainer_barrier();
    ar(object);
  }
  c.trainer_barrier(); // I don't think this is necessary
  {
    std::istringstream iss{sendrecv_string(c, oss.str(), partner_trainer)};
    RootedBinaryInputArchive ar(iss, c.get_trainer_grid());
    ar(object);
  }
  c.trainer_barrier(); // I don't think this is necessary either
}

} // namespace ltfb
} // namespace lbann
#endif // LBANN_SRC_EXECUTION_ALGORITHMS_LTFB_CHECKPOINT_COMMON_HPP_INCLUDED
