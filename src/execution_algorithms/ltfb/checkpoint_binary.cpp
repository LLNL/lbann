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
#include "lbann/execution_algorithms/ltfb/random_pairwise_exchange.hpp"

#include "lbann/comm_impl.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/serialization/rooted_archive_adaptor.hpp"
#include "lbann/weights/data_type_weights.hpp"

#include "checkpoint_common.hpp"

namespace lbann {

namespace ltfb {

CheckpointBinary::CheckpointBinary(std::set<std::string> const& weights_names)
  : BaseType(weights_names)
{}
CheckpointBinary::CheckpointBinary(std::set<std::string>&& weights_names)
  : BaseType(std::move(weights_names))
{}
std::unique_ptr<model>
CheckpointBinary::get_partner_model(model const& m,
                                    El::Int partner_trainer,
                                    size_t /*step*/)
{
  auto const& comm = *m.get_comm();

  // Start by copying this model, then do the exchange.
  auto partner_model_ptr = m.copy_model();
  auto& partner_model = *partner_model_ptr;

  // Keep track of weights that shouldn't be exchanged
  std::unordered_map<std::string, std::unique_ptr<weights>> restore_weights;
  auto const& weights_names = this->weights_names();
  if (!weights_names.empty()) {
    for (auto w : partner_model.get_weights()) {
      if (weights_names.find(w->get_name()) == weights_names.cend()) {
        using TensorDataType = DataType;
        using WeightsType = data_type_weights<TensorDataType>;
        restore_weights[w->get_name()] =
          make_unique<WeightsType>(dynamic_cast<WeightsType&>(*w));
      }
    }
  }

  // Save model checkpoint
  std::ostringstream oss;
  {
    RootedBinaryOutputArchive ar(oss, comm.get_trainer_grid());
    comm.trainer_barrier();
    ar(m);
  }

  // sure, why not
  comm.trainer_barrier();

  // Synchronize with partner trainer
  std::string save_model_ckpt = oss.str(), load_model_ckpt;
  if (comm.am_trainer_master()) {
    std::size_t save_size = save_model_ckpt.size(), load_size = 0;
    comm.sendrecv(&save_size,
                  1,
                  partner_trainer,
                  0,
                  &load_size,
                  1,
                  partner_trainer,
                  0,
                  El::SyncInfo<El::Device::CPU>{});
    load_model_ckpt.resize(load_size);

    auto const* send_buf =
      reinterpret_cast<El::byte const*>(save_model_ckpt.data());
    auto* recv_buf = reinterpret_cast<El::byte*>(load_model_ckpt.data());

    while (save_size || load_size) {
      // Get the max blk size
      auto constexpr max_blk_size = std::numeric_limits<int>::max();
      std::size_t constexpr max_blk_size_size_t = max_blk_size;

      int this_blk_send_size =
        (save_size > max_blk_size_size_t ? max_blk_size : save_size);
      int this_blk_recv_size =
        (load_size > max_blk_size_size_t ? max_blk_size : load_size);

      comm.sendrecv(send_buf,
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
      save_size =
        (save_size > max_blk_size_size_t ? save_size - max_blk_size_size_t : 0);
      load_size =
        (load_size > max_blk_size_size_t ? load_size - max_blk_size_size_t : 0);
    }
  }

  // sure, why not
  comm.trainer_barrier();

  // Load model checkpoint from partner trainer
  {
    std::istringstream iss{std::move(load_model_ckpt)};
    RootedBinaryInputArchive ar(iss, comm.get_trainer_grid());
    ar(partner_model);
  }

  /// @todo Should be unneeded, but we experience hangs without it
  comm.trainer_barrier();

  restore_model_weights(partner_model, restore_weights);

  return partner_model_ptr;
}

} // namespace ltfb
} // namespace lbann
