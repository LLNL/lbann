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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_SETUP_COMMUNITYGAN_DATA_READER_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_SETUP_COMMUNITYGAN_DATA_READER_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

#ifdef LBANN_HAS_COMMUNITYGAN_WALKER
// Forward declaration
namespace lbann {
class communitygan_reader;
}
#endif // LBANN_HAS_COMMUNITYGAN_WALKER

namespace lbann {
namespace callback {

#ifdef LBANN_HAS_COMMUNITYGAN_WALKER

class setup_communitygan_data_reader : public callback_base {
public:

  setup_communitygan_data_reader() : callback_base() {}
  setup_communitygan_data_reader(
    const setup_communitygan_data_reader&) = default;
  setup_communitygan_data_reader& operator=(
    const setup_communitygan_data_reader&) = default;
  setup_communitygan_data_reader* copy() const override;
  std::string name() const override;

  void on_setup_end(model *m) override;

  /** @name Serialization */
  ///@{

  /** @brief Store state to archive for checkpoint and restart */
  template <class Archive> void serialize(Archive & ar);

  ///@}

  static void register_communitygan_data_reader(
    ::lbann::communitygan_reader* reader);

private:

  static communitygan_reader* m_reader;

};

#endif // LBANN_HAS_COMMUNITYGAN_WALKER

// Builder function
std::unique_ptr<callback_base>
build_setup_communitygan_data_reader_callback_from_pbuf(
  const google::protobuf::Message&,
  std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_SETUP_COMMUNITYGAN_DATA_READER_HPP_INCLUDED
