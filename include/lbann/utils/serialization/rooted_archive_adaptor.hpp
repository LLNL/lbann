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
#pragma once
#ifndef LBANN_UTILS_SERIALIZATION_ROOTED_ARCHIVE_ADAPTOR_HPP_
#define LBANN_UTILS_SERIALIZATION_ROOTED_ARCHIVE_ADAPTOR_HPP_

#ifndef __CUDACC__

#include <El.hpp>

#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>

#include <optional>
#include <string>

namespace lbann
{

// An archive that collects data to the root of a grid on save and
// broadcasts/scatters it on load.
template <typename OutputArchiveT>
class RootedOutputArchiveAdaptor
  : public cereal::OutputArchive<RootedOutputArchiveAdaptor<OutputArchiveT>>
{
  static_assert(lbann::utils::IsBuiltinArchive<OutputArchiveT>,
                "At this time only built-in Cereal archives are supported.");
  static_assert(lbann::utils::IsOutputArchive<OutputArchiveT>,
                "The given archive type must be an \"output\" archive type.");

public:
  using archive_type = OutputArchiveT;
private:
  using ThisType_ = RootedOutputArchiveAdaptor<archive_type>;
  using BaseType_ = cereal::OutputArchive<ThisType_>;
public:
  RootedOutputArchiveAdaptor(std::ostream& os,
                             El::Grid const& g,
                             El::Int root=0)
    : BaseType_{this},
      ar_(g.Rank() == root
          ? std::make_optional<archive_type>(os)
          : std::nullopt),
      grid_{&g},
      root_{root}
  {}

  El::Grid const& grid() const noexcept
  {
    return *grid_;
  }

  El::Int root() const noexcept
  {
    return root_;
  }

  bool am_root() const noexcept
  {
    return (this->root() == grid_->Rank());
  }

  void set_next_name(std::string const& name)
  {
    if (!this->am_root())
      return;
    next_name_ = name;
  }

  template <typename T>
  void save_on_root(T const& data)
  {
    if (!this->am_root())
      return;

    if (next_name_.size())
    {
      ar_.value()(::cereal::make_nvp(next_name_, data));
      next_name_.clear();
    }
    else
      ar_.value()(data);
  }

private:
  std::string next_name_;
  std::optional<archive_type> ar_;
  El::Grid const* grid_;
  El::Int root_;
};// RootedOutputArchiveAdaptor

template <typename InputArchiveT>
class RootedInputArchiveAdaptor
  : public cereal::InputArchive<RootedInputArchiveAdaptor<InputArchiveT>>
{
  static_assert(lbann::utils::IsBuiltinArchive<InputArchiveT>,
                "At this time only built-in Cereal archives are supported.");
  static_assert(lbann::utils::IsInputArchive<InputArchiveT>,
                "The given archive type must be an \"input\" archive type.");
public:
  using archive_type = InputArchiveT;
private:
  using ThisType_ = RootedInputArchiveAdaptor<archive_type>;
  using BaseType_ = cereal::InputArchive<ThisType_>;
public:
  RootedInputArchiveAdaptor(std::istream& is,
                            El::Grid const& g,
                            El::Int root=0)
    : BaseType_{this},
      ar_(g.Rank() == root
          ? std::make_optional<archive_type>(is)
          : std::nullopt),
      grid_{&g},
      root_{root}
  {}

  El::Grid const& grid() const noexcept
  {
    return *grid_;
  }

  El::Int root() const noexcept
  {
    return root_;
  }

  bool am_root() const noexcept
  {
    return (this->root() == grid_->Rank());
  }

  void set_next_name(std::string const& name)
  {
    if (!this->am_root())
      return;
    next_name_ = name;
  }

  template <typename T>
  void load_on_root(T& data)
  {
    if (!this->am_root())
      return;

    if (!next_name_.empty())
    {
      // If I read things correctly, this should forward 'data' as an
      // lvalue reference and it should be automatically written by
      // the read.
      ar_.value()(::cereal::make_nvp(next_name_, data));
      next_name_.clear();
    }
    else
      ar_.value()(data);
  }

private:
  std::string next_name_;
  std::optional<archive_type> ar_;
  El::Grid const* grid_;
  El::Int root_;
};// RootedInputArchiveAdaptor

using RootedBinaryInputArchive =
  RootedInputArchiveAdaptor<cereal::BinaryInputArchive>;
using RootedBinaryOutputArchive =
  RootedOutputArchiveAdaptor<cereal::BinaryOutputArchive>;

using RootedJSONInputArchive =
  RootedInputArchiveAdaptor<cereal::JSONInputArchive>;
using RootedJSONOutputArchive =
  RootedOutputArchiveAdaptor<cereal::JSONOutputArchive>;

using RootedXMLInputArchive =
  RootedInputArchiveAdaptor<cereal::XMLInputArchive>;
using RootedXMLOutputArchive =
  RootedOutputArchiveAdaptor<cereal::XMLOutputArchive>;

}// namespace lbann

namespace cereal
{

// POD types are "broadcast" types by default. That is, the root value
// is stored in the archive and merely "forgotten" on non-root
// processes. Ideally, this would be controlled by a
// "HasValidMPIDataType" trait or something.
template <typename OutputArchiveT, typename DataT>
h2::meta::EnableWhen<std::is_arithmetic_v<DataT>, void>
CEREAL_SAVE_FUNCTION_NAME(
  lbann::RootedOutputArchiveAdaptor<OutputArchiveT>& ar,
  DataT const& val)
{
  ar.save_on_root(val);
}

template <typename OutputArchiveT>
void CEREAL_SAVE_FUNCTION_NAME(
  lbann::RootedOutputArchiveAdaptor<OutputArchiveT>& ar, bool const& b)
{
  int val = b;
  ar.save_on_root(val);
}

// POD types are "broadcast" types by default. They are read on the
// root and broadcast across the grid.
template <typename InputArchiveT, typename DataT>
h2::meta::EnableWhen<std::is_arithmetic_v<DataT>, void>
CEREAL_LOAD_FUNCTION_NAME(lbann::RootedInputArchiveAdaptor<InputArchiveT>& ar,
                          DataT& val)
{
  static_assert(!std::is_same_v<DataT,char>,
                "Don't be a basic char.");

  ar.load_on_root(val);
  El::mpi::Broadcast(val, ar.root(), ar.grid().Comm(),
                     El::SyncInfo<El::Device::CPU>{});
}

template <typename InputArchiveT>
void CEREAL_LOAD_FUNCTION_NAME(
  lbann::RootedInputArchiveAdaptor<InputArchiveT>& ar, bool& b)
{
  int val;
  load(ar, val);
  b = val;
}

// TODO: This may need some work. The current implementation is
// inspired by the XML archives in Cereal.
template <class OutputArchiveT, class DataT>
void CEREAL_SAVE_FUNCTION_NAME(
  lbann::RootedOutputArchiveAdaptor<OutputArchiveT>& ar,
  NameValuePair<DataT> const& nvp)
{
  ar.set_next_name(nvp.name);
  ar(nvp.value);
}

// TODO: This may need some work. The current implementation is
// inspired by the XML archives in Cereal.
template <class InputArchiveT, class DataT>
void CEREAL_LOAD_FUNCTION_NAME(
  lbann::RootedInputArchiveAdaptor<InputArchiveT>& ar,
  NameValuePair<DataT>& nvp)
{
  ar.set_next_name(nvp.name);
  ar(nvp.value);
}

template <class ArchiveT, class T>
void CEREAL_SAVE_FUNCTION_NAME(
  lbann::RootedOutputArchiveAdaptor<ArchiveT>& ar,
  SizeTag<T> const& tag)
{
  ar(tag.size);
}

template <class ArchiveT, class T>
void CEREAL_LOAD_FUNCTION_NAME(
  lbann::RootedInputArchiveAdaptor<ArchiveT>& ar,
  SizeTag<T>& tag)
{
  ar(tag.size);
}

}// namespace cereal

CEREAL_REGISTER_ARCHIVE(
  lbann::RootedInputArchiveAdaptor<cereal::BinaryInputArchive>);
CEREAL_REGISTER_ARCHIVE(
  lbann::RootedOutputArchiveAdaptor<cereal::BinaryOutputArchive>);
CEREAL_REGISTER_ARCHIVE(
  lbann::RootedInputArchiveAdaptor<cereal::JSONInputArchive>);
CEREAL_REGISTER_ARCHIVE(
  lbann::RootedOutputArchiveAdaptor<cereal::JSONOutputArchive>);
CEREAL_REGISTER_ARCHIVE(
  lbann::RootedInputArchiveAdaptor<cereal::XMLInputArchive>);
CEREAL_REGISTER_ARCHIVE(
  lbann::RootedOutputArchiveAdaptor<cereal::XMLOutputArchive>);

CEREAL_SETUP_ARCHIVE_TRAITS(
  lbann::RootedInputArchiveAdaptor<cereal::BinaryInputArchive>,
  lbann::RootedOutputArchiveAdaptor<cereal::BinaryOutputArchive>);
CEREAL_SETUP_ARCHIVE_TRAITS(
  lbann::RootedInputArchiveAdaptor<cereal::JSONInputArchive>,
  lbann::RootedOutputArchiveAdaptor<cereal::JSONOutputArchive>);
CEREAL_SETUP_ARCHIVE_TRAITS(
  lbann::RootedInputArchiveAdaptor<cereal::XMLInputArchive>,
  lbann::RootedOutputArchiveAdaptor<cereal::XMLOutputArchive>);

#endif // __CUDACC__
#endif // LBANN_UTILS_SERIALIZATION_ROOTED_ARCHIVE_ADAPTOR_HPP_
