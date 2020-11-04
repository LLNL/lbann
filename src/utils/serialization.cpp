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

/** @file
 *
 *  Serialization functions for arithmetic types. Specializations for
 *  Cereal's Binary, JSON, and XML archives are provided.
 */

#include "lbann/utils/serialize.hpp"

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>

/** @namespace cereal
 *
 *  Extensions to Cereal for extra arithmetic types used by LBANN.
 */
namespace cereal
{

#ifdef LBANN_HAS_HALF
#ifdef LBANN_HAS_GPU_FP16

/** @brief Save this half-precision value in Binary */
void save(BinaryOutputArchive& archive, __half const& value)
{
  archive.saveBinary(std::addressof(value), sizeof(value));
}

/** @brief Load this half-precision value from Binary */
void load(BinaryInputArchive& archive, __half& value)
{
  archive.loadBinary(std::addressof(value), sizeof(value));
}

#endif // LBANN_HAS_GPU_FP16

// Save/load functions for XML archives
float save_minimal(XMLOutputArchive const&,
                   half_float::half const& val) noexcept
{
  return val;
}

void load_minimal(XMLInputArchive const&, half_float::half& val,
                  float const& in_val) noexcept
{
  val = in_val;
}

// Save/load functions for JSON archives
void save(JSONOutputArchive& oarchive, half_float::half const& val)
{
  std::ostringstream oss;
  oss.precision(std::numeric_limits<long double>::max_digits10);
  oss << val;
  oarchive.saveValue(oss.str());
}

void load(JSONInputArchive& iarchive, half_float::half& val)
{
  std::string encoded;
  iarchive.loadValue(encoded);
  val = std::stof(encoded);
}

#endif // LBANN_HAS_HALF

namespace
{
bool doing_catch_testing_ = false;
}
bool doing_catch_testing() noexcept
{
  return doing_catch_testing_;
}
void set_catch_testing() noexcept
{
  doing_catch_testing_ = true;
}
}// namespace cereal

#include <stack>

namespace lbann
{
namespace utils
{
namespace
{
std::stack<El::Grid const*> grid_stack_;

void push_grid_(El::Grid const& g)
{
  grid_stack_.push(&g);
}

void pop_grid_()
{
  grid_stack_.pop();
}

// This is done this way to ensure the warning is only emitted at most
// once per rank per run, if and only if the default grid is actually
// returned. Outside of testing, it should never be used.
El::Grid const& get_default_grid_() noexcept
{
  El::Grid const& default_grid_ = El::Grid::Default();
  return default_grid_;
}
}// namespace <anon>

El::Grid const& get_current_grid() noexcept
{
  if (grid_stack_.empty())
    return get_default_grid_();
  return *grid_stack_.top();
}

grid_manager::grid_manager(El::Grid const& g)
{
  push_grid_(g);
}

grid_manager::~grid_manager()
{
  pop_grid_();
}

}// namespace utils
}// namespace lbann
