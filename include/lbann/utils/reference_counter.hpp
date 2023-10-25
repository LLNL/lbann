////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_UTILS_REFERENCE_COUNTER_HPP_INCLUDED
#define LBANN_UTILS_REFERENCE_COUNTER_HPP_INCLUDED

#include <map>
#include <memory>

namespace lbann {

/**
 * @brief Type-agnostic Hydrogen matrix emptier base class. Used for externally
 * reference-counting matrices.
 */
class MatEmptier
{
public:
  virtual ~MatEmptier() = default;
  virtual void empty() = 0;
};

/**
 * @brief Typed Hydrogen matrix emptier subclass. Used for externally
 * reference-counting matrices.
 */
template <typename T>
class MatEmptierImpl : public MatEmptier
{
public:
  MatEmptierImpl(El::AbstractDistMatrix<T>& mat) : mat_{&mat} {}
  ~MatEmptierImpl() { mat_ = nullptr; }

  void empty() final { mat_->Empty(); }

private:
  El::AbstractDistMatrix<T>* mat_;
};

/**
 * @brief Reference-counts a Hydrogen matrix; empties when refcount is zero.
 */
class MatrixRefCounter
{
public:
  using PtrRange = std::pair<void const*, void const*>;

  template <typename T>
  MatrixRefCounter(El::AbstractDistMatrix<T>& mat, void const* owner = nullptr)
    : emptier_{std::make_unique<MatEmptierImpl<T>>(mat)},
      rfcnt_{0},
      owner_(owner)
  {
    range_ = get_range(mat);
    this->inc();
  }

  /** @brief Increments the reference count. */
  void inc() { ++rfcnt_; }

  /** @brief Decrements the reference count. */
  void dec()
  {
    --rfcnt_;
    if (rfcnt_ <= 0)
      emptier_->empty();
  }

  /** @brief Returns the reference count. */
  int count() const { return rfcnt_; }

  template <typename T>
  static PtrRange get_range(const El::AbstractDistMatrix<T>& mat)
  {
    T const* begin = mat.LockedBuffer();
    T const* end = begin + mat.AllocatedMemory();
    return std::make_pair((void const*)begin, (void const*)end);
  }

  /** @brief Returns the start and end pointers used in the allocated buffer. */
  PtrRange range() const { return this->range_; }

  /** @brief Returns the object that created the tracked buffer. */
  void const* get_owner() const { return this->owner_; }

private:
  std::unique_ptr<MatEmptier> emptier_; ///< Type agnostic matrix emptier object
  int rfcnt_;                           ///< Reference count
  PtrRange range_;    ///< (start,end) pointer range for allocated memory
  void const* owner_; ///< Used for provenance tracking (e.g., parent Layer)
};

/** @brief Checks if a pointer is in range of an allocated matrix. */
static inline bool is_in_range(MatrixRefCounter::PtrRange const& range,
                               void const* const ptr)
{
  auto const& [start, end] = range;
  return ptr >= start && ptr < end;
}

using PointerRangeReferenceCounter =
  std::map<MatrixRefCounter::PtrRange, MatrixRefCounter>;

static inline PointerRangeReferenceCounter::iterator
lookup_pointer(PointerRangeReferenceCounter& counter, void const* const ptr)
{
  return std::find_if(begin(counter), end(counter), [ptr](auto& a) {
    return is_in_range(a.first, ptr);
  });
}

} // namespace lbann

#endif // LBANN_UTILS_REFERENCE_COUNTER_HPP_INCLUDED
