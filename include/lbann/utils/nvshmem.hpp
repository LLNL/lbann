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

#ifndef LBANN_UTILS_NVSHMEM_HPP_INCLUDED
#define LBANN_UTILS_NVSHMEM_HPP_INCLUDED

#include "lbann/base.hpp"
#ifdef LBANN_HAS_NVSHMEM
#include "lbann/utils/gpu/helpers.hpp"
#include "lbann/utils/exception.hpp"
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

namespace lbann {
namespace nvshmem {

/** Whether NVSHMEM has been initialized. */
bool is_initialized() noexcept;

/** Whether NVSHMEM has been finalized. */
bool is_finalized() noexcept;

/** Whether NVSHMEM is active.
 *
 *  Returns true if NVSHMEM has been initialized and has not been
 *  finalized.
 */
bool is_active() noexcept;

/** @brief Initialize NVSHMEM library.
 *
 *  Does nothing if NVSHMEM has already been initialized and throws an
 *  exception if it has already been finalized. This is _not_
 *  thread-safe.
 */
void initialize(MPI_Comm comm=MPI_COMM_WORLD);

/** @brief Finalize NVSHMEM library.
 *
 *  Does nothing if NVSHMEM has not been initialized or has already
 *  been finalized. This is _not_ thread-safe.
 */
void finalize();

/** @brief Allocate GPU buffer on the NVSHMEM symmetric heap.
 *
 *  Initializes NVSHMEM if needed.
 */
template <typename T=void>
T* malloc(size_t size);

/** @brief Resize GPU buffer on the NVSHMEM symmetric heap.
 *
 *  Initializes NVSHMEM if needed.
 */
template <typename T=void>
T* realloc(T* ptr, size_t size);

} // namespace nvshmem
} // namespace lbann

// =============================================
// Implementation
// =============================================

namespace lbann {
namespace nvshmem {

template <typename T>
T* malloc(size_t size) {
  initialize();
  if (size == 0) {
    return nullptr;
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  auto* ptr = nvshmem_malloc(size * sizeof(T));
  if (ptr == nullptr) {
    LBANN_ERROR(
      "NVSHMEM failed to allocate a GPU buffer ",
      "from the symmetric heap ",
      "(requested ",size," B)");
  }
  return reinterpret_cast<T*>(ptr);
}

template <typename T>
T* realloc(T* ptr, size_t size) {
  initialize();

  /// @todo Use nvshmem_realloc once it's supported
  if (ptr != nullptr) {
    nvshmem_free(ptr);
  }
  return malloc<T>(size);

}

} // namespace nvshmem
} // namespace lbann

#endif // LBANN_HAS_NVSHMEM

#endif // LBANN_UTILS_NVSHMEM_HPP_INCLUDED
