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

#include "Catch2BasicSupport.hpp"

#include "MPITestHelpers.hpp"
#include "TestHelpers.hpp"

#include <lbann/base.hpp>
#include <lbann/data_coordinator/buffered_data_coordinator.hpp>
#include <lbann/data_readers/data_reader.hpp>
#include <lbann/utils/memory.hpp>
#include <lbann/utils/threads/thread_pool.hpp>
#include <map>

using lbann::generic_data_reader;

// Minimal test data reader that returns numbers in sequence until the last
// minibatch, which is of size 1
class test_data_reader : public generic_data_reader
{
public:
  test_data_reader(int minibatches, int minibatch_size)
    : generic_data_reader(),
      m_counter(0),
      m_lastbatch(minibatches - 1),
      m_mbsize(minibatch_size)
  {
    this->m_num_iterations_per_epoch = minibatches;
  }
  test_data_reader* copy() const override
  {
    return new test_data_reader(m_lastbatch + 1, m_mbsize);
  }
  void load() override {}
  std::string get_type() const override { return "test_data_reader"; }
  int get_num_data() const override { return m_lastbatch * m_mbsize + 1; }
  int get_linearized_data_size() const override { return 1; }

  bool fetch_datum(lbann::CPUMat& X, int data_id, int mb_idx) override
  {
    auto X_v = El::View(X, El::ALL, El::IR(mb_idx, mb_idx + 1));
    El::Fill(X_v, El::To<lbann::DataType>(m_counter));
    ++m_counter;
    return true;
  }

private:
  int m_counter;
  int m_lastbatch;
  int m_mbsize;
};

using unit_test::utilities::IsValidPtr;
TEST_CASE("Buffered data coordinator test", "[io][data_coordinator]")
{
  constexpr int minibatch_size = 2;
  constexpr int minibatches = 5;
  constexpr auto mode = lbann::execution_mode::training;

  auto& world_comm = unit_test::utilities::current_world_comm();
  auto io_thread_pool = std::make_unique<lbann::thread_pool>();
  io_thread_pool->launch_threads(2);

  std::map<lbann::execution_mode, lbann::generic_data_reader*> readers;
  readers[mode] = new test_data_reader(minibatches, minibatch_size);
  lbann::buffered_data_coordinator<lbann::DataType> bdc(&world_comm);

  // Set up the data coordinator
  REQUIRE_NOTHROW(bdc.setup(*io_thread_pool, minibatch_size, readers));
  REQUIRE_NOTHROW(
    bdc.register_active_data_field("samples", {1}, minibatch_size));

  // Sample matrix
  auto samples = El::DistMatrix<lbann::DataType,
                                El::STAR,
                                El::STAR,
                                El::ELEMENT,
                                El::Device::CPU>(minibatch_size,
                                                 1,
                                                 world_comm.get_trainer_grid());

  // Test first minibatch
  int remaining_minibatches = minibatches, data = 0;
  REQUIRE_NOTHROW(bdc.fetch_active_batch_synchronous(mode));
  REQUIRE_NOTHROW(bdc.distribute_from_local_matrix(mode, "samples", samples));
  --remaining_minibatches;

  // Check shape and contents
  CHECK(samples.Width() == minibatch_size);
  for (int i = 0; i < minibatch_size; ++i) {
    CHECK(samples.LockedMatrix()(0, i) == data);
    ++data;
  }

  // Test subsequent minibatches for continuity
  for (; remaining_minibatches >= 2; --remaining_minibatches) {
    REQUIRE_NOTHROW(bdc.fetch_data(mode));
    REQUIRE_NOTHROW(bdc.distribute_from_local_matrix(mode, "samples", samples));

    CHECK(samples.Width() == minibatch_size);
    for (int i = 0; i < minibatch_size; ++i) {
      CHECK(samples.LockedMatrix()(0, i) == data);
      ++data;
    }
  }

  // Test last minibatch
  REQUIRE_NOTHROW(bdc.fetch_data(mode));
  REQUIRE_NOTHROW(bdc.distribute_from_local_matrix(mode, "samples", samples));
  CHECK(samples.Width() == 1);
  CHECK(samples.LockedMatrix()(0, 0) == data);
}
