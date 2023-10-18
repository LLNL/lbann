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
// #include <lbann/data_readers/data_reader_synthetic.hpp>
#include <lbann/utils/memory.hpp>
#include <lbann/utils/threads/thread_pool.hpp>
#include <map>

using lbann::generic_data_reader;

namespace {

// void fill_matrix(CPUMat& mat)
// {
//   std::normal_distribution<DataType> dist(DataType(0), DataType(1));
//   const El::Int height = mat.Height(); // Width is 1.
//   DataType* __restrict__ buf = mat.Buffer();
//   for (El::Int i = 0; i < height; ++i) {
//     buf[i] = m_sample[i];
//     LBANN_MSG("synthetic data reader is filling a matrix with buf[", i, "]=",
//     buf[i]);
//   }
// }

} // anonymous namespace

// Minimal test data reader that returns numbers in sequence until the last
// minibatch, which is of size 1
class test_data_reader : public generic_data_reader
{
public:
  test_data_reader(int num_mini_batches, int mini_batch_size)
    : generic_data_reader(), m_counter(0), m_lastbatch(num_mini_batches - 1)
  {
    this->m_mini_batch_size = mini_batch_size;
    this->m_num_iterations_per_epoch = num_mini_batches;
    this->m_shuffle = false;
    this->m_use_data_store = false;
  }
  test_data_reader* copy() const override
  {
    return new test_data_reader(m_lastbatch + 1, m_mini_batch_size);
  }
  //  void load() override {}
  std::string get_type() const override { return "test_data_reader"; }
  int get_num_data() const override
  {
    return m_lastbatch * m_mini_batch_size + 1;
  }
  int get_linearized_data_size() const override { return 1; }

  bool fetch_datum(lbann::CPUMat& X, int data_id, int mb_idx) override
  {
    LBANN_MSG("fetch datum is fetching data id ",
              data_id,
              " and mb index ",
              mb_idx);
    auto X_v = El::View(X, El::ALL, El::IR(mb_idx, mb_idx + 1));
    auto Y_v = El::View(m_samples, El::ALL, El::IR(data_id, data_id + 1));
    const El::Int height = X_v.Height(); // Width is 1.
    lbann::DataType* __restrict__ dst = X_v.Buffer();
    lbann::DataType* __restrict__ src = Y_v.Buffer();
    for (El::Int i = 0; i < height; ++i) {
      dst[i] = src[i];
      LBANN_MSG("test data reader is feetching a sample for a matrix with src[",
                i,
                "]=",
                src[i]);
    }
    LBANN_MSG("X_v");
    El::Print(X_v);
    LBANN_MSG("Y_v");
    El::Print(Y_v);
    return true;
  }

  void load()
  {
    LBANN_MSG("Load the matrix");
    m_samples.Resize(1, get_num_data());
    const El::Int height = m_samples.Height(); // Width is 1.
    const El::Int width = m_samples.Width();   // Width is 1.
    for (El::Int i = 0; i < height; ++i) {
      for (El::Int j = 0; j < width; ++j) {
        m_samples.Set(i, j, El::To<float>(m_counter++));
        LBANN_MSG("test data reader is filling a matrix with Y[",
                  i,
                  ", ",
                  j,
                  "]=",
                  m_samples.Get(i, j));
      }
    }
    m_shuffled_indices.resize(get_num_data());
    std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
    for (auto i : m_shuffled_indices) {
      LBANN_MSG("The original shuffled indices are ", i);
    }
  }

private:
  int m_counter;
  int m_lastbatch;
  // int m_mbsize;
  El::DistMatrix<lbann::DataType,
                 El::STAR,
                 El::STAR,
                 El::ELEMENT,
                 El::Device::CPU>
    m_samples;
};

using unit_test::utilities::IsValidPtr;
TEST_CASE("Buffered data coordinator test", "[io][data_coordinator][sync]")
{
  constexpr int minibatch_size = 2;
  constexpr int minibatches = 5;
  constexpr auto mode = lbann::execution_mode::training;

  auto& world_comm = unit_test::utilities::current_world_comm();
  // initialize stuff (boilerplate)
  lbann::init_random(0, 2);
  lbann::init_data_seq_random(42);
  auto io_thread_pool = std::make_unique<lbann::thread_pool>();
  io_thread_pool->launch_threads(1);

  std::map<lbann::execution_mode, lbann::generic_data_reader*> readers;
  //  readers[mode] = new lbann::data_reader_synthetic(10, 1, false, true);
  readers[mode] = new test_data_reader(minibatches, minibatch_size);
  readers[mode]->set_mini_batch_size(minibatch_size);
  //  world_comm.set_procs_per_trainer
  readers[mode]->setup(io_thread_pool->get_num_threads(), io_thread_pool.get());
  readers[mode]->set_comm(&world_comm);
  readers[mode]->set_num_parallel_readers(1);
  readers[mode]->load();
  //  readers[mode]->set_initial_position();
  //  readers[mode] = new test_data_reader(minibatches, minibatch_size);
  lbann::buffered_data_coordinator<lbann::DataType> bdc(&world_comm);

  // Set up the data coordinator
  bdc.setup(*io_thread_pool, minibatch_size, readers);
  //  REQUIRE_NOTHROW(bdc.setup(*io_thread_pool, minibatch_size, readers));
  REQUIRE_NOTHROW(bdc.register_active_data_field("samples", {1}));
  REQUIRE_NOTHROW(bdc.setup_data_fields(minibatch_size));
  readers[mode]->print_config();

  // Sample matrix
  auto samples = El::DistMatrix<lbann::DataType,
                                El::STAR,
                                El::STAR,
                                El::ELEMENT,
                                El::Device::CPU>(minibatch_size,
                                                 1,
                                                 world_comm.get_trainer_grid());
  SECTION("Synchronous I/O")
  {
    // Test first minibatch
    int remaining_minibatches = minibatches, data = 0;
    bool epoch_done = false;
    REQUIRE_NOTHROW(bdc.fetch_active_batch_synchronous(mode));
    REQUIRE_NOTHROW(bdc.distribute_from_local_matrix(mode, "samples", samples));
    REQUIRE_NOTHROW(epoch_done = bdc.ready_for_next_fetch(mode));
    CHECK(epoch_done != true);
    --remaining_minibatches;

    // Check shape and contents
    CHECK(samples.Width() == minibatch_size);
    for (int i = 0; i < minibatch_size; ++i) {
      CHECK(samples.LockedMatrix()(0, i) == data);
      ++data;
    }

    // Test subsequent minibatches for continuity
    for (; remaining_minibatches >= 2; --remaining_minibatches) {
      REQUIRE_NOTHROW(bdc.fetch_active_batch_synchronous(mode));
      //    REQUIRE_NOTHROW(bdc.fetch_data(mode));
      REQUIRE_NOTHROW(
        bdc.distribute_from_local_matrix(mode, "samples", samples));
      REQUIRE_NOTHROW(epoch_done = bdc.ready_for_next_fetch(mode));
      CHECK(epoch_done != true);

      CHECK(samples.Width() == minibatch_size);
      for (int i = 0; i < minibatch_size; ++i) {
        CHECK(samples.LockedMatrix()(0, i) == data);
        ++data;
      }
    }

    // Test last minibatch
    REQUIRE_NOTHROW(bdc.fetch_active_batch_synchronous(mode));
    //  REQUIRE_NOTHROW(bdc.fetch_data(mode));
    REQUIRE_NOTHROW(bdc.distribute_from_local_matrix(mode, "samples", samples));
    REQUIRE_NOTHROW(epoch_done = bdc.ready_for_next_fetch(mode));
    CHECK(epoch_done == true);
    CHECK(samples.Width() == 1);
    CHECK(samples.LockedMatrix()(0, 0) == data);
  }

  SECTION("Asynchronous I/O (Background)")
  {
    // Test first minibatch
    int remaining_minibatches = minibatches, data = 0;
    bool epoch_done = false;

    // For background data fetching, start with a synchronous request.
    REQUIRE_NOTHROW(bdc.fetch_active_batch_synchronous(mode));

    // Test subsequent minibatches for continuity
    for (; remaining_minibatches >= 1; --remaining_minibatches) {
      REQUIRE_NOTHROW(bdc.fetch_data(mode));
      REQUIRE_NOTHROW(
        bdc.distribute_from_local_matrix(mode, "samples", samples));
      REQUIRE_NOTHROW(epoch_done = bdc.ready_for_next_fetch(mode));
      auto current_mini_batch_size = minibatch_size;
      if (epoch_done == true) {
        // Last mini-batch should be only a single sample
        current_mini_batch_size = 1;
      }
      CHECK(samples.Width() == current_mini_batch_size);
      for (int i = 0; i < current_mini_batch_size; ++i) {
        CHECK(samples.LockedMatrix()(0, i) == data);
        ++data;
      }
    }

    CHECK(data == readers[mode]->get_num_data());
  }
}
