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
#include "lbann/proto/lbann.pb.h"
#include "lbann/proto/proto_common.hpp"
#include <google/protobuf/text_format.h>

#include <cstdlib>
#include <errno.h>
#include <string.h>

// #include "./data_reader_common_catch2.hpp"
#include "lbann/data_ingestion/readers/data_reader_synthetic.hpp"
#include "lbann/data_ingestion/readers/utils/input_data_type.hpp"
#include "lbann/utils/dim_helpers.hpp"
#include "lbann/utils/hash.hpp"
#include "lbann/utils/threads/thread_pool.hpp"
#include "lbann/utils/threads/thread_utils.hpp"

class DataReaderSyntheticWhiteboxTester
{
public:
  bool fetch_datum(lbann::data_reader_synthetic& dr,
                   lbann::CPUMat& X,
                   uint64_t data_id,
                   uint64_t mb_idx)
  {
    return dr.fetch_datum(X, data_id, mb_idx);
  }
  bool fetch_label(lbann::data_reader_synthetic& dr,
                   lbann::CPUMat& Y,
                   uint64_t data_id,
                   uint64_t mb_idx)
  {
    return dr.fetch_label(Y, data_id, mb_idx);
  }
  bool fetch_response(lbann::data_reader_synthetic& dr,
                      lbann::CPUMat& Y,
                      uint64_t data_id,
                      uint64_t mb_idx)
  {
    return dr.fetch_response(Y, data_id, mb_idx);
  }
};

TEST_CASE("Synthetic data reader public API tests",
          "[mpi][data_reader][synthetic][public]")
{
  // initialize stuff (boilerplate)
  auto& comm = unit_test::utilities::current_world_comm();
  int seed = 42;
  lbann::init_random(seed, 1);
  lbann::init_data_seq_random(seed);

  // Create a local copy of the RNG to check the synthetic data reader
  lbann::fast_rng_gen ref_fast_generator;
  // Mix in the rank in trainer
  seed = lbann::hash_combine(seed, comm.get_rank_in_trainer());
  // Mix in the I/O thread rank
  ref_fast_generator.seed(lbann::hash_combine(seed, 0));

  // Initalize a per-trainer I/O thread pool
  auto io_thread_pool = std::make_unique<lbann::thread_pool>();
  io_thread_pool->launch_pinned_threads(1, 1);

  std::set<std::string> active_data_fields = {"samples"};
  active_data_fields.insert(
    GENERATE(std::string("labels"), std::string("responses")));
  auto s = GENERATE(range(1, 11));
  El::Int num_samples = s;
  std::vector<El::Int> dims = {s, s};
  El::Int num_labels = s * 2;
  std::vector<El::Int> response_dims = {s + 1, s + 1};

  std::map<lbann::data_field_type, std::unique_ptr<lbann::CPUMat>>
    owning_local_input_buffers;
  std::map<lbann::data_field_type, lbann::CPUMat*> local_input_buffers;
  for (auto& data_field : active_data_fields) {
    auto local_mat = std::make_unique<lbann::CPUMat>();
    if (data_field == INPUT_DATA_TYPE_SAMPLES) {
      local_mat->Resize(dims[0] * dims[1], num_samples);
      El::Zeros_seq(*local_mat, dims[0] * dims[1], num_samples);
    }
    else if (data_field == INPUT_DATA_TYPE_LABELS) {
      local_mat->Resize(num_labels, num_samples);
      El::Zeros_seq(*local_mat, num_labels, num_samples);
    }
    else if (data_field == INPUT_DATA_TYPE_RESPONSES) {
      local_mat->Resize(response_dims[0] * response_dims[1], num_samples);
    }
    local_input_buffers[data_field] = local_mat.get();
    owning_local_input_buffers[data_field] = std::move(local_mat);
  }
  El::Matrix<El::Int> indices_fetched;
  El::Zeros_seq(indices_fetched, num_samples, 1);

  lbann::dataset ds;
  ds.setup(num_samples, "training");

  SECTION("fetch data fields")
  {
    std::unique_ptr<lbann::data_reader_synthetic> dr;
    if (owning_local_input_buffers.find(INPUT_DATA_TYPE_LABELS) !=
        owning_local_input_buffers.end()) {
      dr = std::make_unique<lbann::data_reader_synthetic>(num_samples,
                                                          dims,
                                                          num_labels,
                                                          false);
    }
    else if (owning_local_input_buffers.find(INPUT_DATA_TYPE_RESPONSES) !=
             owning_local_input_buffers.end()) {
      dr = std::make_unique<lbann::data_reader_synthetic>(num_samples,
                                                          dims,
                                                          response_dims,
                                                          false);
    }
    else {
      LBANN_ERROR("Unknown data field");
    }
    dr->setup(io_thread_pool->get_num_threads(), io_thread_pool.get());
    dr->set_comm(&comm);
    dr->load();
    ds.set_mini_batch_size(num_samples);
    ds.set_last_mini_batch_size(num_samples);
    ds.set_initial_position();

    dr->fetch(local_input_buffers,
              indices_fetched,
              ds.get_position(),
              ds.get_sample_stride(),
              num_samples);

    // Check all of the results that were fetched.  Ensure that the
    // data fields are accessed in the same order that they are in the map
    for (El::Int j = 0; j < num_samples; j++) {
      for (auto& data_field : active_data_fields) {
        if (data_field == INPUT_DATA_TYPE_SAMPLES ||
            data_field == INPUT_DATA_TYPE_RESPONSES) {
          auto& X = *(local_input_buffers[data_field]);
          // Create a new normal distribution for each sample.  This ensures
          // that the behavior matches the implementation in the synthetic
          // data reader and handles the case of odd numbers of entries with a
          // normal distriubtion implementation. (Specifically that entries
          // for a normal distribution are generated in pairs.)
          std::normal_distribution<lbann::DataType> dist(float(0), float(1));
          for (El::Int i = 0; i < X.Height(); i++) {
            CHECK(X(i, j) == dist(ref_fast_generator));
          }
        }
        else if (data_field == INPUT_DATA_TYPE_LABELS) {
          auto& Y = *(local_input_buffers[INPUT_DATA_TYPE_LABELS]);
          auto index = lbann::fast_rand_int(ref_fast_generator, num_labels);
          for (El::Int i = 0; i < Y.Height(); i++) {
            if (index == i) {
              CHECK(Y(i, j) == 1);
            }
            else {
              CHECK(Y(i, j) == 0);
            }
          }
        }
      }
    }
  }
}

TEST_CASE("Synthetic data reader public API tests - arbitrary field",
          "[mpi][data_reader][synthetic][public][data_field]")
{
  // initialize stuff (boilerplate)
  auto& comm = unit_test::utilities::current_world_comm();
  int seed = 42;
  lbann::init_random(seed, 1);
  lbann::init_data_seq_random(seed);

  // Create a local copy of the RNG to check the synthetic data reader
  lbann::fast_rng_gen ref_fast_generator;
  // Mix in the rank in trainer
  seed = lbann::hash_combine(seed, comm.get_rank_in_trainer());
  // Mix in the I/O thread rank
  ref_fast_generator.seed(lbann::hash_combine(seed, 0));

  // Initalize a per-trainer I/O thread pool
  auto io_thread_pool = std::make_unique<lbann::thread_pool>();
  io_thread_pool->launch_pinned_threads(1, 1);

  //  std::set<std::string> active_data_fields = {"samples"};
  auto s = GENERATE(range(1, 2));
  El::Int num_samples = s;
  std::set<lbann::data_field_type> data_fields = {"foo", "bar"};
  std::map<lbann::data_field_type, std::vector<El::Int>> fields;
  int f = 0;
  std::map<lbann::data_field_type, std::unique_ptr<lbann::CPUMat>>
    owning_local_input_buffers;
  std::map<lbann::data_field_type, lbann::CPUMat*> local_input_buffers;
  for (auto const& data_field : data_fields) {
    std::vector<El::Int> dims = {s + f, s + f};
    fields[data_field] = dims;
    ++f;
    auto local_mat = std::make_unique<lbann::CPUMat>();
    auto sample_size = lbann::get_linear_size(dims);
    local_mat->Resize(sample_size, num_samples);
    El::Zeros_seq(*local_mat, sample_size, num_samples);
    local_input_buffers[data_field] = local_mat.get();
    owning_local_input_buffers[data_field] = std::move(local_mat);
  }
  El::Matrix<El::Int> indices_fetched;
  El::Zeros_seq(indices_fetched, num_samples, 1);

  lbann::dataset ds;
  ds.setup(num_samples, "training");

  SECTION("fetch arbitrary data fields")
  {
    auto dr = std::make_unique<lbann::data_reader_synthetic>(num_samples,
                                                             fields,
                                                             false);
    dr->setup(io_thread_pool->get_num_threads(), io_thread_pool.get());
    dr->set_comm(&comm);
    dr->load();
    ds.set_mini_batch_size(num_samples);
    ds.set_last_mini_batch_size(num_samples);
    ds.set_initial_position();

    dr->fetch(local_input_buffers,
              indices_fetched,
              ds.get_position(),
              ds.get_sample_stride(),
              num_samples);

    // Check all of the results that were fetched.  Ensure that the
    // data fields are accessed in the same order that they are in the map
    for (El::Int j = 0; j < num_samples; j++) {
      for (auto const& data_field : data_fields) {
        auto& X = *(local_input_buffers[data_field]);
        // Create a new normal distribution for each sample.  This ensures
        // that the behavior matches the implementation in the synthetic
        // data reader and handles the case of odd numbers of entries with a
        // normal distriubtion implementation. (Specifically that entries
        // for a normal distribution are generated in pairs.)
        std::normal_distribution<lbann::DataType> dist(float(0), float(1));
        for (El::Int i = 0; i < X.Height(); i++) {
          CHECK(X(i, j) == dist(ref_fast_generator));
        }
      }
    }
  }

  SECTION("fetch arbitrary bad data field with extra fields")
  {
    std::map<lbann::data_field_type, std::vector<El::Int>> test_fields;
    lbann::data_field_type bad_field = "bar";
    for (auto const& data_field : data_fields) {
      if (data_field != bad_field) {
        test_fields[data_field] = fields[data_field];
      }
    }
    auto dr = std::make_unique<lbann::data_reader_synthetic>(num_samples,
                                                             test_fields,
                                                             false);
    dr->setup(io_thread_pool->get_num_threads(), io_thread_pool.get());
    dr->set_comm(&comm);
    dr->load();
    ds.set_mini_batch_size(num_samples);
    ds.set_last_mini_batch_size(num_samples);
    ds.set_initial_position();

    CHECK_THROWS(dr->fetch(local_input_buffers,
                           indices_fetched,
                           ds.get_position(),
                           ds.get_sample_stride(),
                           num_samples));

    // All data buffers should be empty since it will have thrown an exception
    for (El::Int j = 0; j < num_samples; j++) {
      for (auto const& data_field : data_fields) {
        auto& X = *(local_input_buffers[data_field]);
        for (El::Int i = 0; i < X.Height(); i++) {
          CHECK(X(i, j) == 0.0f);
        }
      }
    }
  }

  SECTION("fetch arbitrary bad data fields - no extra buffers")
  {
    std::map<lbann::data_field_type, std::vector<El::Int>> test_fields;
    std::map<lbann::data_field_type, lbann::CPUMat*> test_local_input_buffers;
    lbann::data_field_type bad_field = "bar";
    for (auto const& data_field : data_fields) {
      if (data_field != bad_field) {
        test_fields[data_field] = fields[data_field];
        test_local_input_buffers[data_field] = local_input_buffers[data_field];
      }
    }
    auto dr = std::make_unique<lbann::data_reader_synthetic>(num_samples,
                                                             test_fields,
                                                             false);
    dr->setup(io_thread_pool->get_num_threads(), io_thread_pool.get());
    dr->set_comm(&comm);
    dr->load();
    ds.set_mini_batch_size(num_samples);
    ds.set_last_mini_batch_size(num_samples);
    ds.set_initial_position();

    dr->fetch(test_local_input_buffers,
              indices_fetched,
              ds.get_position(),
              ds.get_sample_stride(),
              num_samples);

    // Check all of the results that were fetched.  Ensure that the
    // data fields are accessed in the same order that they are in the map
    for (El::Int j = 0; j < num_samples; j++) {
      for (auto const& data_field : data_fields) {
        auto& X = *(local_input_buffers[data_field]);
        if (data_field == bad_field) {
          for (El::Int i = 0; i < X.Height(); i++) {
            CHECK(X(i, j) == 0.0f);
          }
        }
        else {
          // Create a new normal distribution for each sample.  This ensures
          // that the behavior matches the implementation in the synthetic
          // data reader and handles the case of odd numbers of entries with a
          // normal distriubtion implementation. (Specifically that entries
          // for a normal distribution are generated in pairs.)
          std::normal_distribution<lbann::DataType> dist(float(0), float(1));
          for (El::Int i = 0; i < X.Height(); i++) {
            CHECK(X(i, j) == dist(ref_fast_generator));
          }
        }
      }
    }
  }

  SECTION("fetch arbitrary check has data field guard")
  {
    auto dr = std::make_unique<lbann::data_reader_synthetic>(num_samples,
                                                             fields,
                                                             false);
    dr->setup(io_thread_pool->get_num_threads(), io_thread_pool.get());
    dr->set_comm(&comm);
    dr->load();
    ds.set_mini_batch_size(num_samples);
    ds.set_last_mini_batch_size(num_samples);
    ds.set_initial_position();

    for (auto const& data_field : data_fields) {
      dr->set_has_data_field(data_field, false);
    }

    CHECK_THROWS(dr->fetch(local_input_buffers,
                           indices_fetched,
                           ds.get_position(),
                           ds.get_sample_stride(),
                           num_samples));

    // All data buffers should be empty since it will have thrown an exception
    for (El::Int j = 0; j < num_samples; j++) {
      for (auto const& data_field : data_fields) {
        auto& X = *(local_input_buffers[data_field]);
        for (El::Int i = 0; i < X.Height(); i++) {
          CHECK(X(i, j) == 0.0f);
        }
      }
    }
  }
}
