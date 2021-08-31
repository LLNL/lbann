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

#include <catch2/catch.hpp>

#include "MPITestHelpers.hpp"
#include "TestHelpers.hpp"
#include "lbann/proto/proto_common.hpp"
#include <google/protobuf/text_format.h>
#include <lbann.pb.h>

#include <cstdlib>
#include <errno.h>
#include <string.h>

//#include "./data_reader_common_catch2.hpp"
#include "lbann/data_readers/data_reader_synthetic.hpp"
#include "lbann/data_readers/utils/input_data_type.hpp"
#include "lbann/utils/threads/thread_pool.hpp"
#include "lbann/utils/threads/thread_utils.hpp"
#include "lbann/utils/hash.hpp"

class DataReaderSyntheticWhiteboxTester
{
public:
  bool fetch_datum(lbann::data_reader_synthetic& dr, lbann::CPUMat& X, int data_id, int mb_idx) {
    return dr.fetch_datum(X, data_id, mb_idx);
  }
  bool fetch_label(lbann::data_reader_synthetic& dr, lbann::CPUMat& Y, int data_id, int mb_idx) {
    return dr.fetch_label(Y, data_id, mb_idx);
  }
  bool fetch_response(lbann::data_reader_synthetic& dr, lbann::CPUMat& Y, int data_id, int mb_idx) {
    return dr.fetch_response(Y, data_id, mb_idx);
  }
};

TEST_CASE("Synthetic data reader public API classification tests",
          "[mpi][data_reader][synthetic][classification][public]")
{
  // initialize stuff (boilerplate)
  auto& comm = unit_test::utilities::current_world_comm();
  lbann::init_random(42, 1);
  lbann::init_data_seq_random(42);

  // Create a local copy of the RNG to check the synthetic data reader
  lbann::fast_rng_gen ref_fast_generator;
  ref_fast_generator.seed(lbann::hash_combine(42, 0));

  // Initalize a per-trainer I/O thread pool
  auto io_thread_pool = lbann::make_unique<lbann::thread_pool>();
  io_thread_pool->launch_pinned_threads(1, 1);

  std::vector<std::string> active_data_fields = {"samples", "labels"};
  for(auto s = 1; s <= 10; s++) {
    El::Int num_samples = s;
    std::vector<int> dims = {s,s};;
    El::Int num_labels = s*2;

    std::map<lbann::input_data_type, std::unique_ptr<lbann::CPUMat>> owning_local_input_buffers;
    std::map<lbann::input_data_type, lbann::CPUMat*> local_input_buffers;
    for (auto& data_field : active_data_fields) {
      lbann::input_data_type data_field_hack;
      auto local_mat = std::make_unique<lbann::CPUMat>();
      if (data_field == INPUT_DATA_TYPE_SAMPLES) {
        data_field_hack = lbann::input_data_type::SAMPLES;
        local_mat->Resize(dims[0] * dims[1], num_samples);
        El::Zeros_seq(*local_mat, dims[0] * dims[1], num_samples);
      }
      else if (data_field == INPUT_DATA_TYPE_LABELS) {
        data_field_hack = lbann::input_data_type::LABELS;
        local_mat->Resize(num_labels, num_samples);
        El::Zeros_seq(*local_mat, num_labels, num_samples);
      }
      local_input_buffers[data_field_hack] = local_mat.get();
      owning_local_input_buffers[data_field_hack] = std::move(local_mat);
    }
    El::Matrix<El::Int> indices_fetched;
    El::Zeros_seq(indices_fetched, num_samples, 1);

    SECTION("fetch data and label s=" + std::to_string(s))
    {
      auto dr = std::make_unique<lbann::data_reader_synthetic>(
        num_samples,
        dims,
        num_labels,
        false);
      dr->setup(io_thread_pool->get_num_threads(), io_thread_pool.get());
      dr->set_rank(0);
      dr->set_comm(&comm);
      dr->set_num_parallel_readers(1);
      dr->load();
      dr->set_mini_batch_size(num_samples);
      dr->set_last_mini_batch_size(num_samples);
      dr->set_initial_position();

      dr->fetch(local_input_buffers, indices_fetched);

      // for (auto& [field, buf] : local_input_buffers) {
      //   std::cout << "For field " << to_string(field) << std::endl;
      //   El::Print(*buf);
      // }

      auto& X = *(local_input_buffers[lbann::input_data_type::SAMPLES]);
      auto& Y = *(local_input_buffers[lbann::input_data_type::LABELS]);
      CHECK(X.Width() == Y.Width());

      for (El::Int j = 0; j < num_samples; j++) {
        // Create a new normal distribution for each sample.  This ensures
        // that the behavior matches the implementation in the synthetic data
        // reader and handles the case of odd numbers of entries with a normal
        // distriubtion implementation. (Specifically that entries for a
        // normal distribution are generated in pairs.)
        std::normal_distribution<lbann::DataType> dist(float(0), float(1));
        for (El::Int i = 0; i < X.Height(); i++) {
          CHECK(X(i, j) == dist(ref_fast_generator));
        }
        auto index = lbann::fast_rand_int(ref_fast_generator, num_labels);
        // std::cout << "Here is the reference value " << index << std::endl;
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

TEST_CASE("Synthetic data reader public API regression tests",
          "[mpi][data_reader][synthetic][regression][public]")
{
  // initialize stuff (boilerplate)
  auto& comm = unit_test::utilities::current_world_comm();
  lbann::init_random(42, 1);
  lbann::init_data_seq_random(42);

  // Create a local copy of the RNG to check the synthetic data reader
  lbann::fast_rng_gen ref_fast_generator;
  ref_fast_generator.seed(lbann::hash_combine(42, 0));

  // Initalize a per-trainer I/O thread pool
  auto io_thread_pool = lbann::make_unique<lbann::thread_pool>();
  io_thread_pool->launch_pinned_threads(1, 1);

  std::vector<std::string> active_data_fields = {"samples", "responses"};
  for(auto s = 1; s <= 10; s++) {
    El::Int num_samples = s;
    std::vector<int> dims = {s,s};;
    std::vector<int> response_dims = {s+1, s+1};

    SECTION("fetch data and response s=" + std::to_string(s))
    {
      std::map<lbann::input_data_type, std::unique_ptr<lbann::CPUMat>> owning_local_input_buffers;
      std::map<lbann::input_data_type, lbann::CPUMat*> local_input_buffers;
      for (auto& data_field : active_data_fields) {
        lbann::input_data_type data_field_hack;
        auto local_mat = std::make_unique<lbann::CPUMat>();
        if (data_field == INPUT_DATA_TYPE_SAMPLES) {
          data_field_hack = lbann::input_data_type::SAMPLES;
          local_mat->Resize(dims[0] * dims[1], num_samples);
          El::Zeros_seq(*local_mat, dims[0] * dims[1], num_samples);
        }
        else if (data_field == INPUT_DATA_TYPE_RESPONSES) {
          data_field_hack = lbann::input_data_type::RESPONSES;
          local_mat->Resize(response_dims[0] * response_dims[1], num_samples);
        }
        local_input_buffers[data_field_hack] = local_mat.get();
        owning_local_input_buffers[data_field_hack] = std::move(local_mat);
      }
      El::Matrix<El::Int> indices_fetched;
      El::Zeros_seq(indices_fetched, num_samples, 1);

      auto dr = std::make_unique<lbann::data_reader_synthetic>(num_samples,
                                                               dims,
                                                               response_dims,
                                                               false);
      dr->setup(io_thread_pool->get_num_threads(), io_thread_pool.get());
      dr->set_rank(0);
      dr->set_comm(&comm);
      dr->set_num_parallel_readers(1);
      dr->load();
      dr->set_mini_batch_size(num_samples);
      dr->set_last_mini_batch_size(num_samples);
      dr->set_initial_position();

      dr->fetch(local_input_buffers, indices_fetched);

      // for (auto& [field, buf] : local_input_buffers) {
      //   std::cout << "For field " << to_string(field) << std::endl;
      //   El::Print(*buf);
      // }

      auto& X = *(local_input_buffers[lbann::input_data_type::SAMPLES]);
      auto& Y = *(local_input_buffers[lbann::input_data_type::RESPONSES]);
      CHECK(X.Width() == Y.Width());

      for (El::Int j = 0; j < num_samples; j++) {
        {
          // Create a new normal distribution for each sample.  This ensures
          // that the behavior matches the implementation in the synthetic data
          // reader and handles the case of odd numbers of entries with a normal
          // distriubtion implementation. (Specifically that entries for a
          // normal distribution are generated in pairs.)
          std::normal_distribution<lbann::DataType> dist(float(0), float(1));
          for (El::Int i = 0; i < X.Height(); i++) {
            CHECK(X(i, j) == dist(ref_fast_generator));
          }
        }
        {
          // Create a new normal distribution for each sample.  This ensures
          // that the behavior matches the implementation in the synthetic data
          // reader and handles the case of odd numbers of entries with a normal
          // distriubtion implementation. (Specifically that entries for a
          // normal distribution are generated in pairs.)
          std::normal_distribution<lbann::DataType> dist(float(0), float(1));
          for (El::Int i = 0; i < Y.Height(); i++) {
            CHECK(Y(i, j) == dist(ref_fast_generator));
          }
        }
      }
    }
  }
}
