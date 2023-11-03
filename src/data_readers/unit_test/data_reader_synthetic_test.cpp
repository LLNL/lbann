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

#include "TestHelpers.hpp"
#include "lbann/proto/lbann.pb.h"
#include "lbann/proto/proto_common.hpp"
#include <google/protobuf/text_format.h>

#include <cstdlib>
#include <errno.h>
#include <string.h>

// #include "./data_reader_common_catch2.hpp"
#include "lbann/data_readers/data_reader_synthetic.hpp"
#include "lbann/data_readers/utils/input_data_type.hpp"
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
  bool fetch_data_field(lbann::data_reader_synthetic& dr,
                        lbann::data_field_type data_field,
                        lbann::CPUMat& X,
                        uint64_t data_id,
                        uint64_t mb_idx)
  {
    return dr.fetch_data_field(data_field, X, data_id, mb_idx);
  }
};

TEST_CASE("Synthetic data reader classification tests",
          "[data_reader][synthetic][classification]")
{
  // initialize stuff (boilerplate)
  lbann::init_random(42, 1);
  lbann::init_data_seq_random(42);

  DataReaderSyntheticWhiteboxTester white_box_tester;

  // Create a local copy of the RNG to check the synthetic data reader
  lbann::fast_rng_gen ref_fast_generator;
  ref_fast_generator.seed(lbann::hash_combine(42, 0));

  auto s = GENERATE(range(1, 11));
  El::Int num_samples = s;
  std::vector<El::Int> dims = {s, s};
  ;
  El::Int num_labels = s * 2;

  SECTION("fetch data and label")
  {
    auto dr = std::make_unique<lbann::data_reader_synthetic>(num_samples,
                                                             dims,
                                                             num_labels,
                                                             false);
    lbann::CPUMat X;
    X.Resize(dims[0] * dims[1], num_samples);
    lbann::CPUMat Y;
    Y.Resize(num_labels, num_samples);
    El::Zeros_seq(Y, num_labels, num_samples);

    auto io_rng = lbann::set_io_generators_local_index(0);
    for (auto j = 0; j < num_samples; j++) {
      white_box_tester.fetch_datum(*dr, X, 0, j);
      white_box_tester.fetch_label(*dr, Y, 0, j);
    }

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

TEST_CASE("Synthetic data reader regression tests",
          "[data_reader][synthetic][regression]")
{
  // initialize stuff (boilerplate)
  lbann::init_random(42, 1);
  lbann::init_data_seq_random(42);

  DataReaderSyntheticWhiteboxTester white_box_tester;

  // Create a local copy of the RNG to check the synthetic data reader
  lbann::fast_rng_gen ref_fast_generator;
  ref_fast_generator.seed(lbann::hash_combine(42, 0));

  auto s = GENERATE(range(1, 11));
  El::Int num_samples = s;
  std::vector<El::Int> dims = {s, s};
  ;
  std::vector<El::Int> response_dims = {s + 1, s + 1};

  SECTION("fetch data and response")
  {
    auto dr = std::make_unique<lbann::data_reader_synthetic>(num_samples,
                                                             dims,
                                                             response_dims,
                                                             false);

    lbann::CPUMat X;
    X.Resize(dims[0] * dims[1], num_samples);
    lbann::CPUMat Y;
    Y.Resize(response_dims[0] * response_dims[1], num_samples);

    auto io_rng = lbann::set_io_generators_local_index(0);
    for (El::Int i = 0; i < num_samples; i++) {
      white_box_tester.fetch_datum(*dr, X, 0, i);
      white_box_tester.fetch_response(*dr, Y, 0, i);
    }

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

TEST_CASE("Synthetic data reader data field",
          "[data_reader][synthetic][data_field]")
{
  // initialize stuff (boilerplate)
  lbann::init_random(42, 1);
  lbann::init_data_seq_random(42);

  DataReaderSyntheticWhiteboxTester white_box_tester;

  // Create a local copy of the RNG to check the synthetic data reader
  lbann::fast_rng_gen ref_fast_generator;
  ref_fast_generator.seed(lbann::hash_combine(42, 0));

  auto s = GENERATE(range(1, 4));
  El::Int num_samples = s;
  std::vector<lbann::data_field_type> data_fields = {"foo", "bar"};
  std::map<lbann::data_field_type, std::vector<El::Int>> fields;
  int f = 0;
  for (auto const& data_field : data_fields) {
    std::vector<El::Int> dims = {s + f, s + f};
    fields[data_field] = dims;
    ++f;
  }

  SECTION("fetch data field")
  {
    auto dr = std::make_unique<lbann::data_reader_synthetic>(num_samples,
                                                             fields,
                                                             false);
    lbann::CPUMat X;
    for (auto const& [data_field, dims] : fields) {
      X.Resize(dims[0] * dims[1], num_samples);

      auto io_rng = lbann::set_io_generators_local_index(0);
      for (El::Int j = 0; j < num_samples; j++) {
        white_box_tester.fetch_data_field(*dr, data_field, X, 0, j);
      }

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
      }
    }
    REQUIRE(white_box_tester.fetch_data_field(*dr, "foobar", X, 0, 0) == false);
  }
}
