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

#ifndef __DATA_READER_TEST_COMMON_HDF5_TEST_UTILS_HPP__
#define __DATA_READER_TEST_COMMON_HDF5_TEST_UTILS_HPP__

#include "lbann/data_coordinator/data_packer.hpp"
#include "lbann/data_ingestion/readers/data_reader_HDF5.hpp"

#include <conduit/conduit.hpp>

class DataReaderHDF5WhiteboxTester
{
public:
  void normalize(lbann::hdf5_data_reader& x,
                 conduit::Node& node,
                 const std::string& path,
                 const conduit::Node& metadata)
  {
    x.normalize(node, path, metadata);
  }
  void repack_image(lbann::hdf5_data_reader& x,
                    conduit::Node& node,
                    const std::string& path,
                    const conduit::Node& metadata)
  {
    x.repack_image(node, path, metadata);
  }

  void pack(lbann::hdf5_data_reader& x, conduit::Node& node, size_t index)
  {
    x.pack(node, index);
  }

  void parse_schemas(lbann::hdf5_data_reader& x) { return x.parse_schemas(); }

  conduit::Node& get_data_schema(lbann::hdf5_data_reader& x)
  {
    return x.m_data_schema;
  }

  conduit::Node& get_experiment_schema(lbann::hdf5_data_reader& x)
  {
    return x.m_experiment_schema;
  }

  void set_data_schema(lbann::hdf5_data_reader& x, const conduit::Node& s)
  {
    x.set_data_schema(s);
  }

  void set_experiment_schema(lbann::hdf5_data_reader& x, const conduit::Node& s)
  {
    x.set_experiment_schema(s);
  }

  void load_sample(lbann::hdf5_data_reader& x,
                   conduit::Node& node,
                   hid_t file_handle,
                   const std::string& sample_name)
  {
    return x.load_sample(node, file_handle, sample_name);
  }

  void print_metadata(lbann::hdf5_data_reader& x, std::ostream& os = std::cout)
  {
    x.print_metadata(os);
  }

  void set_delete_packed_fields(lbann::hdf5_data_reader& x, const bool flag)
  {
    x.set_delete_packed_fields(flag);
  }

  bool fetch_data_field(lbann::hdf5_data_reader& dr,
                        lbann::data_field_type data_field,
                        lbann::CPUMat& X,
                        uint64_t data_id,
                        uint64_t mb_idx)
  {
    conduit::Node sample;
    dr.fetch_conduit_node(sample, data_id);
    return lbann::data_packer::extract_data_field_from_sample(data_field,
                                                              sample,
                                                              X,
                                                              mb_idx);
  }

  bool fetch_datum(lbann::hdf5_data_reader& dr,
                   lbann::CPUMat& X,
                   uint64_t data_id,
                   uint64_t mb_idx)
  {
    conduit::Node sample;
    dr.fetch_conduit_node(sample, data_id);
    return lbann::data_packer::extract_data_field_from_sample(
      INPUT_DATA_TYPE_SAMPLES,
      sample,
      X,
      mb_idx);
  }

  bool fetch_response(lbann::hdf5_data_reader& dr,
                      lbann::CPUMat& X,
                      uint64_t data_id,
                      uint64_t mb_idx)
  {
    conduit::Node sample;
    dr.fetch_conduit_node(sample, data_id);
    return lbann::data_packer::extract_data_field_from_sample(
      INPUT_DATA_TYPE_RESPONSES,
      sample,
      X,
      mb_idx);
  }

  bool fetch_label(lbann::hdf5_data_reader& dr,
                   lbann::CPUMat& X,
                   uint64_t data_id,
                   uint64_t mb_idx)
  {
    conduit::Node sample;
    dr.fetch_conduit_node(sample, data_id);
    return lbann::data_packer::extract_data_field_from_sample(
      INPUT_DATA_TYPE_LABELS,
      sample,
      X,
      mb_idx);
  }

  int get_linearized_size(lbann::hdf5_data_reader& dr,
                          lbann::data_field_type const& data_field)
  {
    return dr.get_linearized_size(data_field);
  }

  void construct_linearized_size_lookup_tables(lbann::hdf5_data_reader& dr,
                                               conduit::Node& node)
  {
    return dr.construct_linearized_size_lookup_tables(node);
  }
};

void check_node_fields(conduit::Node const& ref_node,
                       conduit::Node const& test_node,
                       conduit::Node const& data_schema,
                       std::vector<std::string> fields,
                       const std::string original_path,
                       const std::string new_pathname);

#endif // __DATA_READER_TEST_COMMON_HDF5_TEST_UTILS_HPP__
