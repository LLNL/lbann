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
//
// data_reader_mesh .hpp .cpp - data reader for mesh data
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_MESH_HPP
#define LBANN_DATA_READER_MESH_HPP

#include "data_reader.hpp"

namespace lbann {

/**
 * Data reader for reading dumped mesh images.
 * Provide the directory containing all the channel subdirectories.
 * This assumes the data is stored as floats in row-major order.
 * The channels to load are currently hardcoded. This only supports regression.
 */
class mesh_reader : public generic_data_reader {
 public:
  mesh_reader(bool shuffle = true);
  ~mesh_reader() override {}

  mesh_reader* copy() const override { return new mesh_reader(*this); }
  std::string get_type() const override { return "mesh_reader"; }

  /// Set a suffix to append to the channel directories.
  void set_suffix(const std::string suffix) { m_suffix = suffix; }
  /// Set the shape (height and width) of the data.
  void set_data_shape(int height, int width) {
    m_data_height = height;
    m_data_width = width;
  }
  /// Set the index length for filenames.
  void set_index_length(int l) {
    m_index_length = l;
  }
  /// Set whether to do random horizontal and vertical flips.
  void set_random_flips(bool b) { m_random_flips = b; }

  void load() override;
  int get_linearized_data_size() const override {
    return m_channels.size() * m_data_height * m_data_width;
  }
  int get_linearized_response_size() const override {
    return m_data_height * m_data_width;
  }
  const std::vector<int> get_data_dims() const override {
    return {static_cast<int>(m_channels.size()),
        m_data_height,
        m_data_width};
  }
 protected:
  bool fetch_datum(CPUMat& X, int data_id, int mb_idx) override;
  bool fetch_response(CPUMat& Y, int data_id, int mb_idx) override;

  /**
   * Load filename into mat.
   * This may do datatype conversion if DataType is not float.
   * mat should be of size (m_data_height, m_data_width).
   */
  void load_file(int data_id, const std::string channel, Mat& mat);
  /// Return the full path to the data file for datum data_id's channel.
  std::string construct_filename(std::string channel, int data_id);

  /// Flip mat horizontally (i.e. about its vertical axis).
  void horizontal_flip(CPUMat& mat);
  /// Flip mat vertically (i.e. about its horizontal axis).
  void vertical_flip(CPUMat& mat);

  /// A suffix to append to each channel directory (e.g. "128").
  std::string m_suffix = "128";
  /// Target channel; contains the relaxation information.
  std::string m_target_name = "mask";
  /// Names of each channel to load as data.
  std::vector<std::string> m_channels = {
    "Density",
    "Pressure",
    "VectorComp_AvgVelocity_R",
    "VolumeFractions_bubble",
    "aspectRatio",
    "conditionNumber",
    "distortion",
    "jacobian",
    "largestAngle",
    "oddy",
    "scaledJacobian",
    "shape",
    //"shapeAndSize",
    "shear",
    //"shearAndSize",
    "skew",
    "smallestAngle",
    "stretch",
    "taper",
    "volume"
  };
  /**
   * Character length of the index in filenames.
   * Indices will be left-padded with zeros to this length.
   */
  int m_index_length = 4;
  /// Format string for the index.
  std::string m_index_format_str;
  /// X dimension of the mesh data.
  int m_data_height = 128;
  /// Y dimension of the mesh data.
  int m_data_width = 128;
  /// Number of samples.
  int m_num_samples = 0;
  /// Buffers for loading data into.
  std::vector<std::vector<DataType>> m_load_bufs;
  /// Whether to do random horizontal/vertical flips.
  bool m_random_flips = false;
  /**
   * This records the flip choices made for each sample.
   * This is needed because both the data and target need the same
   * transformation applied.
   */
  std::vector<std::pair<bool, bool>> m_flip_choices;
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_MESH_HPP
