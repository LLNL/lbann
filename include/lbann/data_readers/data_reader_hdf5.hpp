// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// // may not use this file except in compliance with the License.  You may
// // obtain a copy of the License at:
// //
// // http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// // implied. See the License for the specific language governing
// // permissions and limitations under the license.
// //
// ////////////////////////////////////////////////////////////////////////////////
//
//
#ifndef LBANN_DATA_READER_HDF5_HPP
#define LBANN_DATA_READER_HDF5_HPP
#include "data_reader_image.hpp"
#include "hdf5.h"

namespace lbann {
/**
 * Data reader for data stored in hdf5 files will need to assume the file contains x
 */
class hdf5_reader : public generic_data_reader {
 public:
  hdf5_reader(const bool shuffle);
  hdf5_reader* copy() const override { return new hdf5_reader(*this); }

  std::string get_type() const override {
    return "data_reader_hdf5_images";
  }
  //void set_input_params(int width, int height, int depth, int num_ch, int num_labels);
  void load() override;
  void set_hdf5_paths(const std::vector<std::string> hdf5_paths) {m_file_paths = hdf5_paths;}
  void set_scaling_factor_int16(DataType s) {m_scaling_factor_int16 = s;}

  int get_num_responses() const override {
    return get_linearized_response_size();
  }
  int get_linearized_data_size() const override {
    return m_num_features;
  }
  int get_linearized_response_size() const override {
    return m_num_response_features;
  }
  const std::vector<int> get_data_dims() const override {
    return m_data_dims;
  }
 protected:
  void read_hdf5(hsize_t h_data, hsize_t filespace, int rank, std::string key, hsize_t* dims, DataType * data_out);
  //void set_defaults() override;
  bool fetch_datum(CPUMat& X, int data_id, int mb_idx) override;
  bool fetch_label(CPUMat& Y, int data_id, int mb_idx) override;
  bool fetch_response(CPUMat& Y, int data_id, int mb_idx) override;
  /// Whether to fetch a label from the last column.
  bool m_has_labels = false;
  /// Whether to fetch a response from the last column.
  bool m_has_responses = true;
  int m_image_depth=0;
  int m_num_features;
  int m_num_response_features = 4;
  float m_all_responses[4];
  DataType m_scaling_factor_int16 = 1.0;
  std::vector<std::string> m_file_paths;
  MPI_Comm m_comm;
  std::vector<int> m_data_dims;
 private:
  static const std::string HDF5_KEY_DATA, HDF5_KEY_LABELS, HDF5_KEY_RESPONSES;
};
}
#endif
