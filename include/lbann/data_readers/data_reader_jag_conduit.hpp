////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#ifndef _DATA_READERLBANN_HAS_CONDUITHPP_
#define _DATA_READERLBANN_HAS_CONDUITHPP_

#ifdef LBANN_HAS_CONDUIT
#include "lbann/data_readers/opencv.hpp"
#include "data_reader.hpp"
#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"

namespace lbann {

/**
 * Loads the pairs of JAG simulation inputs and results from a conduit-wrapped hdf5 file
 */
class data_reader_jag_conduit : public generic_data_reader {
 public:
  using ch_t = double; ///< jag output image channel type
  using scalar_t = double; ///< jag scalar output type
  using input_t = double; ///< jag input parameter type

  /**
   * Mode of modeling.
   * - Inverse: image to input param
   * - AutoI: image to image
   * - AutoS: scalar to scalar  
   */
  enum model_mode_t {Inverse, AutoI, AutoS};

  data_reader_jag_conduit(bool shuffle = true);
  data_reader_jag_conduit(const data_reader_jag_conduit&) = default;
  data_reader_jag_conduit& operator=(const data_reader_jag_conduit&) = default;
  ~data_reader_jag_conduit() override;
  data_reader_jag_conduit* copy() const override { return new data_reader_jag_conduit(*this); }

  std::string get_type() const override {
    return "data_reader_jag_conduit";
  }

  /// Set the modeling mode: Inverse, AutoI, or AutoS
  void set_model_mode(const model_mode_t mm);

  /// Set the image dimension
  void set_image_dims(const int width, const int height);

  /// Select the set of scalar output variables to use
  void set_scalar_choices(const std::vector<std::string>& keys);
  /// Set to use the entire set of scalar outputs
  void set_all_scalar_choices();
  /// Report the selected scalar outputs
  const std::vector<std::string>& get_scalar_choices() const;

  /// Select the set of simulation input parameters to use
  void set_input_choices(const std::vector<std::string>& keys);
  /// Set to use the entire set of simulation input parameters
  void set_all_input_choices();
  /// Report the selected simulation input parameters
  const std::vector<std::string>& get_input_choices() const;

  /// Load data and do data reader's chores.
#ifndef _JAG_OFFLINE_TOOL_MODE_
  void load() override;
#else
  void load_conduit(const std::string conduit_file_path);
#endif // _JAG_OFFLINE_TOOL_MODE_

  /// Return the number of samples
  size_t get_num_samples() const;

  /// Return the number of measurement views
  unsigned int get_num_views() const;
  /// Return the linearized size of an image
  size_t get_linearized_image_size() const;
  /// Return the linearized size of scalar outputs
  size_t get_linearized_scalar_size() const;
  /// Return the linearized size of inputs
  size_t get_linearized_input_size() const;

  /// Return the linearized size of data of the current modeling mode
  int get_linearized_data_size() const override;
  /// Return the linearized size of response of the current modeling mode
  int get_linearized_response_size() const override;
  /// Return the data dimension of the current modeling mode
  const std::vector<int> get_data_dims() const override;

  /// Show the description
  std::string get_description() const;

  /// Return the image simulation output of the i-th sample
  std::vector<cv::Mat> get_cv_images(const size_t i) const;

  /**
   * Return the images of the i-th sample as an 1-D vector of lbann::DataType
   * There is one image per view, each of which is taken at closet to the bang time.
   */
  std::vector<ch_t> get_images(const size_t i) const;

  /// Return the scalar simulatino output data of the i-th sample
  std::vector<scalar_t> get_scalars(const size_t i) const;

  /// Return the simulation input parameters of the i-th sample
  std::vector<input_t> get_inputs(const size_t i) const;

  /// Check if the simulation was successful
  int check_exp_success(const size_t sample_id) const;

  void save_image(Mat& pixels, const std::string filename, bool do_scale = true) override;

 protected:
  bool fetch_datum(Mat& X, int data_id, int mb_idx, int tid) override;
  bool fetch_response(Mat& Y, int data_id, int mb_idx, int tid) override;

#ifndef _JAG_OFFLINE_TOOL_MODE_
  /// Load a conduit-packed hdf5 data file
  void load_conduit(const std::string conduit_file_path);
#endif // _JAG_OFFLINE_TOOL_MODE_

  /// Obtain the number of image measurement views
  void set_num_views();
  /// Obtain the linearized size of images of a sample from the meta info
  void set_linearized_image_size();
  /// See if the image size is consistent with the linearized size
  void check_image_size();
  /// Make sure that the keys to choose scalar outputs are valid
  void check_scalar_keys();
  /// Make sure that the keys to choose scalar outputs are valid
  void check_input_keys();

  /// Check if the given sample id is valid
  bool check_sample_id(const size_t i) const;

  /// Choose the image closet to the bang time among those associated with the i-th sample
  std::vector<int> choose_image_near_bang_time(const size_t i) const;

  /// Allow const access to the conduit data structure
  const conduit::Node& get_conduit_node(const std::string key) const;

  /// Obtain the pointers to read-only image data
  std::vector< std::pair<size_t, const ch_t*> > get_image_ptrs(const size_t i) const;

 protected:
  /// The current mode of modeling
  model_mode_t m_model_mode;

  /// The linearized size of an image
  size_t m_linearized_image_size;

  unsigned int m_num_views; ///< number of views result in images
  int m_image_width; ///< image width
  int m_image_height; ///< image height

  /// keys to select a set of scalar simulation outputs to use
  std::vector<std::string> m_scalar_keys;
  /// keys to select a set of simulation input parameters to use
  std::vector<std::string> m_input_keys;

  /// Whether data have been loaded
  bool m_is_data_loaded;

  /// data wrapped in a conduit structure
  conduit::Node m_data;
};

} // end of namespace lbann
#endif // LBANN_HAS_CONDUIT
#endif // _DATA_READERLBANN_HAS_CONDUITHPP_
