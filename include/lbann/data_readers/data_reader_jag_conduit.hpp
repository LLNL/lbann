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

#ifndef _DATA_READER_JAG_CONDUIT_HPP_
#define _DATA_READER_JAG_CONDUIT_HPP_

#include "lbann_config.hpp" // may define LBANN_HAS_CONDUIT

#ifdef LBANN_HAS_CONDUIT
#include "lbann/data_readers/opencv.hpp"
#include "data_reader.hpp"
#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "lbann/data_readers/cv_process.hpp"

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
   * Dependent/indepdendent variable types
   * - JAG_Image: simulation output images
   * - JAG_Scalar: simulation output scalars
   * - JAG_Input: simulation input parameters
   * - Undefined: the default
   */
  enum variable_t {JAG_Image, JAG_Scalar, JAG_Input, Undefined};

  data_reader_jag_conduit(bool shuffle = true) = delete;
  data_reader_jag_conduit(const std::shared_ptr<cv_process>& pp, bool shuffle = true);
  data_reader_jag_conduit(const data_reader_jag_conduit&);
  data_reader_jag_conduit& operator=(const data_reader_jag_conduit&);
  ~data_reader_jag_conduit() override;
  data_reader_jag_conduit* copy() const override { return new data_reader_jag_conduit(*this); }

  std::string get_type() const override {
    return "data_reader_jag_conduit";
  }

  /// Choose which data to use for independent variable
  void set_independent_variable_type(const variable_t independent);
  /// Choose which data to use for dependent variable
  void set_dependent_variable_type(const variable_t dependent);

  /// Tell which data to use for independent variable
  variable_t get_independent_variable_type() const;
  /// Tell which data to use for dependent variable
  variable_t get_dependent_variable_type() const;

  /// Set the image dimension
  void set_image_dims(const int width, const int height, const int ch = 1);

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
  unsigned int get_num_img_srcs() const;
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
   * There is one image per view, each of which is taken at closest to the bang time.
   */
  std::vector<ch_t> get_images(const size_t i) const;

  /// Return the scalar simulation output data of the i-th sample
  std::vector<scalar_t> get_scalars(const size_t i) const;

  /// Return the simulation input parameters of the i-th sample
  std::vector<input_t> get_inputs(const size_t i) const;

  /// Check if the simulation was successful
  int check_exp_success(const size_t sample_id) const;

  void save_image(Mat& pixels, const std::string filename, bool do_scale = true) override;

#ifndef _JAG_OFFLINE_TOOL_MODE_
  /// sets up a data_store.
  void setup_data_store(model *m) override;
#endif // _JAG_OFFLINE_TOOL_MODE_

  static cv::Mat cast_to_cvMat(const std::pair<size_t, const ch_t*> img, const int height);

 protected:
  virtual void set_defaults();
  virtual bool replicate_processor(const cv_process& pp);
  virtual void copy_members(const data_reader_jag_conduit& rhs);

  virtual std::vector<::Mat> create_datum_views(::Mat& X, const int mb_idx) const;

  bool fetch(Mat& X, int data_id, int mb_idx, int tid,
             const variable_t vt, const std::string tag);
  bool fetch_datum(Mat& X, int data_id, int mb_idx, int tid) override;
  bool fetch_response(Mat& Y, int data_id, int mb_idx, int tid) override;

#ifndef _JAG_OFFLINE_TOOL_MODE_
  /// Load a conduit-packed hdf5 data file
  void load_conduit(const std::string conduit_file_path);
#endif // _JAG_OFFLINE_TOOL_MODE_

  /// Obtain the number of image measurement views
  void set_num_img_srcs();
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

  /// Choose the image closest to the bang time among those associated with the i-th sample
  std::vector<int> choose_image_near_bang_time(const size_t i) const;

  /// Allow const access to the conduit data structure
  const conduit::Node& get_conduit_node(const std::string key) const;

  /// Obtain the pointers to read-only image data
  std::vector< std::pair<size_t, const ch_t*> > get_image_ptrs(const size_t i) const;

 protected:
  /// independent variable type
  variable_t m_independent;
  /// dependent variable type
  variable_t m_dependent;

  int m_image_width; ///< image width
  int m_image_height; ///< image height
  int m_image_num_channels; ///< number of image channels
  size_t m_image_linearized_size; ///< The linearized size of an image
  unsigned int m_num_img_srcs; ///< number of views result in images

  /// Whether data have been loaded
  bool m_is_data_loaded;

  /// Keys to select a set of scalar simulation outputs to use. By default, use all.
  std::vector<std::string> m_scalar_keys;
  /// Keys to select a set of simulation input parameters to use. By default, use all.
  std::vector<std::string> m_input_keys;

  /// preprocessor duplicated for each omp thread
  std::vector<std::unique_ptr<cv_process> > m_pps;

  /// data wrapped in a conduit structure
  conduit::Node m_data;
};

} // end of namespace lbann
#endif // LBANN_HAS_CONDUIT
#endif // _DATA_READER_JAG_CONDUIT_HPP_
