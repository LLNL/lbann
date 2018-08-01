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

#ifndef _DATA_READER_JAG_CONDUIT_HDF5_HPP_
#define _DATA_READER_JAG_CONDUIT_HDF5_HPP_

#include "lbann_config.hpp" // may define LBANN_HAS_CONDUIT

#ifdef LBANN_HAS_CONDUIT
#include "lbann/data_readers/opencv.hpp"
#include "data_reader.hpp"
#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "lbann/data_readers/cv_process.hpp"
#include <string>
#include <set>
#include <unordered_map>

namespace lbann {

class jag_store;

/**
 * Loads the pairs of JAG simulation inputs and results from a conduit-wrapped hdf5 file
 */
class data_reader_jag_conduit_hdf5 : public generic_data_reader {
 public:
  using ch_t = float; ///< jag output image channel type
  using scalar_t = double; ///< jag scalar output type
  using input_t = double; ///< jag input parameter type

  /**
   * Dependent/indepdendent variable types
   * - JAG_Image: simulation output images
   * - JAG_Scalar: simulation output scalars
   * - JAG_Input: simulation input parameters
   * - Undefined: the default
   */
  enum variable_t {Undefined=0, JAG_Image, JAG_Scalar, JAG_Input};
  using TypeID = conduit::DataType::TypeID;

  data_reader_jag_conduit_hdf5(bool shuffle = true) = delete;
  data_reader_jag_conduit_hdf5(const std::shared_ptr<cv_process>& pp, bool shuffle = true);
  data_reader_jag_conduit_hdf5(const data_reader_jag_conduit_hdf5&);
  data_reader_jag_conduit_hdf5& operator=(const data_reader_jag_conduit_hdf5&);
  ~data_reader_jag_conduit_hdf5() override;
  data_reader_jag_conduit_hdf5* copy() const override { return new data_reader_jag_conduit_hdf5(*this); }

  std::string get_type() const override {
    return "data_reader_jag_conduit_hdf5";
  }

  /// Load data and do data reader's chores.
  void load() override;

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

  /// Return the total linearized size of data
  int get_linearized_data_size() const override;
  /// Return the total linearized size of response
  int get_linearized_response_size() const override;
  /// Return the per-source linearized sizes of composite data
  std::vector<size_t> get_linearized_data_sizes() const;
  /// Return the per-source linearized sizes of composite response
  std::vector<size_t> get_linearized_response_sizes() const;

  /// Return the dimension of data
  const std::vector<int> get_data_dims() const override;

  int get_num_labels() const override;
  int get_linearized_label_size() const override;

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

  template<typename S>
  static size_t add_val(const std::string key, const conduit::Node& n, std::vector<S>& vals);

  /// sets up a data_store.
  void setup_data_store(model *m) override;

  /// A untiliy function to convert the pointer to image data into an opencv image
  static cv::Mat cast_to_cvMat(const std::pair<size_t, const ch_t*> img, const int height);
  /// A utility function to convert a JAG variable type to name string
  static std::string to_string(const variable_t t);

  void set_image_dims(const int width, const int height, const int ch=1);

 protected:
  virtual void set_defaults();
  virtual bool replicate_processor(const cv_process& pp);
  virtual void copy_members(const data_reader_jag_conduit_hdf5& rhs);

  static std::string to_string(const std::vector<variable_t>& vec);


  virtual std::vector<CPUMat>
    create_datum_views(CPUMat& X, const std::vector<size_t>& sizes, const int mb_idx) const;

  bool fetch(CPUMat& X, int data_id, int mb_idx, int tid,
             const variable_t vt, const std::string tag);
  bool fetch_datum(CPUMat& X, int data_id, int mb_idx, int tid) override;
  bool fetch_response(CPUMat& Y, int data_id, int mb_idx, int tid) override;
  bool fetch_label(CPUMat& X, int data_id, int mb_idx, int tid) override;

#ifndef _JAG_OFFLINE_TOOL_MODE_
  /// Load a conduit-packed hdf5 data file
  void load_conduit(const std::string conduit_file_path);
#endif // _JAG_OFFLINE_TOOL_MODE_

  /// Check if the given sample id is valid
  bool check_sample_id(const size_t i) const;

  /// Choose the image closest to the bang time among those associated with the i-th sample
  std::vector<int> choose_image_near_bang_time(const size_t i) const;

  /// Obtain the pointers to read-only image data
  std::vector< std::pair<size_t, const ch_t*> > get_image_ptrs(const size_t i) const;

  jag_store * get_jag_store() const { return m_jag_store; }

 protected:

  int m_image_width; ///< image width
  int m_image_height; ///< image height
  int m_image_num_channels; ///< number of image channels

  /// Whether data have been loaded
  bool m_is_data_loaded;

  int m_num_labels; ///< number of labels

  /// Keys to select a set of scalar simulation outputs to use. By default, use all.
  std::vector<std::string> m_scalar_keys;
  /// Keys to select a set of simulation input parameters to use. By default, use all.
  std::vector<std::string> m_input_keys;

  /// preprocessor duplicated for each omp thread
  std::vector<std::unique_ptr<cv_process> > m_pps;

  /// jag_store; replaces m_data
  jag_store *m_jag_store;

  bool m_owns_jag_store;

  /**
   * Set of keys that are associated with non_numerical values.
   * Such a variable requires a specific method for mapping to a numeric value.
   * When a key is found in the set, the variable is ignored. Therefore,
   * when a conversion is defined for such a key, remove it from the set.
   */
  static const std::set<std::string> non_numeric_vars;

  /**
   * indicate if all the input variables are of the input_t type, in which case
   * we can rely on a data extraction method with lower overhead.
   */
  bool m_uniform_input_type;

  /**
   * maps integers to sample IDs. In the future the sample IDs may
   * not be integers; also, this map only includes sample IDs that
   * have <sample_id>/performance/success = 1
   */
  std::unordered_map<int, std::string> m_success_map;

  std::set<std::string> m_emi_selectors;
};



} // end of namespace lbann
#endif // LBANN_HAS_CONDUIT
#endif // _DATA_READER_JAG_CONDUIT_HDF5_HPP_
