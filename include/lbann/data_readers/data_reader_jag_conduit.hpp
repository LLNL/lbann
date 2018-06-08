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
#include <string>
#include <set>

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
  enum variable_t {Undefined=0, JAG_Image, JAG_Scalar, JAG_Input};
  using TypeID = conduit::DataType::TypeID;

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
  void set_independent_variable_type(const std::vector<variable_t> independent);
  /// Choose which data to use for dependent variable
  void set_dependent_variable_type(const std::vector<variable_t> dependent);

  /// Tell which data to use for independent variable
  std::vector<variable_t> get_independent_variable_type() const;
  /// Tell which data to use for dependent variable
  std::vector<variable_t> get_dependent_variable_type() const;

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

  /// Check if the simulation was successful
  int check_exp_success(const size_t sample_id) const;

  void save_image(Mat& pixels, const std::string filename, bool do_scale = true) override;

#ifndef _JAG_OFFLINE_TOOL_MODE_
  /// sets up a data_store.
  void setup_data_store(model *m) override;
#endif // _JAG_OFFLINE_TOOL_MODE_

  /// A untiliy function to convert the pointer to image data into an opencv image
  static cv::Mat cast_to_cvMat(const std::pair<size_t, const ch_t*> img, const int height);
  /// A utility function to convert a JAG variable type to name string
  static std::string to_string(const variable_t t);

 protected:
  virtual void set_defaults();
  virtual bool replicate_processor(const cv_process& pp);
  virtual void copy_members(const data_reader_jag_conduit& rhs);

  /// add data type for independent variable
  void add_independent_variable_type(const variable_t independent);
  /// add data type for dependent variable
  void add_dependent_variable_type(const variable_t dependent);

  /// Return the linearized size of a particular JAG variable type
  size_t get_linearized_size(const variable_t t) const;
  /// Return the dimension of a particular JAG variable type
  const std::vector<int> get_dims(const variable_t t) const;
  /// A utility function to make a string to show all the variable types in a vector
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

  /**
   * Check if the key is associated with non-numeric value, that is not and
   * cannot be converted to a numertic.
   */
  static bool check_non_numeric(const std::string key);

  /// Choose the image closest to the bang time among those associated with the i-th sample
  std::vector<int> choose_image_near_bang_time(const size_t i) const;

  /// Allow const access to the conduit data structure
  const conduit::Node& get_conduit_node(const std::string key) const;

  /// Obtain the pointers to read-only image data
  std::vector< std::pair<size_t, const ch_t*> > get_image_ptrs(const size_t i) const;

 protected:
  /// independent variable type
  std::vector<variable_t> m_independent;
  /// dependent variable type
  std::vector<variable_t> m_dependent;

  int m_image_width; ///< image width
  int m_image_height; ///< image height
  int m_image_num_channels; ///< number of image channels
  size_t m_image_linearized_size; ///< The linearized size of an image
  unsigned int m_num_img_srcs; ///< number of views result in images

  /// Whether data have been loaded
  bool m_is_data_loaded;

  int m_num_labels; ///< number of labels

  /// Keys to select a set of scalar simulation outputs to use. By default, use all.
  std::vector<std::string> m_scalar_keys;
  /// Keys to select a set of simulation input parameters to use. By default, use all.
  std::vector<std::string> m_input_keys;

  /// preprocessor duplicated for each omp thread
  std::vector<std::unique_ptr<cv_process> > m_pps;

  /// data wrapped in a conduit structure
  conduit::Node m_data;

  /**
   * Set of keys that are associated with non_numerical values.
   * Such a variable requires a specific method for mapping to a numeric value.
   * When a key is found in the set, the variable is ignored. Therefore,
   * when a conversion is defined for such a key, remove it from the set.
   */
  static const std::set<std::string> non_numeric_vars;
};


template<typename S>
inline size_t data_reader_jag_conduit::add_val(const std::string key, const conduit::Node& n, std::vector<S>& vals) {
  size_t cnt = 0u;

  switch (n.dtype().id()) {
    case TypeID::OBJECT_ID: {
        //std::cout << "O " << n.path() << std::endl;
        if (check_non_numeric(key)) {
          return 0u;
        }
        conduit::NodeConstIterator itr = n.children();
        while (itr.has_next()) {
          const conduit::Node& n_child = itr.next();
          cnt += add_val(itr.name(), n_child, vals);
        }
      }
      break;
    case TypeID::LIST_ID: {
        //std::cout << "L " << n.path() << std::endl;
        if (check_non_numeric(key)) {
          return 0u;
        }
        conduit::NodeConstIterator itr = n.children();
        while (itr.has_next()) {
          const conduit::Node& n_child = itr.next();
          cnt += add_val(itr.name(), n_child, vals);
        }
      }
      break;
    case TypeID::INT8_ID:
    case TypeID::INT16_ID:
    case TypeID::INT32_ID:
    case TypeID::INT64_ID:
    case TypeID::UINT8_ID:
    case TypeID::UINT16_ID:
    case TypeID::UINT32_ID:
    case TypeID::UINT64_ID:
    case TypeID::FLOAT32_ID:
    case TypeID::FLOAT64_ID:
      cnt = 1u;
      //std::cout << "N " << n.path() << ": " << static_cast<S>(n.to_value()) << std::endl;
      vals.push_back(static_cast<S>(n.to_value()));
      break;
    case TypeID::CHAR8_STR_ID: {
        // In case of a charater string, the method to convert it to a float number is specific to each key
        if (check_non_numeric(key)) {
          return 0u;
        //} else if (key == "some_key_with_non_numeric_values_that_can_be_converted_to_numerics_in_a_specific_way") {
        } else {
          const char* c_str = n.as_char8_str();
          // make sure that the std::string does not contain null character
          const std::string str
            = ((c_str == nullptr)? std::string() : std::string(c_str, n.dtype().number_of_elements())).c_str();

          cnt = 1u;
          const S v = static_cast<S>(atof(str.c_str()));
          vals.push_back(v);
          //std::cout << "S " << n.path() << ": " << str << " => " << vals.back() << std::endl;
        }
      }
      break;
    case TypeID::EMPTY_ID:
    default:
      std::string err = std::string("data_reader_jag_conduit::add_val() : invalid dtype (")
                      + n.dtype().name() + ") for " + n.path() + '.';
     #if 1
      std::cerr << err << " Skipping for now." << std::endl;
     #else
      throw lbann_exception(err);
     #endif
  }
  return cnt;
}

} // end of namespace lbann
#endif // LBANN_HAS_CONDUIT
#endif // _DATA_READER_JAG_CONDUIT_HPP_
