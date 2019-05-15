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

#ifndef _DATA_READER_JAG_CONDUIT_HPP_
#define _DATA_READER_JAG_CONDUIT_HPP_

#include "lbann_config.hpp" // may define LBANN_HAS_CONDUIT

#ifdef LBANN_HAS_CONDUIT
#include "lbann/data_readers/opencv.hpp"
#include "data_reader.hpp"
#include "conduit/conduit.hpp"
#include "hdf5.h"
#include "lbann/data_readers/cv_process.hpp"
#include <string>
#include <set>
#include <unordered_map>
#include <map>
#include <memory>
#include "lbann/data_readers/sample_list_jag.hpp"

namespace lbann {

class data_store_conduit;

/**
 * Loads JAG simulation parameters and results from hdf5 files using conduit interfaces
 */
class data_reader_jag_conduit : public generic_data_reader {
 public:
  using ch_t = float; ///< jag output image channel type
  using conduit_ch_t = conduit::float32_array; ///< conduit type for ch_t array wrapper
  using scalar_t = double; ///< jag scalar output type
  using input_t = double; ///< jag input parameter type
  /// Type for the pair of the key string of a sample and the handle of the file that contains it
  using sample_locator_t = std::pair<std::string, hid_t>;
  using sample_map_t = std::vector<sample_locator_t>; ///< valid sample map type
  using sample_t = sample_list_jag::sample_t;
  using sample_file_id_t = sample_list_jag::sample_file_id_t;
  /// linear transform on X defined as: first * X + second => X'
  using linear_transform_t = std::pair<double, double>;

  /**
   * Dependent/indepdendent variable types
   * - JAG_Image: simulation output images
   * - JAG_Scalar: simulation output scalars
   * - JAG_Input: simulation input parameters
   * - Undefined: the default
   */
  enum variable_t {Undefined=0, JAG_Image, JAG_Scalar, JAG_Input};
  using TypeID = conduit::DataType::TypeID;

  /// Type to define a prefix string and the minimum length requirement to filter out a key
  using prefix_t = std::pair<std::string, size_t>;

  data_reader_jag_conduit(bool shuffle = true) = delete;
  data_reader_jag_conduit(const std::shared_ptr<cv_process>& pp, bool shuffle = true);
  data_reader_jag_conduit(const data_reader_jag_conduit&);
  data_reader_jag_conduit(const data_reader_jag_conduit&, const std::vector<int>& ds_sample_move_list);
  data_reader_jag_conduit& operator=(const data_reader_jag_conduit&);
  ~data_reader_jag_conduit() override;
  data_reader_jag_conduit* copy() const override { return new data_reader_jag_conduit(*this); }

  void setup(int num_io_threads, std::shared_ptr<thread_pool> io_thread_pool) override;

  std::string get_type() const override {
    return "data_reader_jag_conduit";
  }

  /// Choose which data to use for independent variable
  void set_independent_variable_type(const std::vector< std::vector<variable_t> >& independent);
  /// Choose which data to use for dependent variable
  void set_dependent_variable_type(const std::vector< std::vector<variable_t> >& dependent);

  /// Tell which data to use for independent variable
  std::vector<variable_t> get_independent_variable_type() const;
  /// Tell which data to use for dependent variable
  std::vector<variable_t> get_dependent_variable_type() const;

  /// Set the common prefix path for any output scalar fields stored
  void set_output_scalar_prefix(const std::string& prefix) { m_output_scalar_prefix = prefix; }
  /// Set the common prefix path for any output images stored
  void set_output_image_prefix(const std::string& prefix) { m_output_image_prefix = prefix; }
  /// Set the common prefix path for any input variables stored
  void set_input_prefix(const std::string& prefix) { m_input_prefix = prefix; }

  /// Set the image dimension
  void set_image_dims(const int width, const int height, const int ch = 1);
  /// Choose images to use. e.g. by measurement views and time indices
  void set_image_choices(const std::vector<std::string> image_keys);
  /// Report the image choices
  const std::vector<std::string>& get_image_choices() const;

  /// Add a scalar key to filter out
  void add_scalar_filter(const std::string& key);
  /// Add a scalar key prefix to filter out
  void add_scalar_prefix_filter(const prefix_t& p);
  /// Add an input key to filter out
  void add_input_filter(const std::string& key);
  /// Add an input key prefix to filter out
  void add_input_prefix_filter(const prefix_t& p);

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

#ifndef _JAG_OFFLINE_TOOL_MODE_
  /// Load data and do data reader's chores.
  void load() override;
  /// Set the type of io_buffer that will rely on this reader
  void set_io_buffer_type(const std::string io_buffer);

  /// Set the id of this local instance
  void set_local_id(const std::string role);
  /// Get the id of this local instance
  int get_local_id(const std::string role) const;
  /// Set the set of open hdf5 data files
  // void set_open_hdf5_files(std::shared_ptr<hdf5_file_handles>& f);
  // /// Get the set of open hdf5 data files
  // std::shared_ptr<hdf5_file_handles>& get_open_hdf5_files();
  /// Set the leader of local data reader group
  void set_leading_reader(data_reader_jag_conduit* r);
  /// Get the leader of local data reader group
  data_reader_jag_conduit* get_leading_reader();
#else
  /// See if the image size is consistent with the linearized size
  void check_image_data();
#endif // _JAG_OFFLINE_TOOL_MODE_

  /// Set every reader instances in a trainer to have an independent index list
  void set_list_per_trainer(bool flag) { m_list_per_trainer = flag; };
  /// Set every reader instances in a model to have an independent index list
  void set_list_per_model(bool flag) { m_list_per_model = flag; };

  bool has_list_per_model() const { return m_list_per_model; }
  bool has_list_per_trainer() const { return m_list_per_trainer; }


  /// Fetch data of a mini-batch or reuse it from the cache of the leading reader
  int fetch_data(CPUMat& X, El::Matrix<El::Int>& indices_fetched) override;
  /// Fetch responses of a mini-batch or reuse it from the cache of the leading reader
  int fetch_responses(CPUMat& Y) override;
  /// Fetch labels of a mini-batch or reuse it from the cache of the leading reader
  int fetch_labels(CPUMat& Y) override;

  /// Return the number of measurement views
  unsigned int get_num_img_srcs() const;
  /// Return the linearized size of an image
  size_t get_linearized_image_size() const;
  /// Return the linearized size of a single channel image
  size_t get_linearized_1ch_image_size() const;
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

  /// Return the slice points for linearized independent variables
  std::vector<El::Int> get_slice_points_independent() const;
  /// Return the slice points for linearized dependent variables
  std::vector<El::Int> get_slice_points_dependent() const;

  int get_num_data() const override;
  int get_num_labels() const override;
  int get_linearized_label_size() const override;
  int get_linearized_size(const std::string& desc) const override;

  void set_split_image_channels();
  void unset_split_image_channels();
  bool check_split_image_channels() const;

  /// Show the description
  std::string get_description() const;

  /// Return the image simulation output of the i-th sample
  std::vector<cv::Mat> get_cv_images(const size_t i, conduit::Node& sample) const;

  /**
   * Return the images of the i-th sample as an 1-D vector of lbann::DataType
   * There is one image per view, each of which is taken at closest to the bang time.
   */
  std::vector<ch_t> get_images(const size_t i, conduit::Node& sample) const;

  /// Return the scalar simulation output data of the i-th sample
  std::vector<scalar_t> get_scalars(const size_t i, conduit::Node& sample) const;

  /// Return the simulation input parameters of the i-th sample
  std::vector<input_t> get_inputs(const size_t i, conduit::Node& sample) const;

  template<typename S>
  static size_t add_val(const std::string key, const conduit::Node& n, std::vector<S>& vals);

  void save_image(Mat& pixels, const std::string filename, bool do_scale = true) override;

  void setup_data_store(int mini_batch_size);

  /// A untiliy function to convert the pointer to image data into an opencv image
  static cv::Mat cast_to_cvMat(const std::pair<size_t, const ch_t*> img,
                               const int height, const int num_ch=1);
  /// A utility function to convert a JAG variable type to name string
  static std::string to_string(const variable_t t);

  /// print the schema of the specific sample identified by a given id
  void print_schema(const size_t i) const;

  void clear_image_normalization_params();
  void clear_scalar_normalization_params();
  void clear_input_normalization_params();
  void add_image_normalization_param(const linear_transform_t& t);
  void add_scalar_normalization_param(const linear_transform_t& t);
  void add_input_normalization_param(const linear_transform_t& t);

 protected:

  /// once the sample_list class and file formats are generalized and
  /// finalized, it should (may?) be possible to code a single
  /// preload_data_store method.
  void preload_data_store() override;

  virtual void set_defaults();
  virtual bool replicate_processor(const cv_process& pp, const int nthreads);
  virtual void copy_members(const data_reader_jag_conduit& rhs, const std::vector<int>& ds_sample_move_list = std::vector<int>());

  /// add data type for independent variable
  void add_independent_variable_type(const variable_t independent);
  /// add data type for dependent variable
  void add_dependent_variable_type(const variable_t dependent);

  /// Check if a key is in the black lists to filter out
  bool filter(const std::set<std::string>& key_filter,
              const std::vector<prefix_t>& prefix_filter, const std::string& name) const;

  using generic_data_reader::get_linearized_size;
  /// Return the linearized size of a particular JAG variable type
  size_t get_linearized_size(const variable_t t) const;
  /// Return the dimension of a particular JAG variable type
  const std::vector<int> get_dims(const variable_t t) const;
  /// Return the slice points for linearized data or responses
  std::vector<El::Int> get_slice_points(const std::vector< std::vector<data_reader_jag_conduit::variable_t> >& var) const;
  /// A utility function to make a string to show all the variable types
  static std::string to_string(const std::vector<variable_t>& vec);
  /// A utility function to make a string to show all the groups of variable types
  static std::string to_string(const std::vector< std::vector<variable_t> >& vec);


  virtual std::vector<CPUMat>
    create_datum_views(CPUMat& X, const std::vector<size_t>& sizes, const int mb_idx) const;

  /// Export cached data minibatch
  int reuse_data(CPUMat& X);
  /// Export cached responses minibatch
  int reuse_responses(CPUMat& Y);
  /// Export cached labels minibatch
  int reuse_labels(CPUMat& Y);

  bool fetch(CPUMat& X, int data_id, conduit::Node& sample, int mb_idx, int tid,
             const variable_t vt, const std::string tag);
  bool fetch_datum(CPUMat& X, int data_id, int mb_idx) override;
  bool fetch_response(CPUMat& Y, int data_id, int mb_idx) override;
  bool fetch_label(CPUMat& X, int data_id, int mb_idx) override;

#ifndef _JAG_OFFLINE_TOOL_MODE_
  /// Shuffle sammple indices using a different RNG
  void shuffle_indices(rng_gen& gen) override;

  /**
   * Compute the number of parallel readers based on the type of io_buffer,
   * the mini batch size, the requested number of parallel readers.
   * This is done before populating the sample indices.
   */
  int compute_max_num_parallel_readers();
  /**
   * Check if there are sufficient number of samples for the given number of
   * data readers with distributed io buffer, based on the number of samples,
   * the number of models and the mini batch size.
   */
  bool check_num_parallel_readers(long data_set_size);
  /// Rely on pre-determined list of samples.
  void load_list_of_samples(const std::string filename, size_t stride=1, size_t offset=0);
  /// Load the sample list from a serialized archive from another rank
  void load_list_of_samples_from_archive(const std::string& sample_list_archive);

  /// See if the image size is consistent with the linearized size
  void check_image_data();
#endif // _JAG_OFFLINE_TOOL_MODE_

#if 0
  /// Open a conduit file and register the open file descriptor
  hid_t open_conduit_file(const std::string& conduit_file_path);

  /// Open all conduit files for all the samples to cache the file descriptor
  void open_all_conduit_files();
#endif

  /// Obtain the linearized size of images of a sample from the meta info
  void set_linearized_image_size();
  /// Make sure that the keys to choose scalar outputs are valid
  void check_scalar_keys();
  /// Make sure that the keys to choose scalar outputs are valid
  void check_input_keys();

  /**
   * Check if the key is associated with non-numeric value, that is not and
   * cannot be converted to a numertic.
   */
  static bool check_non_numeric(const std::string key);

  /// Allow const access to the conduit data structure
  static const conduit::Node& get_conduit_node(const conduit::Node& n_base, const std::string key);
  /** Load the conduit node with the data of the sample i identified by key
   *  from the file that contains the sample, and returm true. Upon failure
   *  to load from file, attempt to retrieve a random conduit node from
   *  the data_store (if --use_data_store) and return false.
   */
  bool load_conduit_node(const size_t i, const std::string& key, conduit::Node& node) const;
  /// Check if a key exist for sample i
  bool has_conduit_path(const size_t i, const std::string& key) const;

  /// Obtain image data
  std::vector< std::vector<ch_t> > get_image_data(const size_t i, conduit::Node& sample) const;

  bool data_store_active() const {
    bool flag = generic_data_reader::data_store_active();
    return (m_data_store != nullptr && flag);
  }

  bool priming_data_store() const {
    bool flag = generic_data_reader::priming_data_store();
    return (m_data_store != nullptr && flag);
  }

 protected:
  /// The flat list of independent variable types
  std::vector<variable_t> m_independent;
  /// The list of independent variable types grouped for slicing
  std::vector< std::vector<variable_t> > m_independent_groups;
  /// The flat list of dependent variable types
  std::vector<variable_t> m_dependent;
  /// The list of independent variable types grouped for slicing
  std::vector< std::vector<variable_t> > m_dependent_groups;

  int m_image_width; ///< image width
  int m_image_height; ///< image height
  int m_image_num_channels; ///< number of image channels
  size_t m_image_linearized_size; ///< The linearized size of an image
  size_t m_1ch_image_linearized_size; ///< The linearized size of a single channel image
  unsigned int m_num_img_srcs; ///< number of views result in images
  bool m_split_channels; ///< Whether to export a separate image per channel

  /// Whether data have been loaded
  bool m_is_data_loaded;

  int m_num_labels; ///< number of labels

  /// Common prefix path to any output scalar fields in Conduit / HDF5
  std::string m_output_scalar_prefix;
  /// Common prefix path to any output image fields in Conduit / HDF5
  std::string m_output_image_prefix;
  /// Common prefix path to any input fields in Conduit / HDF5
  std::string m_input_prefix;

  /// Allow image selection by the view and the time index
  std::vector<std::string> m_emi_image_keys;
  /// Keys to select a set of scalar simulation outputs to use. By default, use all.
  std::vector<std::string> m_scalar_keys;
  /// Keys to select a set of simulation input parameters to use. By default, use all.
  std::vector<std::string> m_input_keys;

  /// preprocessor duplicated for each omp thread
  std::vector<std::unique_ptr<cv_process> > m_pps;
  std::unique_ptr<cv_process> m_master_pps;

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

  /// The set of scalar variables to filter out
  std::set<std::string> m_scalar_filter;
  /// The list of scalar key prefixes to filter out
  std::vector<prefix_t> m_scalar_prefix_filter;
  /// The set of input variables to filter out
  std::set<std::string> m_input_filter;
  /// The list of input key prefixes to filter out
  std::vector<prefix_t> m_input_prefix_filter;

  /**
   * io_buffer type that will rely on this reader.
   * e.g. distributed_io_buffer, partitioned_io_buffer
   */
  std::string m_io_buffer_type;

  /// The number of local instances of this reader type
  static std::unordered_map<std::string, int> m_num_local_readers;
  /// locally addressable id in case of multiple data reader instances attached to a model
  int m_local_reader_id;

  /**
   * The leading data reader among the local readers, which actually does the
   * file IO and data shuffling.
   */
  data_reader_jag_conduit* m_leading_reader;

  CPUMat m_data_cache;
  CPUMat m_response_cache;
  CPUMat m_label_cache;
  int m_cached_data_mb_size;
  int m_cached_response_mb_size;
  int m_cached_label_mb_size;

  /// temporary normalization parameters based on linear transforms
  std::vector<linear_transform_t> m_image_normalization_params;
  std::vector<linear_transform_t> m_scalar_normalization_params;
  std::vector<linear_transform_t> m_input_normalization_params;

  typedef std::pair<std::string, std::string> conduit_sample;
  sample_list_jag m_sample_list;
  bool m_list_per_trainer;
  bool m_list_per_model;

  /** temporary image normalization
   * The inputs are the image to normalize, the image source id and the channel id.
   */
  void image_normalization(cv::Mat& img, size_t i, size_t ch) const;
};

/**
 * To faciliate the type comparison between a c++ native type and a conduit type id.
 * By deafult, each pair of a native type TN and a conduit type TC is not the same.
 * Those that are the same require explicit instantication to say otherwise.
 */
template<typename TN, conduit::DataType::TypeID TC>
struct is_same : std::false_type {};

#define _LBANN_CONDUIT_DTYPE_INSTANTIATION_(TN, TC) \
  template<> struct is_same<TN, TC> : std::true_type {}

_LBANN_CONDUIT_DTYPE_INSTANTIATION_(int8_t,   conduit::DataType::INT8_ID);
_LBANN_CONDUIT_DTYPE_INSTANTIATION_(int16_t,  conduit::DataType::INT16_ID);
_LBANN_CONDUIT_DTYPE_INSTANTIATION_(int32_t,  conduit::DataType::INT32_ID);
_LBANN_CONDUIT_DTYPE_INSTANTIATION_(int64_t,  conduit::DataType::INT64_ID);
_LBANN_CONDUIT_DTYPE_INSTANTIATION_(uint8_t,  conduit::DataType::UINT8_ID);
_LBANN_CONDUIT_DTYPE_INSTANTIATION_(uint16_t, conduit::DataType::UINT16_ID);
_LBANN_CONDUIT_DTYPE_INSTANTIATION_(uint32_t, conduit::DataType::UINT32_ID);
_LBANN_CONDUIT_DTYPE_INSTANTIATION_(uint64_t, conduit::DataType::UINT64_ID);
_LBANN_CONDUIT_DTYPE_INSTANTIATION_(float,    conduit::DataType::FLOAT32_ID);
_LBANN_CONDUIT_DTYPE_INSTANTIATION_(double,   conduit::DataType::FLOAT64_ID);
_LBANN_CONDUIT_DTYPE_INSTANTIATION_(char*,    conduit::DataType::CHAR8_STR_ID);

#undef _LBANN_CONDUIT_DTYPE_INSTANTIATION_

/// Check if type identified by the conduit dtype id is the same type as the type given as the template parameter
template<typename TN>
inline bool is_same_type(const conduit::DataType::TypeID dt) {
  switch(dt) {
    case conduit::DataType::INT8_ID:    return is_same<TN, conduit::DataType::INT8_ID>::value;
    case conduit::DataType::INT16_ID:   return is_same<TN, conduit::DataType::INT16_ID>::value;
    case conduit::DataType::INT32_ID:   return is_same<TN, conduit::DataType::INT32_ID>::value;
    case conduit::DataType::INT64_ID:   return is_same<TN, conduit::DataType::INT64_ID>::value;
    case conduit::DataType::UINT8_ID:   return is_same<TN, conduit::DataType::UINT8_ID>::value;
    case conduit::DataType::UINT16_ID:  return is_same<TN, conduit::DataType::UINT16_ID>::value;
    case conduit::DataType::UINT32_ID:  return is_same<TN, conduit::DataType::UINT32_ID>::value;
    case conduit::DataType::UINT64_ID:  return is_same<TN, conduit::DataType::UINT64_ID>::value;
    case conduit::DataType::FLOAT32_ID: return is_same<TN, conduit::DataType::FLOAT32_ID>::value;
    case conduit::DataType::FLOAT64_ID: return is_same<TN, conduit::DataType::FLOAT64_ID>::value;
    case conduit::DataType::CHAR8_STR_ID: return is_same<TN, conduit::DataType::CHAR8_STR_ID>::value;
    default: return false;
  }
  return false;
}

/**
 * Retrieve a value from the given node n, and add it to the vector of type S, vals.
 * The first argument key is the name of the current node (i.e. the name reported by
 * the node iterator to the node).
 */
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
