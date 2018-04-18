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
//
// lbann_data_reader .hpp - Input data base class for training, testing
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_HPP
#define LBANN_DATA_READER_HPP

#include "lbann/base.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/comm.hpp"
#include "lbann/io/file_io.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/data_readers/image_preprocessor.hpp"
#include <cassert>
#include <algorithm>
#include <string>
#include <vector>
#include <unistd.h>


#define NOT_IMPLEMENTED(n) { \
  std::stringstream s; \
  s << "the method " << n << " has not been implemented"; \
  throw lbann_exception(s.str()); }

namespace lbann {

class generic_data_store;
class model;

/**
 * A data reader manages reading in data in a particular format.
 * This abstract base class manages common functionality. In particular, child
 * classes should implement load and the appropriate subset of fetch_datum,
 * fetch_label, and fetch_response.
 */
class generic_data_reader : public lbann_image_preprocessor {
 public:

  /**
   * ctor
   */
  generic_data_reader(bool shuffle = true) :
    m_data_store(nullptr),
    m_comm(nullptr),
    m_mini_batch_size(0), m_current_pos(0),
    m_stride_to_next_mini_batch(0), m_base_offset(0), m_model_offset(0),
    m_sample_stride(1), m_iteration_stride(1),
    m_last_mini_batch_size(0),
    m_stride_to_last_mini_batch(0),
    m_reset_mini_batch_index(0),
    m_loaded_mini_batch_idx(0),
    m_current_mini_batch_idx(0),
    m_num_iterations_per_epoch(0), m_global_mini_batch_size(0),
    m_global_last_mini_batch_size(0),
    m_world_master_mini_batch_adjustment(0),
    m_num_parallel_readers(0), m_model_rank(0),
    m_file_dir(""), m_data_fn(""), m_label_fn(""),
    m_shuffle(shuffle), m_absolute_sample_count(0), m_validation_percent(0.0),
    m_use_percent(1.0),
    m_master(false),
    m_save_minibatch_indices(false),
    m_compound_rank(0),
    m_num_global_indices(0)
  {}
  generic_data_reader(const generic_data_reader&) = default;
  generic_data_reader& operator=(const generic_data_reader&) = default;

  ~generic_data_reader() override {}
  virtual generic_data_reader* copy() const = 0;

  /// set the comm object
  void set_comm(lbann_comm *comm) {
    m_comm = comm;
    set_master(comm->am_world_master());
  }

  /// returns a (possibly nullptr) to comm
  lbann_comm * get_comm() const {
    return m_comm;
  }

  // These non-virtual methods are used to specify where data is, how much to
  // load, etc.

  /**
   * Set base directory for your data. Optional: if given,
   * then get_data_filename will concatenate the value passed
   * to this method with the value passed to set_data_filename,
   * and similarly for get_label_filename
   */
  void set_file_dir(std::string s);

  /**
   * Returns the base directory for your data.
   * If set_file_dir was not called, returns the empty string
   */
  std::string get_file_dir() const;

  /**
   * Set the filename for your data (images, etc).
   * This may either be a complete filepath, or a subdirectory;
   * see note for set_file_dir(). Also, use this method
   * for cases where the file contains a list of files (e.g, imagenet)
   */
  void set_data_filename(std::string s);

  /**
   * Returns the complete filepath to you data file.
   * See not for set_file_dir()
   */
  std::string get_data_filename() const;

  /**
   * Set the filename for your data (images, etc).
   * This may either be a complete filepath, or a subdirectory;
   * see note for set_file_dir()
   */
  void set_label_filename(std::string s);

  /**
   * Returns the complete filepath to you data file.
   * See not for set_file_dir(). Note: some pipelines (autoencoders)
   * will not make use of this method.
   */
  std::string get_label_filename() const;

  /**
   * If set to false, indices (data samples) are not shuffled
   * default (in ctor) is true.
   */
  void set_shuffle(bool b) { m_shuffle = b; }

  /**
   * Returns true if data samples are shuffled.
   */
  bool is_shuffled() const { return m_shuffle; }

  /**
   * Set shuffled indices; primary use is for testing
   * and reproducibility
   */
  void set_shuffled_indices(const std::vector<int> &indices) {
    m_shuffled_indices = indices;
  }

  /**
   * Returns the shuffled indices; primary use is for testing.
   */
  const std::vector<int> & get_shuffled_indices() const {
    return m_shuffled_indices;
  }

  /**
   * Read the first 'n' samples. If nonzero, this over-rides
   * set_absolute_sample_count, set_use_percent. The intent
   * is to use this for testing. A problem with set_absolute_sample_count
   * and set_use_percent is that the entire data set is read in, then
   * a subset is selected
   */
  void set_first_n(int n);

  /**
   * Sets the absolute number of data samples that will be used for training or
   * testing.
   */
  void set_absolute_sample_count(size_t s);

  /**
   * Set the percentage of the data set to use for training and validation or
   * testing.
   * @param s The percentage used, in the range [0, 1].
   */
  void set_use_percent(double s);

  /**
   * Sets the percentage of the dataset to be used for validation.
   * @param s The percentage used, in the range [0, 1].
   */
  virtual void set_validation_percent(double s);

  /**
   * Set an idenifier for the dataset.
   * The role should be one of "train", "test", or "validate".
   */
  virtual void set_role(std::string role) {
    m_role = role;
  }

  /**
   * Get the role for this dataset.
   */
  std::string get_role() const {
    return m_role;
  }

  /**
   * Load the dataset.
   * Each data reader implementation should implement this to initialize its
   * internal data structures, determine the number of samples and their
   * dimensionality (if needed), and set up and shuffle samples.
   */
  virtual void load() = 0;

  /**
   * Prepare to start processing an epoch of data.
   * If shuffle is true, then shuffle the indices of the data set
   * If the base offset is not specified set it to 0
   * If the stride is not specified set it to batch size
   */
  void setup();

  /** Return this data_reader's type */
  virtual std::string get_type() const = 0;

  /// Fetch this mini-batch's samples into X.
  virtual int fetch_data(CPUMat& X);
  /// Fetch this mini-batch's labels into Y.
  virtual int fetch_labels(CPUMat& Y);
  /// Fetch this mini-batch's responses into Y.
  virtual int fetch_responses(CPUMat& Y);

  /**
   * Save pixels to an image. The implementing data reader is responsible for
   * handling format detection, conversion, etc.
   */
  // TODO: This function needs to go away from here
  void save_image(Mat& pixels, const std::string filename,
                          bool do_scale = true) override {
    NOT_IMPLEMENTED("save_image");
  }
  bool is_data_reader_done(bool is_active_reader);
  /**
   * During the network's update phase, the data reader will
   * advanced the current position pointer.  If the pointer wraps
   * around, then reshuffle the data indicies.
   */
  virtual bool update(bool is_active_reader);

  /// Return the number of labels (classes) in this dataset.
  virtual int get_num_labels() const {
    return 0;
  }
  /// Return the number of responses in this dataset.
  virtual int get_num_responses() const {
    return 1;
  }
  /// Get the linearized size (i.e. number of elements) in a sample.
  virtual int get_linearized_data_size() const {
    return 0;
  }
  /// Get the linearized size (i.e. number of elements) in a label.
  virtual int get_linearized_label_size() const {
    return 0;
  }
  /// Get the linearized size (i.e. number of elements) in a response.
  virtual int get_linearized_response_size() const {
    return 1;
  }
  /// Get the dimensions of the data.
  virtual const std::vector<int> get_data_dims() const {
    return std::vector<int>(0);
  }
  /// True if the data reader's current position is valid.
  bool position_valid() const {
    return (m_current_pos < (int)m_shuffled_indices.size());
  }
  /// True if the data reader is at the start of an epoch.
  bool at_new_epoch() const {
    /// Note that data readers can start at a non-zero index if there
    /// are parallel data readers in a model
    return ((m_loaded_mini_batch_idx == m_reset_mini_batch_index)
            && (m_current_mini_batch_idx == 0));
  }
  /// Set the mini batch size
  void set_mini_batch_size(const int s) {
    m_mini_batch_size = s;
  }
  /// Get the mini batch size
  int get_mini_batch_size() const {
    return m_mini_batch_size;
  }
  /// Get the loaded mini-batch size
  int get_loaded_mini_batch_size() const;
  /// Get the current mini-batch size.
  int get_current_mini_batch_size() const;
  /// Get the current global mini-batch size.
  int get_current_global_mini_batch_size() const;
  /// Get the current mini-batch size.
  int get_current_world_master_mini_batch_adjustment(int model_rank) const;
  /// Return the full mini_batch_size.
  int get_mini_batch_max() const {
    return m_mini_batch_size;
  }
  /// Set the mini batch size across all models (global)
  void set_global_mini_batch_size(const int s) {
    m_global_mini_batch_size = s;
  }
  /// Return the mini_batch_size across all models (global)
  int get_global_mini_batch_size() const {
    return m_global_mini_batch_size;
  }
  /// Set the mini batch stride
  void set_stride_to_next_mini_batch(const int s) {
    m_stride_to_next_mini_batch = s;
  }
  /// Return the mini batch stride.
  int get_stride_to_next_mini_batch() const {
    return m_stride_to_next_mini_batch;
  }
  /// Set the sample stride
  void set_sample_stride(const int s) {
    m_sample_stride = s;
  }
  /// Return the sample stride.
  int get_sample_stride() const {
    return m_sample_stride;
  }
  /// Set the iteration stride
  void set_iteration_stride(const int s) {
    m_iteration_stride = s;
  }
  /// Return the iteration stride.
  int get_iteration_stride() const {
    return m_iteration_stride;
  }
  /// Return the base offset.
  void set_base_offset(const int s) {
    m_base_offset = s;
  }
  /// Return the base offset.
  int get_base_offset() const {
    return m_base_offset;
  }
  /// Set the model offset
  void set_model_offset(const int s) {
    m_model_offset = s;
  }
  /// Return the model offset.
  int get_model_offset() const {
    return m_model_offset;
  }
  /// Set the last mini batch size
  void set_last_mini_batch_size(const int s) {
    m_last_mini_batch_size = s;
  }
  /// Return the last mini batch size
  int get_last_mini_batch_size() const {
    return m_last_mini_batch_size;
  }
  /// Set the last mini batch size across all models (global)
  void set_global_last_mini_batch_size(const int s) {
    m_global_last_mini_batch_size = s;
  }
  /// Return the last mini batch size across all models (global)
  int get_global_last_mini_batch_size() const {
    return m_global_last_mini_batch_size;
  }
  /// Set the world master mini batch adjustment (global)
  void set_world_master_mini_batch_adjustment(const int s) {
    m_world_master_mini_batch_adjustment = s;
  }
  /// Return the world master mini batch adjustment (global)
  int get_world_master_mini_batch_adjustment() const {
    return m_world_master_mini_batch_adjustment;
  }
  /// Set the last mini batch stride
  void set_stride_to_last_mini_batch(const int s) {
    m_stride_to_last_mini_batch = s;
  }
  /// Return the last mini batch stride
  int get_stride_to_last_mini_batch() const {
    return m_stride_to_last_mini_batch;
  }
  /// Set the number of parallel readers per model
  void set_num_parallel_readers(const int s) {
    m_num_parallel_readers = s;
  }
  /// Return the number of parallel readers per model
  int get_num_parallel_readers() const {
    return m_num_parallel_readers;
  }
  /// Set the starting mini-batch index for the epoch
  void set_reset_mini_batch_index(const int s) {
    m_reset_mini_batch_index = s;
  }
  /// Return the starting mini-batch index for the epoch
  int get_reset_mini_batch_index() const {
    return m_reset_mini_batch_index;
  }
  /// Return the current mini-batch index for the epoch
  int get_loaded_mini_batch_index() const {
    return m_loaded_mini_batch_idx;
  }
  /// Return the current mini-batch index for the epoch
  int get_current_mini_batch_index() const {
    return m_current_mini_batch_idx;
  }
  /// Set the current position based on the base and model offsets
  void set_initial_position() {
    m_current_pos = m_base_offset + m_model_offset;
    m_loaded_mini_batch_idx = m_reset_mini_batch_index;
    m_current_mini_batch_idx = 0;
  }
  /// Get the current position in the data reader.
  int get_position() const {
    return m_current_pos;
  }
  /// Get the next position in the data reader.
  int get_next_position() const;
  /// Get a pointer to the start of the shuffled indices.
  int *get_indices() {
    return &m_shuffled_indices[0];
  }
  /// Get the number of samples in this dataset.
  int get_num_data() const {
    return (int)m_shuffled_indices.size();
  }
  /// Get the number of unused samples in this dataset.
  int get_num_unused_data() const {
    return (int)m_unused_indices.size();
  }
  /// Get a pointer to the start of the unused sample indices.
  int *get_unused_data() {
    return &m_unused_indices[0];
  }
  /// Get a pointer to the fetched indices matrix.
  El::Matrix<El::Int>* get_indices_fetched_per_mb() {
    return &m_indices_fetched_per_mb;
  }
  /// Set the number of iterations in each epoch.
  void set_num_iterations_per_epoch(int num_iterations_per_epoch) {
    m_num_iterations_per_epoch = num_iterations_per_epoch;  /// @todo BVE FIXME merge this with alternate approach
  }
  /// Get the number of iterations in each epoch.
  int get_num_iterations_per_epoch() const {
    return m_num_iterations_per_epoch;  /// @todo BVE FIXME merge this with alternate approach
  }

  /// Return the index of the current iteration step in the epoch (also the mini-batch index)
  int get_current_step_in_epoch() const {
    return  m_current_mini_batch_idx;
  }

  /// only the master may write to cerr or cout; primarily for use in debugging during development
  virtual void set_master(bool m) {
    m_master = m;
  }

  /// only the master may write to cerr or cout; primarily for use in debugging during development
  bool is_master() const {
    return m_master;
  }

  /// Allow the reader to know where it is in the model hierarchy
  virtual void set_rank(int rank) {
    m_model_rank = rank;
  }

  /// Allow the reader to know where it is in the model hierarchy
  int get_rank() const {
    return m_model_rank;
  }

  /**
   * Select the appropriate subset of data based on settings.
   */
  void select_subset_of_data();

  /**
   * Replaced the shuffled index set with the unused index set, empying the
   * unused set.
   */
  void use_unused_index_set();

  /** \brief Given directory to store checkpoint files, write state to file and add to number of bytes written */
  bool save_to_checkpoint_shared(persist& p, const char *name);

  /** \brief Given directory to store checkpoint files, read state from file and add to number of bytes read */
  bool load_from_checkpoint_shared(persist& p, const char *name);

  struct packing_header {
    uint64_t mini_batch_size;
    uint64_t current_pos;
    uint64_t current_mini_batch_idx;
    uint64_t data_size;
    uint64_t unused_data_size;
    uint64_t stride_to_last_mini_batch;
    uint64_t stride_to_next_mini_batch;
    uint64_t base_offset;
    uint64_t model_offset;
    uint64_t sample_stride;
    uint64_t iteration_stride;
    uint64_t loaded_mini_batch_idx;
    uint64_t reset_mini_batch_index;
    uint64_t last_mini_batch_size;
    uint64_t num_iterations_per_epoch;
    uint64_t global_mini_batch_size;
    uint64_t global_last_mini_batch_size;
    uint64_t num_parallel_readers;
    uint64_t model_rank;
    uint64_t world_master_mini_batch_adjustment;
  };
  bool pack_scalars(persist& p, const char *name) {
    char fieldname[1024];
    lbann::persist_type persist_value;
    std::string s_name(name);
    if(s_name.compare("data_reader_validation") == 0){
      persist_value = persist_type::validate;
    } else {
       persist_value= persist_type::train;
    }

    snprintf(fieldname, sizeof(fieldname), "%s__mini_batch_size", name);
    p.write_uint64(persist_value, fieldname, (uint64_t) m_mini_batch_size);

    snprintf(fieldname, sizeof(fieldname), "%s_current_mini_batch_idx", name);
    p.write_uint64(persist_value, fieldname, (uint64_t) m_current_mini_batch_idx);

    int size = m_shuffled_indices.size();
    snprintf(fieldname, sizeof(fieldname), "%s_data_size", name);
    p.write_uint64(persist_value, fieldname, (uint64_t) size);

    snprintf(fieldname, sizeof(fieldname), "%s_data_position", name);
    p.write_uint64(persist_value, fieldname, (uint64_t) m_current_pos);

    snprintf(fieldname, sizeof(fieldname), "%s_data_indices", name);
    p.write_int32_contig(persist_value, fieldname, &m_shuffled_indices[0], (uint64_t) size);

    snprintf(fieldname, sizeof(fieldname), "%s_stride_to_last_mini_batch", name);
    p.write_uint64(persist_value, fieldname, (uint64_t) m_stride_to_last_mini_batch);

    snprintf(fieldname, sizeof(fieldname), "%s_stride_to_next_mini_batch", name);
    p.write_uint64(persist_value, fieldname, (uint64_t) m_stride_to_next_mini_batch);

    snprintf(fieldname, sizeof(fieldname), "%s_base_offset", name);
    p.write_uint64(persist_value, fieldname, (uint64_t) m_base_offset);

    snprintf(fieldname, sizeof(fieldname), "%s_model_offset", name);
    p.write_uint64(persist_value, fieldname, (uint64_t) m_model_offset);

    snprintf(fieldname, sizeof(fieldname), "%s_sample_stride", name);
    p.write_uint64(persist_value, fieldname, (uint64_t) m_sample_stride);

    snprintf(fieldname, sizeof(fieldname), "%s_iteration_stride", name);
    p.write_uint64(persist_value, fieldname, (uint64_t) m_iteration_stride);

    snprintf(fieldname, sizeof(fieldname), "%s_loaded_mini_batch_idx", name);
    p.write_uint64(persist_value, fieldname, (uint64_t) m_loaded_mini_batch_idx);

    snprintf(fieldname, sizeof(fieldname), "%s_reset_mini_batch_index", name);
    p.write_uint64(persist_value, fieldname, (uint64_t) m_reset_mini_batch_index);


    int unused_size = m_unused_indices.size();
    snprintf(fieldname, sizeof(fieldname), "%s_unused_data_size", name);
    p.write_uint64(persist_value, fieldname, (uint64_t) unused_size);

    snprintf(fieldname, sizeof(fieldname), "%s_unused_data_indices", name);
    p.write_int32_contig(persist_value, fieldname, &m_unused_indices[0], (uint64_t) unused_size);

    snprintf(fieldname, sizeof(fieldname), "%s_last_mini_batch_size", name);
    p.write_uint64(persist_value, fieldname, (uint64_t) m_last_mini_batch_size);

    snprintf(fieldname, sizeof(fieldname), "%s_num_iteration_per_epoch", name);
    p.write_uint64(persist_value, fieldname, (uint64_t) m_num_iterations_per_epoch);

    snprintf(fieldname, sizeof(fieldname), "%s_global_mini_batch_size", name);
    p.write_uint64(persist_value, fieldname, (uint64_t) m_global_mini_batch_size);

    snprintf(fieldname, sizeof(fieldname), "%s_global_last_mini_batch_size", name);
    p.write_uint64(persist_value, fieldname, (uint64_t) m_global_last_mini_batch_size);

    snprintf(fieldname, sizeof(fieldname), "%s_num_parallel_readers", name);
    p.write_uint64(persist_value, fieldname, (uint64_t) m_num_parallel_readers);

    snprintf(fieldname, sizeof(fieldname), "%s_model_rank", name);
    p.write_uint64(persist_value, fieldname, (uint64_t) m_model_rank);

    snprintf(fieldname, sizeof(fieldname), "%s_world_master_mini_batch_adjustment", name);
    p.write_uint64(persist_value, fieldname, (uint64_t) m_world_master_mini_batch_adjustment);
    return true;
  }

  bool unpack_scalars(persist& p, struct packing_header *header, const char *name){
    char fieldname[1024];
    lbann::persist_type persist_value;
    std::string s_name(name);
    if(s_name.compare("data_reader_validation") == 0){
      persist_value = persist_type::validate;
    } else {
       persist_value= persist_type::train;
    }
    // Closest to non checkpoint run only loads m_current_pos

    // record minibatch index
    uint64_t val;

    snprintf(fieldname, sizeof(fieldname), "%s_mini_batch_size", name);
    p.read_uint64(persist_value, fieldname, &val);
    m_mini_batch_size = (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_current_mini_batch_idx", name);
    p.read_uint64(persist_value, fieldname, &val);
    m_current_mini_batch_idx = (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_data_size", name);
    p.read_uint64(persist_value, fieldname, &val);
    auto size = (int) val;

    // get current position within data
    snprintf(fieldname, sizeof(fieldname), "%s_data_position", name);
    p.read_uint64(persist_value, fieldname, &val);
    m_current_pos = (int) val;
    //resize shuffled index array to hold values
    m_shuffled_indices.resize(size);

     //read list of indices
    snprintf(fieldname, sizeof(fieldname), "%s_data_indices", name);
    p.read_int32_contig(persist_value, fieldname, &m_shuffled_indices[0], (uint64_t) size);
    // BEGIN TEST
    snprintf(fieldname, sizeof(fieldname), "%s_stride_to_last_mini_batch", name);
    p.read_uint64(persist_value, fieldname, &val);
    m_stride_to_last_mini_batch = (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_stride_to_next_mini_batch", name);
    p.read_uint64(persist_value, fieldname, &val);
    m_stride_to_next_mini_batch = (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_base_offset", name);
    p.read_uint64(persist_value, fieldname, &val);
    m_base_offset = (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_model_offset", name);
    p.read_uint64(persist_value, fieldname, &val);
    m_model_offset = (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_sample_stride", name);
    p.read_uint64(persist_value, fieldname, &val);
    m_sample_stride= (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_iteration_stride", name);
    p.read_uint64(persist_value, fieldname, &val);
    m_iteration_stride= (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_loaded_mini_batch_idx", name);
    p.read_uint64(persist_value, fieldname, &val);
    m_loaded_mini_batch_idx = (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_reset_mini_batch_index", name);
    p.read_uint64(persist_value, fieldname, &val);
    m_reset_mini_batch_index = (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_unused_data_size", name);
    p.read_uint64(persist_value, fieldname, &val);
    auto unused_size = (int) val;

    m_unused_indices.resize(unused_size);
    snprintf(fieldname, sizeof(fieldname), "%s_unused_data_indices", name);
    p.read_int32_contig(persist_value, fieldname, &m_unused_indices[0], (uint64_t) unused_size);

    snprintf(fieldname, sizeof(fieldname), "%s_last_mini_batch_size", name);
    p.read_uint64(persist_value, fieldname, &val);
    m_last_mini_batch_size = (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_num_iterations_per_epoch", name);
    p.read_uint64(persist_value, fieldname, &val);
    m_num_iterations_per_epoch = (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_global_mini_batch_size", name);
    p.read_uint64(persist_value, fieldname, &val);
    m_global_mini_batch_size = (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_global_last_mini_batch_size", name);
    p.read_uint64(persist_value, fieldname, &val);
    m_global_last_mini_batch_size = (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_num_parallel_readers", name);
    p.read_uint64(persist_value, fieldname, &val);
    m_num_parallel_readers = (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_model_rank", name);
    p.read_uint64(persist_value, fieldname, &val);
    m_model_rank = (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_world_master_mini_batch_adjustment", name);
    p.read_uint64(persist_value, fieldname, &val);
    m_world_master_mini_batch_adjustment = (int) val;

    if(header != nullptr){
      //shuffled/unused data indices array size, used for resize after broadcast. Not unpacked.
      header->data_size = size;
      header->unused_data_size = unused_size;
      // all else, unpacked and set in unpack header.
      header->mini_batch_size = m_mini_batch_size;
      header->current_pos = m_current_pos;
      header->current_mini_batch_idx = m_current_mini_batch_idx;
      header->stride_to_last_mini_batch = m_stride_to_last_mini_batch;
      header->stride_to_next_mini_batch = m_stride_to_next_mini_batch;
      header->base_offset = m_base_offset;
      header->model_offset = m_model_offset;
      header->sample_stride = m_sample_stride;
      header->iteration_stride = m_iteration_stride;
      header->loaded_mini_batch_idx = m_loaded_mini_batch_idx;
      header->reset_mini_batch_index = m_reset_mini_batch_index;
      header->last_mini_batch_size = m_last_mini_batch_size;
      header->num_iterations_per_epoch = m_num_iterations_per_epoch;
      header->global_mini_batch_size = m_global_mini_batch_size;
      header->global_last_mini_batch_size = m_global_last_mini_batch_size;
      header->num_parallel_readers = m_num_parallel_readers;
      header->model_rank = m_model_rank;
      header->world_master_mini_batch_adjustment = m_world_master_mini_batch_adjustment;
    }

  return true;
  }

  void unpack_header(struct packing_header& header){
    m_mini_batch_size = (int) header.mini_batch_size;
    m_current_pos = (int) header.current_pos;
    m_current_mini_batch_idx = (int) header.current_mini_batch_idx;
    m_stride_to_last_mini_batch = (int) header.stride_to_last_mini_batch;
    m_stride_to_next_mini_batch = (int) header.stride_to_next_mini_batch;
    m_base_offset = (int) header.base_offset;
    m_model_offset = (int) header.model_offset;
    m_sample_stride = (int) header.sample_stride;
    m_iteration_stride = (int) header.iteration_stride;
    m_loaded_mini_batch_idx = (int) header.loaded_mini_batch_idx;
    m_reset_mini_batch_index = (int) header.reset_mini_batch_index;
    m_last_mini_batch_size = (int) header.last_mini_batch_size;
    m_num_iterations_per_epoch = (int) header.num_iterations_per_epoch;
    m_global_mini_batch_size = (int) header.global_mini_batch_size;
    m_global_last_mini_batch_size = (int) header.global_last_mini_batch_size;
    m_num_parallel_readers = (int) header.num_parallel_readers;
    m_model_rank = (int) header.model_rank;
    m_world_master_mini_batch_adjustment = (int) header.world_master_mini_batch_adjustment;
  }

  /// returns the data store
  generic_data_store * get_data_store() const {
    if (m_data_store == nullptr) {
      std::stringstream err;
      err << __FILE__  << " :: " << __LINE__ << " :: "
          << " m_data_store is nullptr";
    }
    return m_data_store;
  }

  /// sets up a data_store.
  virtual void setup_data_store(model *m);

  /** This call changes the functionality of fetch_data(); when set,
    * indices are added to m_my_minibatch_indices, but fetch_datum()
    * is not called. This method is added to support data store functionality.
    */
  void set_save_minibatch_entries(bool b);

  /// support of data store functionality
  const std::vector<std::vector<int> > & get_minibatch_indices() const {
    return m_my_minibatch_indices;
  }

  /// support of data store functionality
  int get_compound_rank() {
    return m_compound_rank;
  }

  /// support of data store functionality
  void set_compound_rank(int r) {
    m_compound_rank = r;
  }

  /// support of data store functionality
  void set_data_store(generic_data_store *g);

 protected:

  /**
   * Return the absolute number of data samples that will be used for training
   * or testing.
   */
  size_t get_absolute_sample_count() const;

  /**
   * Returns the percent of the dataset to be used for training or testing.
   * If training, this is the total for training and validation. Throws if
   * set_use_percent was not called.
   */
  double get_use_percent() const;

  /**
   * Return the percent of the dataset to be used for validation.
   */
  double get_validation_percent() const;

  /**
   * Returns the number of global indices. For train and validation,
   * this is the sum of their numbers
   */
  size_t get_num_global_indices() {
    return m_num_global_indices;
  }
 protected:

   int m_rank;

   generic_data_store *m_data_store;

   lbann_comm *m_comm;

  /**
   * Fetch a single sample into a matrix.
   * @param X The matrix to load data into.
   * @param data_id The index of the datum to fetch.
   * @param mb_idx The index within the mini-batch.
   */
  virtual bool fetch_datum(CPUMat& X, int data_id, int mb_idx, int tid) {
    NOT_IMPLEMENTED("fetch_dataum");
    return false;
  }

  /**
   * Fetch a single label into a matrix.
   * @param Y The matrix to load data into.
   * @param data_id The index of the datum to fetch.
   * @param mb_idx The index within the mini-batch.
   */
  virtual bool fetch_label(CPUMat& Y, int data_id, int mb_idx, int tid) {
    NOT_IMPLEMENTED("fetch_label");
    return false;
  }

  /**
   * Fetch a single response into a matrix.
   * @param Y The matrix to load data into.
   * @param data_id The index of the datum to fetch.
   * @param mb_idx The index within the mini-batch.
   */
  virtual bool fetch_response(CPUMat& Y, int data_id, int mb_idx, int tid) {
    NOT_IMPLEMENTED("fetch_response");
    return false;
  }

  /**
   * Called before fetch_datum/label/response to allow initialization.
   */
  virtual void preprocess_data_source(int tid) {};
  /**
   * Called after fetch_datum/label/response to allow initialization.
   */
  virtual void postprocess_data_source(int tid) {};

  /// Shuffle indices
  virtual void shuffle_indices();

  int m_mini_batch_size;
  int m_current_pos;
  /// Batch Stride is typically batch_size, but may be a multiple of batch size if there are multiple readers
  int m_stride_to_next_mini_batch;
  /// If there are multiple instances of the reader,
  /// then it may not reset to zero
  int m_base_offset;
  /// If there are multiple models with multiple instances of the reader,
  /// each model's set of readers may not reset to zero
  /// Provide a set of size, strides, and thresholds to handle the last mini batch of a dataset
  int m_model_offset;
  /// Sample stride is used when a mini-batch is finely interleaved across a DATA_PARALELL
  /// distribution.
  int m_sample_stride;
  /// Stride used by parallel data readers within the model
  int m_iteration_stride;

  std::vector<int> m_shuffled_indices;
  /// Record of the indicies that are not being used for training
  std::vector<int> m_unused_indices;

  int m_last_mini_batch_size;
  int m_stride_to_last_mini_batch;
  /// The index at which this data reader starts its epoch
  int m_reset_mini_batch_index;
  /// The index of the current mini-batch that has been loaded
  int m_loaded_mini_batch_idx;
  /// The index of the current mini-batch that is being processed (train/test/validate)
  int m_current_mini_batch_idx;
  int m_num_iterations_per_epoch; /// How many iterations all readers will execute

  int m_global_mini_batch_size;
  int m_global_last_mini_batch_size;
  int m_world_master_mini_batch_adjustment;

  int m_num_parallel_readers; /// How many parallel readers are being used

  int m_model_rank;  /// What is the rank of the data reader within a given model
  std::string m_file_dir;
  std::string m_data_fn;
  std::string m_label_fn;
  bool m_shuffle;
  size_t m_absolute_sample_count;
  double m_validation_percent;
  double m_use_percent;
  int m_first_n;
  std::string m_role;

  bool m_master;

  /// 1-D Matrix of which indices were fetched in this mini-batch
  El::Matrix<El::Int> m_indices_fetched_per_mb;

  friend class data_reader_merge_features;
  friend class data_reader_merge_samples;

 protected :
   /// added to support data store functionality
   bool m_save_minibatch_indices;

   /// added to support data store functionality
   std::vector<std::vector<int> > m_my_minibatch_indices;

   /// added to support data store functionality
   int m_compound_rank;

   /// added to support data store functionality
   size_t m_num_global_indices;
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_HPP
