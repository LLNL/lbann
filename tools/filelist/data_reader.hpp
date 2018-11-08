#ifndef _DATA_READER_HPP_
#define _DATA_READER_HPP_

#include <string>
#include <vector>
#include "lbann/base.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/omp_pragma.hpp"
#include <omp.h>

namespace lbann {

class generic_data_reader {
 public:
  generic_data_reader(bool);
  generic_data_reader(const generic_data_reader&) = default;
  generic_data_reader& operator=(const generic_data_reader&) = default;
  virtual generic_data_reader* copy() const { return nullptr; }
  virtual ~generic_data_reader() {}

  virtual std::string get_type() const = 0;

  bool is_master() const { return m_master; }

  /// Set the mini batch size
  void set_mini_batch_size(const int s) {
    m_mini_batch_size = s;
  }
  virtual int get_linearized_data_size() const = 0;
  virtual int get_linearized_response_size() const = 0;
  virtual int get_linearized_label_size() const = 0;
  virtual int get_num_labels() const = 0;
  virtual int get_linearized_size(const std::string& desc) const { return 0; }
  virtual const std::vector<int> get_data_dims() const = 0;
  virtual void save_image(Mat& pixels, const std::string filename, bool do_scale = true) = 0;

  virtual int fetch_data(CPUMat& X);
  virtual int fetch_responses(CPUMat& Y);
  virtual int fetch_labels(CPUMat& Y);

  void set_num_samples(size_t ns) { m_num_samples = ns; }
  virtual bool init();
  virtual void update();
  int get_cur_mini_batch_size() const;
  virtual void load() { init(); }

 protected:
  virtual bool fetch_datum(CPUMat& X, int data_id, int mb_idx, int tid) = 0;
  virtual bool fetch_response(CPUMat& Y, int data_id, int mb_idx, int tid) = 0;
  virtual bool fetch_label(CPUMat& Y, int data_id, int mb_idx, int tid) = 0;
  int get_current_mini_batch_size() const { return 128; }

  bool m_master;
  bool m_gan_labelling = false;
  int m_gan_label_value = 0;
  std::string m_role;
  int m_mini_batch_size;

  int m_sid; // sample index
  int m_num_samples;
  int m_current_mini_batch_idx;
  int m_num_full_mbs;
  int m_num_rem_samples;
};

inline generic_data_reader::generic_data_reader(bool b)
 : m_master(true), m_role("offline"), m_mini_batch_size(8), m_sid(0),
   m_num_samples(0), m_current_mini_batch_idx(0),
   m_num_full_mbs(0), m_num_rem_samples(0)
{}

template<typename T>
inline void set_minibatch_item(CPUMat& M, const int mb_idx, const T* const ptr, const size_t count) {
  if ((count > 0u) && (ptr == nullptr)) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
                          " :: attempt to dereference a nullptr ");
  }
  for (size_t i = 0u; i < count; ++i) {
    M.Set(static_cast<El::Int>(i), static_cast<El::Int>(mb_idx), static_cast<DataType>(ptr[i]));
  }
}

inline int generic_data_reader::fetch_data(CPUMat& X) {
  const int mbsz = get_cur_mini_batch_size();
  LBANN_DATA_FETCH_OMP_FOR (int s = 0; s < mbsz; s++) {
    fetch_datum(X, static_cast<int>(m_sid + s), static_cast<int>(s), LBANN_OMP_THREAD_NUM);
  }
  return mbsz;
}

inline int generic_data_reader::fetch_responses(CPUMat& X) {
  const int mbsz = get_cur_mini_batch_size();
  LBANN_DATA_FETCH_OMP_FOR (int s = 0; s < mbsz; s++) {
    fetch_response(X, static_cast<int>(m_sid + s), static_cast<int>(s), LBANN_OMP_THREAD_NUM);
  }
  return mbsz;
}

inline int generic_data_reader::fetch_labels(CPUMat& X) {
  const int mbsz = get_cur_mini_batch_size();
  LBANN_DATA_FETCH_OMP_FOR (int s = 0; s < mbsz; s++) {
    fetch_label(X, static_cast<int>(m_sid + s), static_cast<int>(s), LBANN_OMP_THREAD_NUM);
  }
  return mbsz;
}

inline int generic_data_reader::get_cur_mini_batch_size() const {
  return (m_current_mini_batch_idx >= m_num_full_mbs)? m_num_rem_samples : m_mini_batch_size;
}

inline bool generic_data_reader::init() {
  if (m_num_samples == 0 || m_mini_batch_size == 0) {
    return false;
  }
  m_num_full_mbs = (m_num_samples / m_mini_batch_size) ;
  m_num_rem_samples = m_num_samples - m_num_full_mbs * m_mini_batch_size;
  m_sid = 0;
  m_current_mini_batch_idx = 0;
  std::cout << "m_num_full_mbs " << m_num_full_mbs << " m_num_rem_samples " << m_num_rem_samples << std::endl;
  return true;
}

inline void generic_data_reader::update() {
  m_sid += get_cur_mini_batch_size();
  m_current_mini_batch_idx ++;
  std::cout << "m_current_mini_batch_idx: " << m_current_mini_batch_idx << " m_sid" << m_sid << std::endl;
}

} // namespace lbann

#endif // _DATA_READER_HPP_
