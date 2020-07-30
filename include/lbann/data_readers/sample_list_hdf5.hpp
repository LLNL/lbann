#ifndef __SAMPLE_LIST_HDF5_HPP__
#define __SAMPLE_LIST_HDF5_HPP__

#include "sample_list_open_files.hpp"
#include "hdf5.h"
#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"

namespace lbann {

template <typename sample_name_t>
class sample_list_hdf5 : public sample_list_open_files<sample_name_t, hid_t> {
 public:
  using file_handle_t = hid_t;
  using typename sample_list_open_files<sample_name_t, hid_t>::sample_file_id_t;
  using typename sample_list_open_files<sample_name_t, hid_t>::sample_t;
  using typename sample_list_open_files<sample_name_t, hid_t>::samples_t;
  using typename sample_list_open_files<sample_name_t, hid_t>::file_id_stats_t;
  using typename sample_list_open_files<sample_name_t, hid_t>::file_id_stats_v_t;
  using typename sample_list_open_files<sample_name_t, hid_t>::fd_use_map_t;

  sample_list_hdf5();
  ~sample_list_hdf5() override;

  bool is_file_handle_valid(const hid_t& h) const override;

 protected:
  void obtain_sample_names(hid_t& h, std::vector<std::string>& sample_names) const override;
  hid_t open_file_handle_for_read(const std::string& path) override;
  void close_file_handle(hid_t& h) override;
  void clear_file_handle(hid_t& h) override;
};


template <typename sample_name_t>
inline sample_list_hdf5<sample_name_t>::sample_list_hdf5()
: sample_list_open_files<sample_name_t, hid_t>() {} 

template <typename sample_name_t>
inline sample_list_hdf5<sample_name_t>::~sample_list_hdf5() {
  // Close the existing open files
  for(auto& f : this->m_file_id_stats_map) {
    file_handle_t& h = std::get<1>(f);
    close_file_handle(h);
    clear_file_handle(h);
    std::get<2>(f).clear();
  }
  this->m_file_id_stats_map.clear();
}

template <typename sample_name_t>
inline void sample_list_hdf5<sample_name_t>
::obtain_sample_names(hid_t& h, std::vector<std::string>& sample_names) const {
  conduit::relay::io::hdf5_group_list_child_names(h, "/", sample_names);
}

template <typename sample_name_t>
inline bool sample_list_hdf5<sample_name_t>
::is_file_handle_valid(const hid_t& h) const {
  return (h > static_cast<hid_t>(0));
}

template <typename sample_name_t>
inline hid_t sample_list_hdf5< sample_name_t>
::open_file_handle_for_read(const std::string& file_path) {
  return conduit::relay::io::hdf5_open_file_for_read(file_path);
}

template <typename sample_name_t>
inline void sample_list_hdf5<sample_name_t>
::close_file_handle(hid_t& h) {
  if(is_file_handle_valid(h)) {
    conduit::relay::io::hdf5_close_file(h);
  }
}

template <>
inline hid_t uninitialized_file_handle<hid_t>() {
  return static_cast<hid_t>(0);
}

template <typename sample_name_t>
inline void sample_list_hdf5<sample_name_t>
::clear_file_handle(hid_t& h) {
  h = uninitialized_file_handle<hid_t>();
}

} // end of namespace lbann

#endif // __SAMPLE_LIST_HDF5_HPP__
