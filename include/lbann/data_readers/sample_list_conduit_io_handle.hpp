#ifndef __SAMPLE_LIST_CONDUIT_IO_HANDLE_HPP__
#define __SAMPLE_LIST_CONDUIT_IO_HANDLE_HPP__

#include "sample_list_open_files.hpp"
#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_handle.hpp"

namespace lbann {

template <typename sample_name_t>
class sample_list_conduit_io_handle : public sample_list_open_files<sample_name_t, conduit::relay::io::IOHandle*> {
 public:
  using file_handle_t = conduit::relay::io::IOHandle*;
  using typename sample_list_open_files<sample_name_t, file_handle_t>::sample_file_id_t;
  using typename sample_list_open_files<sample_name_t, file_handle_t>::sample_t;
  using typename sample_list_open_files<sample_name_t, file_handle_t>::samples_t;
  using typename sample_list_open_files<sample_name_t, file_handle_t>::file_id_stats_t;
  using typename sample_list_open_files<sample_name_t, file_handle_t>::file_id_stats_v_t;
  using typename sample_list_open_files<sample_name_t, file_handle_t>::fd_use_map_t;

  sample_list_conduit_io_handle();
  ~sample_list_conduit_io_handle() override;

  bool is_file_handle_valid(const file_handle_t& h) const override;

 protected:
  void obtain_sample_names(file_handle_t& h, std::vector<std::string>& sample_names) const override;
  file_handle_t open_file_handle_for_read(const std::string& path) override;
  void close_file_handle(file_handle_t& h) override;
  void clear_file_handle(file_handle_t& h) override;
};


template <typename sample_name_t>
inline sample_list_conduit_io_handle<sample_name_t>::sample_list_conduit_io_handle()
: sample_list_open_files<sample_name_t, file_handle_t>() {}

template <typename sample_name_t>
inline sample_list_conduit_io_handle<sample_name_t>::~sample_list_conduit_io_handle() {
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
inline void sample_list_conduit_io_handle<sample_name_t>
::obtain_sample_names(sample_list_conduit_io_handle<sample_name_t>::file_handle_t& h, std::vector<std::string>& sample_names) const {
  sample_names.clear();
  if (h != nullptr) {
    h->list_child_names("/", sample_names);
  }
}

template <typename sample_name_t>
inline bool sample_list_conduit_io_handle<sample_name_t>
::is_file_handle_valid(const sample_list_conduit_io_handle<sample_name_t>::file_handle_t& h) const {
  return ((h != nullptr) && (h->is_open()));
}

template <typename sample_name_t>
inline typename sample_list_conduit_io_handle<sample_name_t>::file_handle_t sample_list_conduit_io_handle<sample_name_t>
::open_file_handle_for_read(const std::string& file_path) {
  file_handle_t h = new conduit::relay::io::IOHandle;
  h->open(file_path, "hdf5");
  return h;
}

template <typename sample_name_t>
inline void sample_list_conduit_io_handle<sample_name_t>
::close_file_handle(file_handle_t& h) {
  if(is_file_handle_valid(h)) {
    h->close();
  }
}

template <>
inline conduit::relay::io::IOHandle* uninitialized_file_handle<conduit::relay::io::IOHandle*>() {
  return nullptr;
}

template <typename sample_name_t>
inline void sample_list_conduit_io_handle<sample_name_t>
::clear_file_handle(sample_list_conduit_io_handle<sample_name_t>::file_handle_t& h) {
  h = uninitialized_file_handle<file_handle_t>();
}

} // end of namespace lbann

#endif // __SAMPLE_LIST_CONDUIT_IO_HANDLE_HPP__
