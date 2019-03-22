#ifndef __SAMPLE_LIST_CONDUIT_IO_HANDLE_HPP__
#define __SAMPLE_LIST_CONDUIT_IO_HANDLE_HPP__

#include "sample_list.hpp"
#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_handle.hpp"

namespace lbann {

template <typename sample_name_t>
class sample_list_conduit_io_handle : public sample_list<conduit::relay::io::IOHandle*, sample_name_t> {
 public:
  using io_t = conduit::relay::io::IOHandle*;
  using typename sample_list<io_t, sample_name_t>::sample_file_id_t;
  using typename sample_list<io_t, sample_name_t>::sample_t;
  using typename sample_list<io_t, sample_name_t>::samples_t;
  using typename sample_list<io_t, sample_name_t>::file_id_stats_t;
  using typename sample_list<io_t, sample_name_t>::file_id_stats_v_t;
  using typename sample_list<io_t, sample_name_t>::fd_use_map_t;

  sample_list_conduit_io_handle();
  ~sample_list_conduit_io_handle() override;

 protected:
  void obtain_sample_names(io_t& h, std::vector<std::string>& sample_names) const override;
  bool is_file_handle_valid(const io_t& h) const override;
  io_t open_file_handle_for_read(const std::string& path) override;
  void close_file_handle(io_t& h) override;
  void clear_file_handle(io_t& h) override;
};


template <typename sample_name_t>
inline sample_list_conduit_io_handle<sample_name_t>::sample_list_conduit_io_handle()
: sample_list<io_t, sample_name_t>() {} 

template <typename sample_name_t>
inline sample_list_conduit_io_handle<sample_name_t>::~sample_list_conduit_io_handle() {
}

template <typename sample_name_t>
inline void sample_list_conduit_io_handle<sample_name_t>
::obtain_sample_names(sample_list_conduit_io_handle<sample_name_t>::io_t& h, std::vector<std::string>& sample_names) const {
  sample_names.clear();
  if (h != nullptr) {
    h->list_child_names("/", sample_names);
  }
}

template <typename sample_name_t>
inline bool sample_list_conduit_io_handle<sample_name_t>
::is_file_handle_valid(const sample_list_conduit_io_handle<sample_name_t>::io_t& h) const {
  return ((h != nullptr) && (h->is_open()));
}

template <typename sample_name_t>
inline typename sample_list_conduit_io_handle<sample_name_t>::io_t sample_list_conduit_io_handle< sample_name_t>
::open_file_handle_for_read(const std::string& file_path) {
  io_t h = new conduit::relay::io::IOHandle;
  h->open(file_path, "hdf5");
  return h;
}

template <typename sample_name_t>
inline void sample_list_conduit_io_handle<sample_name_t>
::close_file_handle(io_t& h) {
  if(is_file_handle_valid(h)) {
    h->close();
  }
}

template <typename sample_name_t>
inline void sample_list_conduit_io_handle<sample_name_t>
::clear_file_handle(sample_list_conduit_io_handle<sample_name_t>::io_t& h) {
  h = nullptr;
}


} // end of namespace lbann

#endif // __SAMPLE_LIST_CONDUIT_IO_HANDLE_HPP__
