namespace lbann {

template <typename sample_name_t, typename file_handle_t>
inline sample_list_open_files<sample_name_t, file_handle_t>::sample_list_open_files() {
  m_max_open_files = getdtablesize() - LBANN_MAX_OPEN_FILE_MARGIN;
}

template <typename sample_name_t, typename file_handle_t>
inline sample_list_open_files<sample_name_t, file_handle_t>::~sample_list_open_files() {
  m_open_fd_pq.clear();
}

template <typename sample_name_t, typename file_handle_t>
inline sample_list_open_files<sample_name_t, file_handle_t>
::sample_list_open_files(const sample_list_open_files& rhs) {
  copy_members(rhs);
}

template <typename sample_name_t, typename file_handle_t>
inline sample_list_open_files<sample_name_t, file_handle_t>&
sample_list_open_files<sample_name_t, file_handle_t>
::operator=(const sample_list_open_files& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }

  copy_members(rhs);

  return (*this);
}

template <typename sample_name_t, typename file_handle_t>
inline sample_list_open_files<sample_name_t, file_handle_t>&
sample_list_open_files<sample_name_t, file_handle_t>
::copy(const sample_list_open_files& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }

  copy_members(rhs);

  return (*this);
}

template <typename sample_name_t, typename file_handle_t>
inline void sample_list_open_files<sample_name_t, file_handle_t>
::copy_members(const sample_list_open_files& rhs) {
  sample_list<sample_name_t>::copy_members(rhs);
  m_sample_list = rhs.m_sample_list;
  m_file_map = rhs.m_file_map;
  m_max_open_files = rhs.m_max_open_files;

  /// Keep track of existing filenames but do not copy any file
  /// descriptor information
  m_file_id_stats_map.assign(rhs.m_file_id_stats_map.size(),
                             std::make_tuple("",
                                             uninitialized_file_handle<file_handle_t>(),
                                             std::deque<std::pair<int,int>>{}));

  for(size_t i = 0u; i < m_file_id_stats_map.size(); ++i) {
    set_samples_filename(i, rhs.get_samples_filename(i));
  }

  /// Do not copy the open file descriptor priority queue
  /// File handle ownership is not transfered in the copy
  m_open_fd_pq.clear();
}

template <typename sample_name_t, typename file_handle_t>
inline size_t sample_list_open_files<sample_name_t, file_handle_t>
::size() const {
  return m_sample_list.size();
}

template <typename sample_name_t, typename file_handle_t>
inline size_t sample_list_open_files<sample_name_t, file_handle_t>
::get_num_files() const {
  return m_file_id_stats_map.size();
}

template <typename sample_name_t, typename file_handle_t>
inline void sample_list_open_files<sample_name_t, file_handle_t>
::read_exclusive_list(std::istream& istrm,
                      size_t stride, size_t offset) {
  const std::string whitespaces(" \t\f\v\n\r");
  size_t cnt_files = 0u;
  std::string line;

  while (std::getline(istrm, line)) {
    const size_t end_of_str = line.find_last_not_of(whitespaces);
    if (end_of_str == std::string::npos) { // empty line
      continue;
    }
    if (cnt_files++ >= m_header.get_num_files()) {
      break;
    }
    // Check to see if there is a strided load and skip the lines that are not for this rank
    if ((cnt_files-1)%stride != offset) {
      continue;
    }

    std::stringstream sstr(line.substr(0, end_of_str + 1)); // clear trailing spaces for accurate parsing
    std::string filename;
    size_t included_samples;
    size_t excluded_samples;
    std::unordered_set<std::string> excluded_sample_indices;

    sstr >> filename >> included_samples >> excluded_samples;

    const std::string file_path = add_delimiter(m_header.get_file_dir()) + filename;

    if (filename.empty() || !check_if_file_exists(file_path)) {
      LBANN_ERROR(std::string{} + " :: data file '" + file_path + "' does not exist.");
    }

    excluded_sample_indices.reserve(excluded_samples);

    while(!sstr.eof()) {
      std::string index;
      sstr >> index;
      excluded_sample_indices.insert(index);
    }

    if(excluded_sample_indices.size() != excluded_samples) {
      LBANN_ERROR(std::string("Index file does not contain the correct number of excluded samples: expected ")
                  + std::to_string(excluded_samples)
                  + std::string(" exclusions but found ")
                  + std::to_string(excluded_sample_indices.size()));
    }

    std::vector<std::string> sample_names;
    file_handle_t file_hnd = get_bundled_sample_names(file_path, sample_names, included_samples, excluded_samples);
    if (!is_file_handle_valid(file_hnd)) {
      continue; // skipping the file
    }

    if(m_file_map.count(filename) > 0) {
      if(sample_names.size() != m_file_map[filename]) {
        LBANN_ERROR(std::string("The same file ")
                    + filename
                    + " was opened multiple times and reported different sizes: "
                    + std::to_string(sample_names.size())
                    + " and "
                    + std::to_string(m_file_map[filename]));
      }
    }else {
      m_file_map[filename] = sample_names.size();
    }

    sample_file_id_t index = m_file_id_stats_map.size();
    m_file_id_stats_map.emplace_back(std::make_tuple(filename, uninitialized_file_handle<file_handle_t>(), std::deque<std::pair<int,int>>{}));
    set_files_handle(filename, file_hnd);

    size_t valid_sample_count = 0u;
    for(auto s : sample_names) {
      std::unordered_set<std::string>::const_iterator found = excluded_sample_indices.find(s);
      if (found != excluded_sample_indices.cend()) {
        continue;
      }
      m_sample_list.emplace_back(index, to_sample_name_t<sample_name_t>(s));
      valid_sample_count++;
    }

    if(valid_sample_count != included_samples) {
      LBANN_ERROR(std::string("Bundle file does not contain the correct number of included samples: expected ")
                  + std::to_string(included_samples)
                  + std::string(" samples, but found ")
                  + std::to_string(valid_sample_count));
    }
  }

  if (m_header.get_num_files() != cnt_files) {
    LBANN_ERROR(std::string("Sample list ")
                + m_header.get_sample_list_filename()
                + std::string(": number of files requested ")
                + std::to_string(m_header.get_num_files())
                + std::string(" does not equal number of files loaded ")
                + std::to_string(cnt_files));
  }

  m_header.m_is_exclusive = false;
}


template <typename sample_name_t, typename file_handle_t>
inline void sample_list_open_files<sample_name_t, file_handle_t>
::read_inclusive_list(std::istream& istrm,
                      size_t stride, size_t offset) {
  const std::string whitespaces(" \t\f\v\n\r");
  size_t cnt_files = 0u;
  std::string line;

  while (std::getline(istrm, line)) {
    const size_t end_of_str = line.find_last_not_of(whitespaces);
    if (end_of_str == std::string::npos) { // empty line
      continue;
    }
    if (cnt_files++ >= m_header.get_num_files()) {
      break;
    }
    // Check to see if there is a strided load and skip the lines that are not for this rank
    if ((cnt_files-1)%stride != offset) {
      continue;
    }

    std::stringstream sstr(line.substr(0, end_of_str + 1)); // clear trailing spaces for accurate parsing
    std::string filename;
    size_t included_samples;
    size_t excluded_samples;

    sstr >> filename >> included_samples >> excluded_samples;

    const std::string file_path = add_delimiter(m_header.get_file_dir()) + filename;

    if (filename.empty() || !check_if_file_exists(file_path)) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                            + " :: data file '" + filename + "' does not exist.");
    }

    std::vector<std::string> sample_names;
    file_handle_t file_hnd = get_bundled_sample_names(file_path, sample_names, included_samples, excluded_samples);
    if (!is_file_handle_valid(file_hnd)) {
      continue; // skipping the file
    }

    if(m_file_map.count(filename) > 0) {
      if(sample_names.size() != m_file_map[filename]) {
        LBANN_ERROR(std::string("The same file ")
                    + filename
                    + " was opened multiple times and reported different sizes: "
                    + std::to_string(sample_names.size())
                    + " and "
                    + std::to_string(m_file_map[filename]));
      }
    }else {
      m_file_map[filename] = sample_names.size();
    }

    std::unordered_set<std::string> set_of_samples(sample_names.begin(), sample_names.end());

    sample_file_id_t index = m_file_id_stats_map.size();
    m_file_id_stats_map.emplace_back(std::make_tuple(filename, uninitialized_file_handle<file_handle_t>(), std::deque<std::pair<int,int>>{}));
    set_files_handle(filename, file_hnd);

    size_t valid_sample_count = 0u;
    while(!sstr.eof()) {
      std::string sample_name_str;
      sstr >> sample_name_str;
      std::unordered_set<std::string>::const_iterator found = set_of_samples.find(sample_name_str);
      if (found == set_of_samples.cend()) {
        LBANN_ERROR(std::string("Illegal request for a data ID that does not exist: ") + sample_name_str);
      }
      m_sample_list.emplace_back(index, to_sample_name_t<sample_name_t>(sample_name_str));
      valid_sample_count++;
    }
    if(valid_sample_count != included_samples) {
      LBANN_ERROR(std::string("Bundle file does not contain the correct number of included samples: expected ")
                  + std::to_string(included_samples)
                  + std::string(" samples, but found ")
                  + std::to_string(valid_sample_count));
    }
  }

  if (m_header.get_num_files() != cnt_files) {
    LBANN_ERROR(std::string("Sample list number of files requested ")
                + std::to_string(m_header.get_num_files())
                + std::string(" does not equal number of files loaded ")
                + std::to_string(cnt_files));
  }
}


template <typename sample_name_t, typename file_handle_t>
inline void sample_list_open_files<sample_name_t, file_handle_t>
::read_sample_list(std::istream& istrm, size_t stride, size_t offset) {
  if (m_header.is_exclusive()) {
    read_exclusive_list(istrm, stride, offset);
  } else {
    read_inclusive_list(istrm, stride, offset);
  }
}


template <typename sample_name_t, typename file_handle_t>
template <class Archive>
void sample_list_open_files<sample_name_t, file_handle_t>
::save( Archive & ar ) const {
  using ar_file_stats_t = std::tuple<std::string, std::deque<std::pair<int,int>>>;
  std::vector<ar_file_stats_t> file_stats;
  file_stats.reserve(m_file_id_stats_map.size());
  for(auto&& e : m_file_id_stats_map) {
    file_stats.emplace_back(std::make_tuple(std::get<0>(e), std::get<2>(e)));
  }
  ar(m_header, m_sample_list, file_stats);
}

template <typename sample_name_t, typename file_handle_t>
template <class Archive>
void sample_list_open_files<sample_name_t, file_handle_t>
::load( Archive & ar ) {
  using ar_file_stats_t = std::tuple<std::string, std::deque<std::pair<int,int>>>;
  std::vector<ar_file_stats_t> file_stats;
  ar(m_header, m_sample_list, file_stats);
  m_file_id_stats_map.reserve(file_stats.size());
  for(auto&& e : file_stats) {
    //m_file_id_stats_map.emplace_back(std::make_tuple(std::get<0>(e), uninitialized_file_handle<file_handle_t>(), std::deque<std::pair<int,int>>{}));
    m_file_id_stats_map.emplace_back(std::make_tuple(std::get<0>(e), uninitialized_file_handle<file_handle_t>(), std::get<1>(e)));
    //m_file_id_stats_map.emplace_back(std::make_tuple(std::get<0>(e), file_handle_t(), std::get<1>(e)));
  }
}

template <typename sample_name_t, typename file_handle_t>
inline bool sample_list_open_files<sample_name_t, file_handle_t>
::to_string(std::string& sstr) const {
  std::map<std::string, std::template vector<sample_name_t>> tmp_file_map;
  for (const auto& s : m_sample_list) {
    const std::string& filename = get_samples_filename(s.first);
    tmp_file_map[filename].emplace_back(s.second);
  }

  sstr.clear();

  // reserve the string to hold the entire sample lit
  size_t estimated_len = 30 + 42 + m_header.get_file_dir().size() + 1;
  for (const auto& f : tmp_file_map) {
    estimated_len += f.first.size()
                   + std::to_string(f.second.size()).size()
                   + std::to_string(m_file_map.at(f.first) - f.second.size()).size()
                   + 3u;
    for (const auto& s : f.second) {
      estimated_len += lbann::to_string(s).size() + 1u;
    }
  }
  sstr.reserve(estimated_len);

  // write the list header
  this->write_header(sstr, tmp_file_map.size());

  // write the list body
  for (const auto& f : tmp_file_map) {
    // File name
    sstr += f.first;
    // Number of included samples
    sstr += std::string(" ") + std::to_string(f.second.size());
    // Number of excluded samples
    sstr += std::string(" ") + std::to_string(m_file_map.at(f.first) - f.second.size());
    // Inclusion sample list
    for (const auto& s : f.second) {
      sstr += ' ' + lbann::to_string(s);
    }
    sstr += '\n';
  }

  return true;
}

template <typename sample_name_t, typename file_handle_t>
inline void sample_list_open_files<sample_name_t, file_handle_t>
::get_num_samples(size_t& total, size_t& included, size_t& excluded) const {
  total = 0u;
  for ( const auto f: m_file_map) {
    total += f.second;
  }
  included = size();
  excluded = total - included;
}

template <typename sample_name_t, typename file_handle_t>
inline const typename sample_list_open_files<sample_name_t, file_handle_t>::samples_t&
sample_list_open_files<sample_name_t, file_handle_t>::get_list() const {
  return m_sample_list;
}

template <typename sample_name_t, typename file_handle_t>
inline const typename sample_list_open_files<sample_name_t, file_handle_t>::sample_t&
sample_list_open_files<sample_name_t, file_handle_t>::operator[](size_t idx) const {
  return m_sample_list[idx];
}

template <typename sample_name_t, typename file_handle_t>
inline const std::string& sample_list_open_files<sample_name_t, file_handle_t>
::get_samples_filename(sample_file_id_t id) const {
  return std::get<0>(m_file_id_stats_map[id]);
}

template <typename sample_name_t, typename file_handle_t>
inline file_handle_t sample_list_open_files<sample_name_t, file_handle_t>
::get_samples_file_handle(sample_file_id_t id) const {
  file_handle_t h = std::get<1>(m_file_id_stats_map[id]);
  return h;
}

template <typename sample_name_t, typename file_handle_t>
inline void sample_list_open_files<sample_name_t, file_handle_t>
::set_samples_filename(sample_file_id_t id, const std::string& filename) {
  std::get<0>(m_file_id_stats_map[id]) = filename;
}

template <typename sample_name_t, typename file_handle_t>
inline void sample_list_open_files<sample_name_t, file_handle_t>
::set_files_handle(const std::string& filename, file_handle_t h) {
  sample_file_id_t id = sample_file_id_t(0);
  for (auto&& e : m_file_id_stats_map) {
    if(std::get<0>(e) == filename) {
      std::get<1>(e) = h;
      break;
    }
    id++;
  }
  manage_open_file_handles(id, true);
}

template <typename sample_name_t, typename file_handle_t>
inline void sample_list_open_files<sample_name_t, file_handle_t>
::obtain_sample_names(file_handle_t& h, std::vector<std::string>& sample_names) const {
  LBANN_ERROR(std::string{} + " :: abstract class does not implement this method");
}

template <typename sample_name_t, typename file_handle_t>
inline file_handle_t sample_list_open_files<sample_name_t, file_handle_t>
::get_bundled_sample_names(std::string file_path,
                           std::vector<std::string>& sample_names,
                           size_t included_samples,
                           size_t excluded_samples) {
  file_handle_t file_hnd;
  clear_file_handle(file_hnd);
  bool retry = false;
  int retry_cnt = 0;
  do {
    try {
      file_hnd = open_file_handle_for_read( file_path );
    }catch (conduit::Error const& e) {
      LBANN_WARNING(" :: trying to open the file " + file_path + " and got " + e.what());
      retry = true;
      retry_cnt++;
    }
  }while(retry && retry_cnt < LBANN_MAX_OPEN_FILE_RETRY);

  if (!is_file_handle_valid(file_hnd)) {
    std::cout << "Opening the file didn't work" << std::endl;
    return file_hnd;
  }

  obtain_sample_names(file_hnd, sample_names);

  if(sample_names.size() != (included_samples + excluded_samples)) {
    LBANN_ERROR(std::string("File does not contain the correct number of samples: found ")
                + std::to_string(sample_names.size())
                + std::string(" -- this does not equal the expected number of samples that are marked for inclusion: ")
                + std::to_string(included_samples)
                + std::string(" and exclusion: ")
                + std::to_string(excluded_samples));
  }

  return file_hnd;
}

template <typename sample_name_t, typename file_handle_t>
inline void sample_list_open_files<sample_name_t, file_handle_t>
::all_gather_packed_lists(lbann_comm& comm) {
  int num_ranks = comm.get_procs_per_trainer();
  typename std::vector<samples_t> per_rank_samples(num_ranks);
  typename std::vector<std::vector<std::string>> per_rank_files(num_ranks);
  std::vector<std::string> my_files;
  my_files.reserve(m_file_id_stats_map.size());
  std::vector<std::unordered_map<std::string, size_t>> per_rank_file_map(num_ranks);

  // Close the existing open files
  for(auto&& e : m_file_id_stats_map) {
    auto& h = std::get<1>(e);
    close_file_handle(h);
    clear_file_handle(h);
    std::get<2>(e).clear();
    my_files.emplace_back(std::get<0>(e));
  }
  m_open_fd_pq.clear();

  size_t num_samples = this->all_gather_field(m_sample_list, per_rank_samples, comm);
  size_t num_ids = this->all_gather_field(my_files, per_rank_files, comm);
  size_t num_files = this->all_gather_field(m_file_map, per_rank_file_map, comm);

  m_sample_list.clear();
  m_file_id_stats_map.clear();

  m_sample_list.reserve(num_samples);
  m_file_id_stats_map.reserve(num_ids);
  m_file_map.reserve(num_files);

  for(int r = 0; r < num_ranks; r++) {
    const samples_t& s_list = per_rank_samples[r];
    const auto& files = per_rank_files[r];
    const std::unordered_map<std::string, size_t>& file_map = per_rank_file_map[r];
    for (const auto& s : s_list) {
      sample_file_id_t index = s.first;
      const std::string& filename = files[index];
      if(index >= m_file_id_stats_map.size()
         || (std::get<0>(m_file_id_stats_map.back()) != filename)) {
        index = m_file_id_stats_map.size();
        m_file_id_stats_map.emplace_back(std::make_tuple(filename, uninitialized_file_handle<file_handle_t>(), std::deque<std::pair<int,int>>{}));
        // Update the file map structure
        if(m_file_map.count(filename) == 0) {
          m_file_map[filename] = file_map.at(filename);
        }
      }else {
        for(size_t i = 0; i < m_file_id_stats_map.size(); i++) {
          if(filename == get_samples_filename(i)) {
            index = i;
            break;
          }
        }
      }
      m_sample_list.emplace_back(std::make_pair(index, s.second));
    }
  }

  return;
}

template <typename sample_name_t, typename file_handle_t>
inline void sample_list_open_files<sample_name_t, file_handle_t>
::compute_epochs_file_usage(const std::vector<int>& shuffled_indices,
                            int mini_batch_size,
                            const lbann_comm& comm) {
  for (auto&& e : m_file_id_stats_map) {
    auto& h = std::get<1>(e);
    close_file_handle(h);
    clear_file_handle(h);
    std::get<2>(e).clear();
  }
  // Once all of the file handles are closed, clear the priority queue
  m_open_fd_pq.clear();
  for (size_t i = 0; i < shuffled_indices.size(); i++) {
    int idx = shuffled_indices[i];
    const auto& s = m_sample_list[idx];
    sample_file_id_t index = s.first;

    if((i % mini_batch_size) % comm.get_procs_per_trainer() == static_cast<size_t>(comm.get_rank_in_trainer())) {
      /// Enqueue the iteration step when the sample will get used
      int step = i / mini_batch_size;
      int substep = (i % mini_batch_size) / comm.get_procs_per_trainer();
      std::get<2>(m_file_id_stats_map[index]).emplace_back(std::make_pair(step, substep));
    }
  }
}

template <typename sample_name_t, typename file_handle_t>
inline void sample_list_open_files<sample_name_t, file_handle_t>
::delete_file_handle_pq_entry(sample_file_id_t id) {
  for (std::deque<fd_use_map_t>::iterator it = m_open_fd_pq.begin(); it!=m_open_fd_pq.end(); ++it) {
    if(it->first == id) {
      it = m_open_fd_pq.erase(it);
      break;
    }
  }
  return;
}

template <typename sample_name_t, typename file_handle_t>
inline void sample_list_open_files<sample_name_t, file_handle_t>
::manage_open_file_handles(sample_file_id_t id, bool pre_open_fd) {
  /// When we enter this function the priority queue is either empty or a heap
  if(!m_open_fd_pq.empty()) {
    if(m_open_fd_pq.size() > m_max_open_files) {
      auto& f = m_open_fd_pq.front();
      auto& victim = m_file_id_stats_map[f.first];
      auto& victim_fd = std::get<1>(victim);
      std::pop_heap(m_open_fd_pq.begin(), m_open_fd_pq.end(), pq_cmp);
      m_open_fd_pq.pop_back();
      close_file_handle(victim_fd);
      clear_file_handle(victim_fd);
    }
  }

  /// Before we can enqueue the any new access times for this descriptor, remove any
  /// earlier descriptor
  std::sort_heap(m_open_fd_pq.begin(), m_open_fd_pq.end(), pq_cmp);
  if(m_open_fd_pq.front().first == id) {
    m_open_fd_pq.pop_front();
  }
  std::make_heap(m_open_fd_pq.begin(), m_open_fd_pq.end(), pq_cmp);

  auto& e = m_file_id_stats_map[id];
  auto& file_access_queue = std::get<2>(e);
  if(!file_access_queue.empty()) {
    if(!pre_open_fd) {
      file_access_queue.pop_front();
    }
  }
  if(!file_access_queue.empty()) {
    m_open_fd_pq.emplace_back(std::make_pair(id,file_access_queue.front()));
  }else {
    /// If there are no future access of the file place a terminator entry to track
    /// the open file, but is always sorted to the top of the heap
    m_open_fd_pq.emplace_back(std::make_pair(id,std::make_pair(INT_MAX,id)));
  }
  std::push_heap(m_open_fd_pq.begin(), m_open_fd_pq.end(), pq_cmp);
  return;
}

template <typename sample_name_t, typename file_handle_t>
inline file_handle_t sample_list_open_files<sample_name_t, file_handle_t>
::open_samples_file_handle(const size_t i, bool pre_open_fd) {
  const sample_t& s = m_sample_list[i];
  sample_file_id_t id = s.first;
  file_handle_t h = get_samples_file_handle(id);
  if (!is_file_handle_valid(h)) {
    const std::string& file_name = get_samples_filename(id);
    const std::string& file_dir = this->get_samples_dirname();
    const std::string file_path = add_delimiter(file_dir) + file_name;
    if (file_name.empty() || !check_if_file_exists(file_path)) {
      LBANN_ERROR(std::string{} + " :: data file '" + file_path + "' does not exist.");
    }
    bool retry = false;
    int retry_cnt = 0;
    do {
      try {
        h = open_file_handle_for_read( file_path );
      }catch (conduit::Error const& e) {
        LBANN_WARNING(" :: trying to open the file " + file_path + " and got " + e.what());
        retry = true;
        retry_cnt++;
      }
    }while(retry && retry_cnt < 3);

    if (!is_file_handle_valid(h)) {
      LBANN_ERROR(std::string{} + " :: data file '" + file_path + "' could not be opened.");
    }
    auto& e = m_file_id_stats_map[id];
    std::get<1>(e) = h;
    /// If a new file is opened, place it in the priority queue
    manage_open_file_handles(id, pre_open_fd);
  }
  return h;
}

template <typename sample_name_t, typename file_handle_t>
inline void sample_list_open_files<sample_name_t, file_handle_t>
::close_if_done_samples_file_handle(const size_t i) {
  const sample_t& s = m_sample_list[i];
  sample_file_id_t id = s.first;
  auto h = get_samples_file_handle(id);
  if (!is_file_handle_valid(h)) {
    auto& e = m_file_id_stats_map[id];
    auto& file_access_queue = std::get<2>(e);
    if(file_access_queue.empty()) {
      auto& fh = std::get<1>(e);
      close_file_handle(fh);
      clear_file_handle(fh);
      delete_file_handle_pq_entry(id);
    }
  }
}

template <typename sample_name_t, typename file_handle_t>
inline bool sample_list_open_files<sample_name_t, file_handle_t>
::is_file_handle_valid(const file_handle_t& h) const {
  LBANN_ERROR(std::string{} + " :: abstract class does not implement this method");
  return false;
}

template <typename sample_name_t, typename file_handle_t>
inline file_handle_t sample_list_open_files<sample_name_t, file_handle_t>
::open_file_handle_for_read(const std::string& file_path) {
  LBANN_ERROR(std::string{} + " :: abstract class does not implement this method");
  return file_handle_t();
}

template <typename sample_name_t, typename file_handle_t>
inline void sample_list_open_files<sample_name_t, file_handle_t>
::close_file_handle(file_handle_t& h) {
  LBANN_ERROR(std::string{} + " :: abstract class does not implement this method");
}

template <typename sample_name_t, typename file_handle_t>
inline void sample_list_open_files<sample_name_t, file_handle_t>
::clear_file_handle(file_handle_t& h) {
  LBANN_ERROR(std::string{} + " :: abstract class does not implement this method");
}

} // end of namespace lbann
