#include "helpers.hpp"

#include <regex>
#include <stdexcept>

namespace {
void get_matching_node_paths_impl(conduit::Node const& node,
                                  conduit::Schema const& prototype,
                                  std::vector<std::string>& sample_ids)
{
  if (node.schema().equals(prototype)) {
    sample_ids.push_back(node.path());
    return; // Break the recursion
  }

  // recursion
  conduit::NodeConstIterator child = node.children();
  while (child.has_next()) {
    get_matching_node_paths_impl(child.next(), prototype, sample_ids);
  }
}

bool is_rooted_path(std::string_view const& path, char delimiter) noexcept
{
  return path.front() == delimiter;
}

} // namespace

std::vector<std::string> data_utils::split(std::string_view const& str,
                                           char delimiter)
{
  using SizeT = std::string_view::size_type;

  // Special case:
  if (str.empty())
    return {};

  std::vector<std::string> out;
  SizeT start = 0;
  do {
    auto const stop = str.find(delimiter, start);
    auto const count =
      (stop == std::string_view::npos ? str.length() - start : stop - start);
    if (count != 0)
      out.emplace_back(str.substr(start, count));
    start += count + 1;
  } while (start < str.length());
  return out;
}

std::vector<std::string>
data_utils::get_matching_node_paths(conduit::Node const& root,
                                    conduit::Schema const& prototype_schema)
{
  std::vector<std::string> out;
  get_matching_node_paths_impl(root, prototype_schema, out);
  return out;
}

conduit::Node const&
data_utils::get_prototype_sample(conduit::Node const& root,
                                 std::string const& sample_path)
{
  return root.fetch_existing(sample_path);
}

std::string
data_utils::get_longest_common_prefix(std::vector<std::string> const& paths)
{
  using SizeT = std::string_view::size_type;

  // Short-circuit
  if (paths.empty()) {
    return "";
  }

  auto const path_view = std::string_view(paths.front());

  // Check some boundary cases
  {
    SizeT const full_prefix_size = path_view.rfind('/');

    // Just a file name. Assume relative to ".".
    if (full_prefix_size == path_view.npos)
      return ".";

    // Single file, so it's longest prefix is the longest prefix.
    if (paths.size() == 1)
      return std::string(path_view.substr(0, full_prefix_size));

    bool const rootedness = is_rooted_path(path_view, '/');
    if (!std::all_of(cbegin(paths), cend(paths), [rootedness](auto const& p) {
          return (is_rooted_path(p, '/') == rootedness);
        })) {
      throw std::runtime_error("detected mix of rooted and nonrooted paths");
    }
  }

  // Full search.
  SizeT token_start = path_view.front() == '/' ? 1 : 0;
  SizeT const max_path_length = path_view.length();
  std::string prefix;
  auto const num_paths = paths.size();
  do {
    auto const token_stop = path_view.find('/', token_start);
    if (token_stop == path_view.npos)
      break;

    auto const count = token_stop - token_start;
    auto const token = path_view.substr(token_start, count);

    // Check that the same token exists in every other path.
    bool done = false;
    for (size_t p = 1; p < num_paths; ++p) {
      auto const other = std::string_view(paths[p]).substr(token_start, count);

      // Search is over if any path doesn't match.
      if (other != token)
        done = true;
    }
    if (done)
      break;
    else
      token_start += count + 1;
  } while (token_start < max_path_length);

  // Empty prefixes return "."
  if (token_start == 0UL)
    return ".";
  else
    return normalize_path(std::string(path_view.substr(0, token_start)));
}

std::string data_utils::normalize_path(std::string path, char delimiter_char)
{
  if (path.empty())
    return path;

  // Squash all delimiters down to one.
  std::string const delimiter(&delimiter_char, 1);
  std::regex const multi_delimiter_re(delimiter + "+");
  path = std::regex_replace(path, multi_delimiter_re, delimiter);

  if (path.length() > 1 && path.back() == delimiter_char)
    path.pop_back();
  if (path.empty())
    return ".";
  return path;
}
