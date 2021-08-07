#ifndef LBANN_DATA_UTILS_HDF5_HELPERS_HPP_INCLUDED
#define LBANN_DATA_UTILS_HDF5_HELPERS_HPP_INCLUDED

#include <conduit/conduit_node.hpp>

#include <conduit/conduit_schema.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace data_utils {

/** @brief Split a delimited string into component pieces.
 *  @details Repeated delimiters are treated as one delimiter. Leading
 *           delimiters are dropped.
 *  @param[in] str A view of the delimited string.
 *  @param[in] delimiter The delimiter character. Default: forward slash (/).
 *  @returns A vector storing each component.
 */
std::vector<std::string> split(std::string_view const& str,
                               char delimiter = '/');

/** @brief Get the paths (relative to root) of subtrees that match the
 *         given schema.
 */
std::vector<std::string>
get_matching_node_paths(conduit::Node const& root,
                        conduit::Schema const& prototype_schema);

/** @brief Get a reference to the specified subnode. */
conduit::Node const& get_prototype_sample(conduit::Node const& root,
                                          std::string const& sample_path);

/** @brief Extract the longest prefix from the list of (normalized) paths. */
std::string get_longest_common_prefix(std::vector<std::string> const& paths);

/** @brief Do some normalization of the path.
 *
 *  This is not a complete normalization. Specifically, this just
 *  collapses mutliple consecutive path delimiters into a single
 *  delimiter and removes trailing delimiters. It does not handle
 *  mid-path "dot" or "dot-dot" components. This may change in the
 *  future based on need.
 */
std::string normalize_path(std::string path, char delimiter = '/');

} // namespace data_utils
#endif // LBANN_DATA_UTILS_HDF5_HELPERS_HPP_INCLUDED
