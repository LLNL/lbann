#ifndef LBANN_UTILS_SYSTEM_INFO_HPP_INCLUDED
#define LBANN_UTILS_SYSTEM_INFO_HPP_INCLUDED

#include <string>

namespace lbann {
namespace utils {

/** @class SystemInfo
 *  @brief Query basic system information
 *
 *  The class structure here is, strictly speaking, unnecessary. It is
 *  used to provide a "hook" for stubbing this information during
 *  testing.
 */
class SystemInfo
{
public:
  /** @brief Virtual destructor */
  virtual ~SystemInfo() noexcept = default;

  /** @brief Get the current process ID.
   *
   *  This returns the value as a string to avoid system differences
   *  in `pid_t`. However, it's probably safe to return either int64_t
   *  or uint64_t here.
   */
  virtual std::string pid() const;

  /** @brief Get the host name for this process. */
  virtual std::string host_name() const;

  /** @brief Get the MPI rank of this process.
   *
   *  If this is not an MPI job, or cannot be determined to be an MPI
   *  job, this will return 0.
   *
   *  The return type is chosen for consistency with MPI 3.0.
   */
  virtual int mpi_rank() const;

  /** @brief Get the size of the MPI universe in which this process is
   *         participating.
   *
   *  If this is not an MPI job, or cannot be determined to be an MPI
   *  job, this will return 1.
   *
   *  The return type is chosen for consistency with MPI 3.0.
   */
  virtual int mpi_size() const;

  /** @brief Get the value of the given variable from the environment.
   *
   *  If the variable doesn't exist, the empty string is returned.
   */
  virtual std::string env_variable_value(std::string const& var_name) const;

};

}// namespace utils
}// namespace lbann
#endif // LBANN_UTILS_SYSTEM_INFO_HPP_INCLUDED
