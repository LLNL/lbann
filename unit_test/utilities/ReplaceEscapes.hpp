#ifndef LBANN_UNIT_TEST_UTILITIES_REPLACE_ESCAPES_HPP_INCLUDED
#define LBANN_UNIT_TEST_UTILITIES_REPLACE_ESCAPES_HPP_INCLUDED

#include <lbann/utils/system_info.hpp>

#include <stdexcept>
#include <string>

namespace unit_test
{
namespace utilities
{

/** @brief Substitute basic escape sequences in a string.
 *
 *  The following patterns are supported:
 *
 *  Pattern        | Replacement
 *  -------------- | -----------
 *  %%             | A literal percent sign ("%")
 *  %h             | The hostname of the current process
 *  %p             | The PID of the current process
 *  %r             | The MPI rank of the current process, if available, or 0
 *  %s             | The MPI size of the current job, if available, or 1
 *  %env{\<NAME\>} | The value of ${NAME} in the current environment
 *
 *  The MPI runtime is queried if available for MPI information. After
 *  that, environment variables are checked for common libraries
 *  (SLURM, Open-MPI, MVAPICH2). If neither of these methods gives the
 *  required information, default information consistent with a
 *  sequential job is returned: the rank will be 0 and the size will
 *  be 1.
 *
 *  If the `%env{<NAME>}` substitution fails to find `NAME` in the
 *  current environment, the replacement will be the empty string.
 *
 *  The double-percent sequence is extracted first, so "%%r" will
 *  return "%r" and "%%%r" will return "%\<mpi-rank\>".
 *
 *  @param str The string to which substitutions should be applied.
 *  @param sys_info The source of system information. This is
 *                  primarily exposed for stubbing the functionality
 *                  to test this function.
 *
 *  @throws BadSubstitutionPattern An escape sequence is found in
 *          the string that has no valid substitution.
 *
 *  @returns A copy of the input string with all substitutions applied.
 */
std::string replace_escapes(
  std::string const& str, lbann::utils::SystemInfo const& sys_info);

/** @brief Indicates that an invalid pattern is detected. */
struct BadSubstitutionPattern : std::runtime_error
{
  BadSubstitutionPattern(std::string const& str);
};// struct BadSubstitutionPattern

}// namespace utilities
}// namespace unit_test
#endif // LBANN_UNIT_TEST_UTILITIES_REPLACE_ESCAPES_HPP_INCLUDED
