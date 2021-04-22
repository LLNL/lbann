#ifndef LBANN_EXECUTION_ALGORITHMS_LTFB_TERMINATION_CRITERIA_HPP_INCLUDED
#define LBANN_EXECUTION_ALGORITHMS_LTFB_TERMINATION_CRITERIA_HPP_INCLUDED

#include "lbann/execution_algorithms/ltfb/execution_context.hpp"
#include "lbann/execution_contexts/execution_context.hpp"

namespace lbann {
namespace ltfb {

/** @class TerminationCriteria
 *  @brief The stopping criteria for an LTFB-type algorithm
 *
 *  An object here needs to manage
 */
class TerminationCriteria final : public lbann::termination_criteria
{
public:
  TerminationCriteria(size_t max_num_steps)
    : lbann::termination_criteria{max_num_steps}
  {}
  ~TerminationCriteria() = default;
  /** @brief Decide if the criteria are fulfilled. */
  bool operator()(ExecutionContext const& exe_state) const
  {
    return exe_state.get_step() >= this->max_num_steps();
  }

}; // class TerminationCriteria

} // namespace ltfb
} // namespace lbann
#endif // LBANN_EXECUTION_ALGORITHMS_LTFB_TERMINATION_CRITERIA_HPP_INCLUDED
