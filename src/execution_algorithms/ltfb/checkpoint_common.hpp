#ifndef LBANN_SRC_EXECUTION_ALGORITHMS_LTFB_CHECKPOINT_COMMON_HPP_INCLUDED
#define LBANN_SRC_EXECUTION_ALGORITHMS_LTFB_CHECKPOINT_COMMON_HPP_INCLUDED

#include "lbann/models/model.hpp"
#include "lbann/weights/data_type_weights_impl.hpp"

#include <unordered_set>

namespace lbann {
namespace ltfb {

inline void restore_model_weights(
  model& m,
  std::unordered_map<std::string, std::unique_ptr<weights>>& restore_weights)
{
  // Restore weights that shouldn't be exchanged
  if (restore_weights.empty())
    return;

  // FIXME: Generalize this; enable ptr move??
  for (auto w : m.get_weights()) {
    if (restore_weights.count(w->get_name()) > 0) {
      using TensorDataType = DataType;
      using WeightsType = data_type_weights<TensorDataType>;
      dynamic_cast<WeightsType&>(*w)
        = dynamic_cast<WeightsType&>(*restore_weights[w->get_name()]);
    }
  }
}


} // namespace ltfb
} // namespace lbann
#endif // LBANN_SRC_EXECUTION_ALGORITHMS_LTFB_CHECKPOINT_COMMON_HPP_INCLUDED
