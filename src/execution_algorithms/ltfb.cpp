#include "lbann/execution_algorithms/ltfb.hpp"
#include "lbann/base.hpp"
#include "lbann/execution_algorithms/factory.hpp"
#include "lbann/execution_algorithms/ltfb/meta_learning_strategy.hpp"
#include "lbann/execution_algorithms/ltfb/termination_criteria.hpp"
#include "lbann/models/model.hpp"

#include "training_algorithm.pb.h"

// FIXME (trb 04/14/21): This code is copied with only minimal
// modification from the LTFB callback implementation. It should be
// reviewed for any potential simplification and/or optimization.

namespace lbann {

void LTFB::apply(execution_context& context,
                 model& m,
                 data_coordinator& dc,
                 execution_mode /*mode*/)
{
  auto const& ltfb_term = m_termination_criteria;
  auto& ltfb_ctxt = dynamic_cast<ExeContextType&>(context);

  // Sync trainers (Assumption: all trainers in this lbann_comm are
  // participating in this training algorithm)
  m.get_comm()->intertrainer_barrier();

  // LTFB likely has different stopping criteria than SGD (e.g., K
  // tournament rounds; some specified relative or absolute
  // reduction in objective function value; etc.), or its stopping
  // criteria might be defined in terms of the SGD stopping criteria
  // (e.g., N total sgd batches). That complexity lives in the
  // ltfb::TerminationCriteria class.
  while (!ltfb_term(ltfb_ctxt)) {
    m_local_algo->apply(m, dc);
    m_meta_learning_strategy->select_next(m, ltfb_ctxt, dc);
    ltfb_ctxt.inc_step();
  }

  // TODO: How do we support aggregate outputs? What does "output"
  // mean here? Do we communicate among all trainers and just write
  // some interesting subset to disk? Top-k best models, e.g.
  //
  // maybe:
  //
  // intertrainer_postprocess(m);
}

} // namespace lbann

// build_abstract impl for metalearning strategies
template <>
std::unique_ptr<lbann::LTFB>
lbann::make<lbann::LTFB>(google::protobuf::Message const& msg_in)
{
  auto const& msg = dynamic_cast<lbann_data::TrainingAlgorithm const&>(msg_in);

  // Extract the solver parameters.
  lbann_data::LTFB params;
  LBANN_ASSERT(msg.parameters().UnpackTo(&params));

  auto const& stopping = params.stopping_criteria();
  return make_unique<LTFB>(
    msg.name(),
    make_abstract<training_algorithm>(params.local_training_algorithm()),
    make_abstract<ltfb::MetaLearningStrategy>(params.meta_learning_strategy()),
    ltfb::TerminationCriteria{stopping.max_tournaments()});
}
