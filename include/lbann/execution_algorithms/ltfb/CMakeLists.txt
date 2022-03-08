# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  execution_context.hpp
  meta_learning_strategy.hpp
  mutation_strategy.hpp
  random_pairwise_exchange.hpp
  regularized_evolution.hpp
  termination_criteria.hpp
  timing_visitor.hpp
  truncation_selection_exchange.hpp
  visitor.hpp
  visitor_factory.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
