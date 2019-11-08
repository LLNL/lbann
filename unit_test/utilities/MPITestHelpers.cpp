#include "MPITestHelpers.hpp"

namespace unit_test {
namespace utilities {
namespace {
lbann::lbann_comm* global_comm_;
}

lbann::lbann_comm& current_world_comm()
{
  LBANN_ASSERT_POINTER(global_comm_);
  return *global_comm_;
}

namespace expert {
void register_world_comm(lbann::lbann_comm& comm) noexcept
{
  global_comm_ = &comm;
}

void reset_world_comm() noexcept
{
  global_comm_ = nullptr;
}
} // namespace expert
} // namespace utilities
} // namespace unit_test
