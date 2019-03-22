#ifndef __LBANN_TYPE_ERASED_FUNCTION_HPP__
#define __LBANN_TYPE_ERASED_FUNCTION_HPP__

#include <type_traits>

#include <lbann/utils/memory.hpp>

namespace lbann {

/** @class type_erased_function
 *  @brief A move-only callable type for wrapping functions
 */
class type_erased_function {
public:

  /** @brief Erase the type of input function F */
  template <typename FunctionT>
  type_erased_function(FunctionT&& F)
    : held_function_(make_unique<Function<FunctionT>>(std::move(F))) {}

  /** @brief Move constructor */
  type_erased_function(type_erased_function&& other) = default;

  /** @brief Move assignment */
  type_erased_function& operator=(type_erased_function&& other) = default;

  /** @brief Make the function callable */
  void operator()() { held_function_->call_held(); }

  /** @name Deleted functions */
  ///@{

  /** @brief Deleted constructor */
  type_erased_function() = delete;

  /** @brief Deleted copy constructor */
  type_erased_function(const type_erased_function& other) = delete;

  /** @brief Deleted copy assignment */
  type_erased_function& operator=(const type_erased_function& other) = delete;

  ///@}

private:
  /** @name Type erasure template types */
  ///@{

  /** @class FunctionHolder
   *  @brief Simple function object holder
   */
  struct FunctionHolder
  {
    /** @brief Destructor */
    virtual ~FunctionHolder() = default;

    /** @brief Call the held function */
    virtual void call_held() = 0;
  };

  /** @class Function
   *  @brief A wrapper for a specific type of function
   *
   *  @tparam FunctionT Must be MoveConstructible and Callable
   */
  template <typename FunctionT>
  struct Function : FunctionHolder
  {
    static_assert(std::is_move_constructible<FunctionT>::value,
                  "Given type is not move constructible!");

    /** @brief Construct by moving from the input function type */
    Function(FunctionT&& f)
      : F__(std::move(f)) {}

    /** @brief Destructor */
    ~Function() = default;

    /** @brief Call the held function */
    void call_held() override { F__(); }

    /** @brief The held function */
    FunctionT F__;
  };
  ///@}

  /** @brief A type-erased function */
  std::unique_ptr<FunctionHolder> held_function_;
};// class type_erased_function

}// namespace lbann
#endif /* __LBANN_TYPE_ERASED_FUNCTION_HPP__ */
