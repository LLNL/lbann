#ifndef LBANN_UTILS_ETI_MACROS_HPP_INCLUDED
#define LBANN_UTILS_ETI_MACROS_HPP_INCLUDED

#define LBANN_CLASS_ETI_SIGNATURE(class_name, ...)      \
  template class class_name<__VA_ARGS__>

#define LBANN_CLASS_ETI_DECL(...)               \
  extern LBANN_CLASS_ETI_SIGNATURE(__VA_ARGS__)

#define LBANN_CLASS_ETI_INST(...)               \
  LBANN_CLASS_ETI_SIGNATURE(__VA_ARGS__)

#endif // LBANN_UTILS_ETI_MACROS_HPP_INCLUDED
