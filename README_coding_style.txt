For LBANN coding style we are going to use one that is based on the standard C++ libraries and Boost projects.

* overriding style is lowercase separated with underbar
* member fields: m_*
* function names, class names: lowercase with underbar
* templated types: start with uppercase (camel case after that e.g. DataType)
  * derived typedef types: typedef DataType::value_type value_type
* header preprocessor guard: <NAMESPACE_PATH_NAME>_HPP_INCLUDED (e.g. LBANN_LAYERS_FULLYCONNECTED_HPP_INCLUDED)
* 2 space, no tabs
* comments:
  * doxygen:  
    * /// single line comment 
    * /** multi-line comment */
    * @todo  - TODO note
  * inside of a function use //
  * outside use a doxygen comment
    * minimize blocks of // or /* */ comments that are not picked up by doxygen
  * If you have a complicated function or algorithm, either explain it in a doxygen comment or cite an appropriate reference
  * if the implementation details or meta-parameters are derived from another opensource ML toolkit - cite it
* global variable - see above about lowercase with underbar
