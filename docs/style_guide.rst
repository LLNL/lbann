LBANN Style Guide
====================

In-Source Documentation
-------------------------

In-source documentation should be written using `Doxygen
<http://www.doxygen.nl/manual/>`_. LBANN will use C-style Doxygen
comments (:code:`/** @brief A short comment */`) and with ampersats
(:code:`@details`) instead of backslashes to denote directives. The
aim is maximal :code:`grep`-ability and readability of the source
code. Using C-style comments for Doxygen helps differentiate quickly
between C-style Doxygen and C++-style source-only documentation.

.. note:: C-style comments on classes and functions default to
          :code:`@details`, *not* :code:`@brief`, even for one-line
          comments. Be sure to add :code:`@brief` when appropriate.

.. _sg-doc-functions:

Documentation of Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every function should be decorated with the maximally applicable set
of the following:

+ :code:`@brief`: A short description of the class. May span multiple
  lines if necessary for maintaining line character limits.

+ :code:`@details`: Begin a detailed description of the function. This is
  not explicitly needed if a blank line is inserted between the
  :code:`@brief` and the body of the :code:`@details` section.

+ :code:`@param <name>` Decribe a parameter to the function. It may be
  helpful to annotate with :code:`@param[in]`, :code:`@param[out]`, or
  :code:`@param[in,out]` if not clear from the types. Repeat this
  directive for each applicable parameter.

+ :code:`@tparam <name>` Describe a template parameter. This can be
  useful for explaining any assumptions (or even better, static
  assertions) about satisfied concepts/predicates. Repeat this
  directive for each applicable template parameter.

+ :code:`@returns` Describe the return value of the function. This is
  not needed for trivial "getters".

+ :code:`@throws <exception>` Indicate an exception that may be
  thrown. It is not expected that every possible exception (e.g.,
  those coming from corner-cases of the STL) be documented. However,
  if a function's implementation has an explicit :code:`throw`
  statement, the exception should be noted with this
  directive. Repeat this directive for each applicable exception.

+ :code:`@pre` Description of preconditions. This is most useful for
  functions that use in/out parameters or those that require various
  conditions on objects in the case of member functions. Repeat this
  directive for each precondition.

+ :code:`@post` Description of postconditions. This is most useful for
  functions that use in/out parameters or those that require various
  conditions on objects in the case of member functions. Repeat this
  directive for each postcondition.

Some hypothetical examples of appropriately marked up functions are:

.. code-block:: c++

    /** @brief Does a foo.
     *
     *  These are details.
     *
     *  @tparam T The type of parameter. Must implement `operator+=`
     *  @param param This is a parameter. It says how to foo.
     *  @throws crazy_error If a crazy error occurs.
     *  @pre param is not foo'd yet
     *  @post param has been foo'd
     */
    template <typename T>
    void foo(T& param);

    /** @brief Computes a result.
     *  @details The algorithm is simple @f$ret=A+B@f$.
     *  @param A the first value
     *  @param B the second value
     *  @returns The output of the complicated algorithm
     */
    int compute_result(int A, int B) noexcept;


Documentation of Classes
~~~~~~~~~~~~~~~~~~~~~~~~~

Every class should be decorated with the maximally applicable set of
the following:

+ :code:`@brief`: A short description of the class. May span multiple
  lines if necessary for maintaining line character limits.

+ :code:`@details`: Begin a detailed description of the function. This is
  not explicitly needed if a blank line is inserted between the
  :code:`@brief` and the body of the :code:`@details` section.

+ :code:`@tparam <name>`: Describe a template parameter. This can be
  useful for explaining any assumptions (or even better, static
  assertions) about satisfied concepts/predicates. Repeat this
  directive for each applicable template parameter.

+ :code:`@name <name>`, :code:`@{`, :code:`@}`: Group members
  into named sections.

Member functions are functions and should be documented as
:ref:`above<sg-doc-functions>`. An example of a completely marked up
file is `include/lbann/utils/any.hpp`.
