function (write_python_features_module)
  set(_OPTIONS)
  set(_SINGLE_VAL_PARAMS OUTPUT_FILE)
  set(_MULTI_VAL_PARAMS VARIABLES)

  cmake_parse_arguments(_WRITE_PY
    "${_OPTIONS}" "${_SINGLE_VAL_PARAMS}" "${_MULTI_VAL_PARAMS}" ${ARGN})

  # Create the file, overwriting if it exists.
  file(WRITE
    "${_WRITE_PY_OUTPUT_FILE}"
    "# The lbann_features module tracks what features lbann is currently\n"
    "# using. To check if the current installation has a feature use the\n"
    "# method lbann.has_feature('FEATURE'). lbann.get_features() returns\n"
    "# a set of the features that are currently set.\n"
    "lbann_cmake_options = {")

  foreach (_variable ${_WRITE_PY_VARIABLES})
    if (${_variable})
      file(APPEND "${_WRITE_PY_OUTPUT_FILE}"
        "\"${_variable}\",")
    endif ()
  endforeach()

  # Close the set
  file(APPEND "${_WRITE_PY_OUTPUT_FILE}"
    "}\n"
    "\n"
    "# Add prefix-stripped options to feature list\n"
    "lbann_features = set()\n"
    "for option in lbann_cmake_options:\n"
    "    lbann_features.add(option[10:])\n"
    "\n"
    "def has_feature(feature_name):\n"
    "  return feature_name.upper() in lbann_features\n"
    "\n"
    "def get_features():\n"
    "    return lbann_features\n"
    )
endfunction ()
