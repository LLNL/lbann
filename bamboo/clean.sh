# Clean up generated files from previous builds.
# Then, we know that generated files come from the current build.

LBANN_DIR=$(git rev-parse --show-toplevel)

# Compiler Tests
rm ${LBANN_DIR}/bamboo/compiler_tests/*.pyc
rm -r ${LBANN_DIR}/bamboo/compiler_tests/__pycache__
rm -r ${LBANN_DIR}/bamboo/compiler_tests/builds/*_debug
rm -r ${LBANN_DIR}/bamboo/compiler_tests/builds/*_rel
rm ${LBANN_DIR}/bamboo/compiler_tests/output/*.txt

# Integration Tests
rm ${LBANN_DIR}/bamboo/integration_tests/*.pgm
rm ${LBANN_DIR}/bamboo/integration_tests/*.prototext*
rm ${LBANN_DIR}/bamboo/integration_tests/*.pyc
rm -r ${LBANN_DIR}/bamboo/integration_tests/__pycache__
rm ${LBANN_DIR}/bamboo/integration_tests/*.tfevents.*
rm ${LBANN_DIR}/bamboo/integration_tests/output/*.txt

# Unit Tests
rm ${LBANN_DIR}/bamboo/unit_tests/*.prototext*
rm ${LBANN_DIR}/bamboo/unit_tests/*.pyc
rm -r ${LBANN_DIR}/bamboo/unit_tests/__pycache__
rm ${LBANN_DIR}/bamboo/unit_tests/*.tfevents.*
