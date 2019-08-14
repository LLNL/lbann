# Clean up generated files from previous builds.
# Then, we know that generated files come from the current build.

LBANN_DIR=$(git rev-parse --show-toplevel)

# Compiler Tests
rm -f ${LBANN_DIR}/bamboo/compiler_tests/*.pyc
rm -rf ${LBANN_DIR}/bamboo/compiler_tests/__pycache__
rm -rf ${LBANN_DIR}/bamboo/compiler_tests/builds/*_debug
rm -rf ${LBANN_DIR}/bamboo/compiler_tests/builds/*_rel
rm -f ${LBANN_DIR}/bamboo/compiler_tests/error/*.txt
rm -f ${LBANN_DIR}/bamboo/compiler_tests/output/*.txt

# Integration Tests
rm -f ${LBANN_DIR}/bamboo/integration_tests/*.pgm
rm -f ${LBANN_DIR}/bamboo/integration_tests/*.prototext*
rm -f ${LBANN_DIR}/bamboo/integration_tests/*.pyc
rm -rf ${LBANN_DIR}/bamboo/integration_tests/__pycache__
rm -f ${LBANN_DIR}/bamboo/integration_tests/*.tfevents.*
rm -f ${LBANN_DIR}/bamboo/integration_tests/error/*.txt
rm -f ${LBANN_DIR}/bamboo/integration_tests/output/*.txt

# Unit Tests
rm -rf ${LBANN_DIR}/bamboo/unit_tests/ckpt_*
rm -f ${LBANN_DIR}/bamboo/unit_tests/*.prototext*
rm -f ${LBANN_DIR}/bamboo/unit_tests/*.pyc
rm -rf ${LBANN_DIR}/bamboo/unit_tests/__pycache__
rm -f ${LBANN_DIR}/bamboo/unit_tests/*.tfevents.*
rm -f ${LBANN_DIR}/bamboo/unit_tests/error/*.txt
rm -f ${LBANN_DIR}/bamboo/unit_tests/output/*.txt
