# Clean up generated files from previous builds.
# Then, we know that generated files come from the current build.

LBANN_DIR=$(git rev-parse --show-toplevel)

# Compiler Tests
rm -f ${LBANN_DIR}/ci_test/compiler_tests/*.pyc
rm -rf ${LBANN_DIR}/ci_test/compiler_tests/__pycache__
rm -rf ${LBANN_DIR}/ci_test/compiler_tests/builds/*
rm -f ${LBANN_DIR}/ci_test/compiler_tests/error/*
rm -f ${LBANN_DIR}/ci_test/compiler_tests/output/*

# Integration Tests
rm -f ${LBANN_DIR}/ci_test/integration_tests/*.pgm
rm -f ${LBANN_DIR}/ci_test/integration_tests/*.prototext*
rm -f ${LBANN_DIR}/ci_test/integration_tests/*.pyc
rm -rf ${LBANN_DIR}/ci_test/integration_tests/__pycache__
rm -f ${LBANN_DIR}/ci_test/integration_tests/*.tfevents.*
rm -rf ${LBANN_DIR}/ci_test/integration_tests/experiments/*

# Unit Tests
rm -rf ${LBANN_DIR}/ci_test/unit_tests/ckpt*
rm -rf ${LBANN_DIR}/ci_test/unit_tests/lbann2_*
rm -f ${LBANN_DIR}/ci_test/unit_tests/*.prototext*
rm -f ${LBANN_DIR}/ci_test/unit_tests/*.pyc
rm -rf ${LBANN_DIR}/ci_test/unit_tests/__pycache__
rm -f ${LBANN_DIR}/ci_test/unit_tests/*.tfevents.*
rm -f ${LBANN_DIR}/ci_test/unit_tests/error/*
rm -f ${LBANN_DIR}/ci_test/unit_tests/output/*
rm -rf ${LBANN_DIR}/ci_test/unit_tests/experiments/*
