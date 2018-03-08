# Clean up generated files from previous builds.
# Then, we know that generated files come from the current build.

LBANN_DIR=$(git rev-parse --show-toplevel)
rm ${LBANN_DIR}/bamboo/compiler_tests/output/*_output.txt
rm ${LBANN_DIR}/bamboo/compiler_tests/builds/*_rel
rm ${LBANN_DIR}/bamboo/compiler_tests/builds/*_debug
rm ${LBANN_DIR}/bamboo/integration_tests/output/*_output.txt
