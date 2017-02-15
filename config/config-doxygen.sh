#!/bin/sh
# Exit with nonzero exit code if anything fails
set -e

sed -ie 's#@PROJECT_SOURCE_DIR@#$TRAVIS_BUILD_DIR/$GH_REPO_NAME#g' $DOXYFILE                                                                                                                  |
sed -ie 's#@DOXYGEN_OUTPUT_DIR@#$TRAVIS_BUILD_DIR/$GH_REPO_NAME/$SCRIPT_DIR#g' $DOXYFILE                                                                                                      |
sed -ie 's#@LBANN_MAJOR_VERSION@.@LBANN_MINOR_VERSION@#$TRAVIS_JOB_ID#g'  $DOXYFILE
