#!/bin/sh
# Exit with nonzero exit code if anything fails
set -e
# Echo on
set -x

GIT_VER=`git --git-dir .git describe --abbrev=7 --dirty --always --tags`

echo "$GIT_VER"

sed -ie "s#@PROJECT_SOURCE_DIR@#$TRAVIS_BUILD_DIR#g" $DOXYFILE
sed -ie "s#@DOXYGEN_OUTPUT_DIR@#$TRAVIS_BUILD_DIR/$SCRIPT_DIR#g" $DOXYFILE
sed -ie "s#@LBANN_MAJOR_VERSION@.@LBANN_MINOR_VERSION@#$GIT_VER-ci$TRAVIS_JOB_ID#g"  $DOXYFILE
