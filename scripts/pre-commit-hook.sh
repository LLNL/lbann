#!/usr/bin/env bash

CLANG_FORMAT="${CLANG_FORMAT:-clang-format}"

# Test for modified files
echo -n "Testing modified files... "
files=$(git diff --cached --name-only --diff-filter=ACM | grep -E "(pp|cu|proto)$")
if [ "$files" == "" ]; then
    echo "No files to test."
    exit 0
fi

if ! [ -x "$CLANG_FORMAT" ]; then
    echo "clang-format not found. Either install it or set the CLANG_FORMAT environment variable to its path"
    exit 1
fi

# Run clang-format
olddiff=$(git diff)
$CLANG_FORMAT -i $files
newdiff=$(git diff)

# Test if diff changed
if [ "$olddiff" != "$newdiff" ]; then
    echo "Files were modified, re-commit."
    exit 1
fi
echo "Success!"
