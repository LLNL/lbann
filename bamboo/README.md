# Plan - Nightly Develop

The Nightly Develop plan runs using the latest commits to LLNL/lbann/develop.

## Stage - Build

Checkout the latest develop and execute `./build_default_executable.sh`

## Stage - Tasks

There are three jobs - Compiler Tests, Integration Tests, and Unit Tests.
Each job has an associated subdirectory in the 'bamboo' folder.
Each job has the same basic structure:

1. Checkout the latest develop.
2. Execute `python -m pytest -s --junitxml=results.xml; exit 0`, which will run all the pytests in the job's associated folder.
3. Run the JUnit Parser. This allows Bamboo to render test cases under the "Tests" tab.

# Directory Structure

'bamboo/compiler_tests', 'bamboo/integration_tests', 'bamboo/unit_tests' each have a 'conftest.py' that pytest requires. They also contain one or more python files. Each file has a number of tests to run. pytest recognizes methods that begin with `test_` as tests. Test methods should use the `assert` keyword.

A test can be as simple as asserting the output of a shell command is 0. The output of a command can be found using Python's `os.system()`.
