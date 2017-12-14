# LBANN CI

Bamboo is the continuous integration (CI) framework we use. The Nightly Develop plan runs using the latest commits to `LLNL/lbann/develop` each night at midnight. The plan consists of two stages, detailed below.

## Stage - Build

Checkout the latest develop and execute `./build_default_executable.sh`.

## Stage - Tasks

There are three jobs - Compiler Tests, Integration Tests, and Unit Tests.
Each job has an associated subdirectory in the 'bamboo' folder.
Each job has the same basic structure:

1. Checkout the latest develop.
2. Execute `python -m pytest -s --junitxml=results.xml; exit 0`, which will run all the pytests in the job's associated folder.
3. Run the JUnit Parser. This allows Bamboo to render test cases under the "Tests" tab.

# Directory Structure

'bamboo/compiler_tests', 'bamboo/integration_tests', 'bamboo/unit_tests' each have a 'conftest.py' that pytest requires. They also contain one or more python files. Each file has a number of tests to run. 

# Writing Your Own Tests

A side effect of our Bamboo setup is that tests must be written using pytest. Test files must begin with `test_` to be recognized by pytest. Individual test methods must also begin with `test_`. Test methods should use the `assert` keyword. A test will only fail if the assertion turns out to be false. Not putting an assertion will automatically cause the test to pass.

How then to test non-Python code? You can just wrap your test with Python. A test can be as simple as asserting the output of a shell command is 0. The output of a command can be found using Python's `os.system()`.

# Running Tests On Your Own Plan

Each member of the LBANN team has a duplicate plan of the Nightly Develop plan. Instead of running off the latest commits to `LLNL/lbann/develop` it will run off the latest commits to `<your fork>/lbann/develop`.

You can manually start a build by clicking Run > Run Plan.

Keep in mind that the tests will run off what has been pushed to your GitHub fork of LBANN and not your local copy of the LBANN repository.

# Running Tests From The Command Line

To run tests locally (outside of the Bamboo infrastructure) you will likely need to specify the build directory, e.g:

 `python -m pytest -s --exe ../../build/catalyst.llnl.gov/model_zoo/lbann`

You can also run an individual test by specifying the test filename on the command line, e.g:

 `python -m pytest -s --exe ../../build/catalyst.llnl.gov/model_zoo/lbann test_ridge_regression.py`
