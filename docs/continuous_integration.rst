.. role:: bash(code)
          :language: bash

.. role:: python(code)
          :language: python

LBANN CI
====================

Continuous integration testing is managed by GitLab CI. The testing
pipeline is configured for machines on the Lawrence Livermore National
Laboratory's "Collaboration Zone" (CZ) clusters at Livermore
Computing. The tests themselves should run anywhere; however, the
GitLab pipelines would require editing to run on other machines.

Developers are required to test their own work. The details are
provided on the `CZ Confluence wiki
<https://lc.llnl.gov/confluence/display/LBANN/LBANN%27s+Gitlab+CI+for+Internal+Developers>_`. A
procedure for external developers (i.e., those without CZ access) has
not been established; please open a PR as usual and we will contact
you about testing it.


Plan Configuration
----------------------------------------

The testing consists of a build phase followed by two test phases,
"unit" tests and integration tests. The default testing runs on two
nodes of the Catalyst, Corona, Lassen, Pascal, and Ray clusters at
Livermore Computing and requires approximately an hour to complete
once the allocation is obtained.

Test results for the unit tests, the "unit" tests, and the integration
tests are parsed by GitLab's JUnit parser, and a list of tests and
their status (pass/fail) are available in the GitLab pipeline view
under the "Tests" tab.

Writing Your Own Tests
----------------------------------------

A side effect of our current testing setup is that the easiest way to
write tests is to use the pytest framework. Test files must begin with
:bash:`test_` to be recognized by pytest. Individual test methods must
also begin with :python:`test_`. Test methods should use the
:python:`assert` keyword or raise an :python:`AssertionError`. A test
will only fail if the assertion turns out to be false.  Not putting an
assertion will automatically cause the test to pass.

How then to test non-Python code?
You can just wrap your test with Python.
A test can be as simple as asserting the output code of a shell command is 0.
The output code of a command can be found using Python's :python:`os.system()`.


Running Tests Yourself
----------------------------------------

This process is documented on the `CZ Confluence wiki
<https://lc.llnl.gov/confluence/display/LBANN/LBANN%27s+Gitlab+CI+for+Internal+Developers>_`.


Running Tests From The Command Line
----------------------------------------

After building LBANN and setting up your `PYTHONPATH` (either
manually, by loading a Spack environment, or by loading LBANN's
modulefile), navigate to `ci_test/integration_tests`, or
`ci_test/unit_tests`.

Running :bash:`python -m pytest [--weekly] <pytest options>` will run
all tests in that directory. Note that running all tests can take a
substantial amount of time and it will take substantially more time if
:bash:`--weekly` is used.

To run all tests in a specific test file, simply pass that file as an
argument to pytest: :bash:`python -m pytest -s <test_file>.py`.

To run specific test cases, use pytest's `-k` option:
:bash:`python -m pytest -s <test_file>.py -k '<test_name>'`.

Consult the pytest help message or online documentation for other
options that pytest supports.


Helpful Files
----------------------------------------

Error logs are available as artifacts in GitLab's CI interface.
