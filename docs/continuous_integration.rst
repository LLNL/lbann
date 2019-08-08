.. role:: bash(code)
          :language: bash

.. role:: python(code)
          :language: python

LBANN CI
====================

Bamboo is the continuous integration (CI) framework we use.
A Bamboo plan consists of stages (which run sequentially),
which consist of jobs (which run in parallel),
which consist of tasks (which run sequentially).

The LBANN build project has many plans.
Two plans run off of `LLNL/lbann/develop <https://github.com/LLNL/lbann/tree/develop>`_
- Nightly Develop and Weekly Develop.
Nightly Develop runs every night (except Saturday) at midnight.
Weekly Develop runs every Saturday at midnight.
The other plans in the build project are for each individual LBANN developer's
fork of LBANN.

All plans run off the latest *pushed* commits to the repository.
That means if you have local commits that you have not pushed to your fork,
these commits will *not* be tested by Bamboo.
If you have pushed commits to your fork but have not merged your branch into
the main repository's "develop",
your commits will be tested on your individual plan,
but not on Nightly Develop or Weekly Develop.

Plan Configuration
----------------------------------------
Each plan is identical (except Weekly Develop, which will be explained below).
The plans consist of a single stage "Tests".
The stage consists of two jobs - "x86_cpu" (Catalyst), and "x86_gpu" (Pascal).
Each of these jobs can run in parallel.
They consist of an identical list of tasks:

1. Checkout Default Repository (checkout the repository)

2. Run :bash:`./allocate_and_run.sh`;
   Weekly Develop adds the :bash:`--weekly` option.
   This script allocates nodes and then runs "run.sh" which does the following:

   a. Remove Generated Files (each build creates a large number of files.
      We may look at these files between builds,
      so we cannot delete them at the end of a build.
      So, instead we delete them before doing any real work in the next build.
      This also ensures the generated files came from the latest build and not
      a previous build).

   b. Compiler Tests (run tests in "bamboo/compiler_tests")

   c. Integration Tests (run tests in "bamboo/integration_tests")

   d. Unit Tests (run tests in "bamboo/unit_tests")

3. JUnit Parser (this allows Bamboo to render test results in a nice UI)


The tests in Task 2 run
:bash:`$PYTHON -m pytest -s [--weekly] --junitxml=results.xml`,
which will run all the pytests in the job's associated directory.
Note that :bash:`$PYTHON` refers to the Python build to use.
Also note that only Weekly Develop adds the :bash:`--weekly` option.
Many (mostly longer-running) tests are set to not run unless this option is on.
Weekly Develop runs a superset of the tests that Nightly Develop runs.

Directory Structure
----------------------------------------

"bamboo/compiler_tests", "bamboo/integration_tests", "bamboo/unit_tests" each
have a "conftest.py" that pytest requires.
They also contain one or more python files.
Each of these files have a number of tests to run.

Writing Your Own Tests
----------------------------------------

A side effect of our Bamboo setup is that tests must be written using pytest.
Test files must begin with :bash:`test_` to be recognized by pytest.
Individual test methods must also begin with :python:`test_`.
Test methods should use the :python:`assert` keyword.
A test will only fail if the assertion turns out to be false.
Not putting an assertion will automatically cause the test to pass.

How then to test non-Python code?
You can just wrap your test with Python.
A test can be as simple as asserting the output code of a shell command is 0.
The output code of a command can be found using Python's :python:`os.system()`.

Running Tests On Your Individual Plan
----------------------------------------

Unlike Nightly Develop, the individual plans are triggered to run by polling
your fork for commits.
They do not run nightly.
If you push new commits to your fork, a new build should start automatically.
You can also manually start a build by navigating to your individual plan and
clicking Run > Run plan
(this will say "Run branch" if you have plan branches set up).
Once again, keep in mind that the tests will run off what has been pushed to
your GitHub fork of LBANN and not your local copy of the LBANN repository.

Plan branches allow you to test multiple branches simultaneously instead
of simply testing "<fork-name>/develop".
You can create plan branches by navigating to your individual plan,
clicking Actions > Configure plan > Branches > Create plan branch.

Navigating Bamboo
----------------------------------------

From the `LBANN Project Summary <https://lc.llnl.gov/bamboo/browse/LBANN>`_,
click on a plan.
From there, click on a build (builds are listed under "Recent History" and can
also be accessed from the pass/fail marks in the top right,
to the left of the "Run" button).
This will bring you to a certain build's page.
The most relevant tabs are "Tests" and "Logs".
It is recommended to look at failures first in the "Tests" tab,
as the build logs can be difficult to parse through.
The build's "Tests" tab shows "New test failures", "Existing test failures",
"Fixed tests", and "Skipped Tests".

From the build's page, you can also click on individual	jobs,
which have the same tabs.
The "Tests" tabs of the individual jobs have two sub-tabs,
"Failed tests" and "Successful tests".
They do not display skipped tests.
The Bamboo agent that ran the job can be found by looking at the "Agent" field
under the "Job Summary" tab.
Alternatively, you can determine the agent from one of the first lines in the
build logs:
"Build working directory is /usr/workspace/wsb/lbannusr/bamboo/<bamboo-agent-name>/xml-data/build-dir/<build-plan-and-job>".

Some build logs can be very large (e.g. over 100,000 lines).
Beyond about 5,000 lines it is a good idea to download a log instead of
viewing it in the browser.
Beyond about 10,000 lines, some text editors may experience slowness.
At this point it is good to split up the files with
:bash:`split -l 10000 <log-file>`, which creates files of the form `x*` and of
length 10,000.
You can then run a command such as :bash:`grep -in "Errors for:" x*` to find
which files have reported errors.
After you are done, you can remove the files with :bash:`rm x*`.
Note that the original log file is not modified by any of these steps.

As an alternative to splitting the file,
errors can be searched for with
:bash:`grep -in -A <expected-number-of-errors> "Errors for:" <log-file>`.

Bamboo Agent Properties
----------------------------------------

Bamboo agent properties are used to specify requirements for each job.

+--------------------------------+-------------+--------------+----------+------------------+------------------------+
| Agents (jobs)                  | agent_owner | architecture | cluster  | gpu_architecture | sys_type               |
+================================+=============+==============+==========+==================+========================+
| Catalyst Agents (x86_cpu)      | lbannusr    | x86_64       | catalyst | none             | toss_3_x86_64_ib       |
+--------------------------------+-------------+--------------+----------+------------------+------------------------+
| Corona Agents (x86_cpu_corona) | lbannusr    | x86_64       | corona   | none             | toss_3_x86_64_ib       |
+--------------------------------+-------------+--------------+----------+------------------+------------------------+
| Lassen Agents (ppc64le_gpu)    | lbannusr    | ppc64le      | lassen   | volta            | blueos_3_ppc64le_ib_p9 |
+--------------------------------+-------------+--------------+----------+------------------+------------------------+
| Pascal Agents (x86_gpu_pascal) | lbannusr    | x86_64       | pascal   | pascal           | chaos_6_x86_64_ib      |
+--------------------------------+-------------+--------------+----------+------------------+------------------------+
| Ray Agents (ppc64le_gpu)       | lbannusr    | ppc64le      | ray      | pascal           | blueos_3_ppc64le_ib    |
+--------------------------------+-------------+--------------+----------+------------------+------------------------+

Currently, "agent_owner", "architecture", and "gpu_architecture" are used to
determine agents to run a job.

Running Tests From The Command Line
----------------------------------------

Navigate to "bamboo/compiler_tests", "bamboo/integration_tests",
or "bamboo/unit_tests".

To run all the tests in a subdirectory: :bash:`python -m pytest -s --weekly`.
Note that running all tests can take a substantial amount of time.

To run the tests that Nightly Develop or the individual plans run in a
subdirectory: :bash:`python -m pytest -s`.

To run a specific test file: :bash:`python -m pytest -s <test_file>.py`.

To run a specific test:
:bash:`python -m pytest -s <test_file>.py -k '<test_name>'`.

Most integration and unit tests allow for running a test with a different
executable.
The convention is to have a similarly structured test replacing
:python:`_<compiler_name>` with :python:`_exe`.
These tests are set to be skipped in Bamboo, but can be run locally.
There should be a line above the test that gives the command to run the test
locally, likely in the following form:
:bash:`python -m pytest -s <test_file>.py -k '<test_name>' --exe=<executable>`.

At this time, there is no way to run all the :python:`_exe` tests in a subdirectory
and only those.

Helpful Files
----------------------------------------

First, run :bash:`sudo lbannusr`.

To look at output and error from previous builds:
:bash:`cd /usr/workspace/wsb/lbannusr/bamboo/<bamboo-agent-name>/xml-data/build-dir/<build-plan-and-job>/bamboo/<compiler_tests, integration_tests, or unit_tests>/<error or output>`

To look at archived results from previous builds:
:bash:`cd /usr/workspace/wsb/lbannusr/archives/<build-plan>`

To look at Bamboo agent properties:
:bash:`cat /usr/global/tools/bamboo/agents/lbannusr/<bamboo-agent-name>/bin/bamboo-capabilities.properties`

You can copy these files over to your own machine as follows:

- :bash:`sudo lbannusr`

- :bash:`give <lc-username> <absolute-path>`

- :bash:`exit` - to go back to your own LC account, not lbannusr's.

- :bash:`take lbannusr` - now the file exists on your LC account,
  but not yet on your own machine.

From your own machine, not a ssh terminal:

- :bash:`scp <lc-username>@<cluster>.llnl.gov:<absolute-path> .`
