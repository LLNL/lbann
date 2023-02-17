# Contributing Guidelines for LBANN

We welcome any contributions to LBANN in the form of Pull Requests.
Please follow the guidelines below for more information.

## Attribution

If you have not added yourself to the authors list in 
[CONTRIBUTORS](https://github.com/LLNL/lbann/blob/develop/CONTRIBUTORS), please do so in the appropriate place.

## Setting up the repository

After cloning LBANN, se
Install `pre-commit` via the [instructions](https://pre-commit.com/#install) and set it up on the repository as follows:

```sh
/path/to/lbann $ pre-commit install
```

Make sure you have `clang-format` installed for C/C++ formatting and `yapf` for Python formatting.

## git guidelines

When ready for review and merge, Pull Requests must match the latest `develop` branch commit.
If not ready, **rebase** the commits onto the latest commit. Avoid merge commits.

## Style guidelines

For C/C++ and GPU code, we follow the [LLVM coding style](https://llvm.org/docs/CodingStandards.html) with
adaptations, see the [coding style README](https://github.com/LLNL/lbann/blob/develop/README_coding_style.txt) and the
[clang-format configuration](https://github.com/LLNL/lbann/blob/develop/.clang-format) for more information.

For Python code, we follow the [Google coding style](https://google.github.io/styleguide/pyguide.html) guidelines,
but allow some exceptions to create layers in the LBANN Python frontend.
