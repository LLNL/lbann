# Contributing Guidelines for LBANN

We welcome any contributions to LBANN in the form of Pull Requests.
Please follow the guidelines below for more information.

## Attribution

If you have not added yourself to the authors list in 
[CONTRIBUTORS](https://github.com/LLNL/lbann/blob/develop/CONTRIBUTORS), please do so in the appropriate place.

## git guidelines

When ready for review and merge, Pull Requests must match the latest `develop` branch commit.
If not ready, **rebase** the commits onto the latest commit. Avoid merge commits.

## Style guidelines

For C/C++ and GPU code, we follow the [LLVM coding style](https://llvm.org/docs/CodingStandards.html) with
adaptations, see the [coding style README](https://github.com/LLNL/lbann/blob/develop/README_coding_style.txt) and the
[clang-format configuration](https://github.com/LLNL/lbann/blob/develop/.clang-format) for more information.

For Python code, we follow the [Google coding style](https://google.github.io/styleguide/pyguide.html) guidelines,
but allow some exceptions to create layers in the LBANN Python frontend.

## Setting up automatic formatting

To enforce file formatting at every commit, you can use the pre-commit hook provided in the repository.
Make a symbolic link from `.git/hooks/pre-commit` to our script by running the following command
**from the root of your git repository**:

```sh
user@/path/to/lbann$ ln -s ../../scripts/pre-commit-hook.sh .git/hooks/pre-commit
```

Make sure you have `clang-format` installed for C/C++ formatting. If you do not have it installed in the path,
you may override it by setting the `$CLANG_FORMAT` environment variable to its path.
