.. role:: bash(code)
          :language: bash

LBANN Documentation Building
============================

.. warning:: Some of the directions in this section are Mac-specific.

Adding Documentation Outside Code
----------------------------------

1. Create a file such as "new_docs.rst" in "lbann/docs".

2. Add "new_docs" (no ".rst") to the appropriate documentation block in
   "lbann/docs/index.rst".

3. Look at the other ".rst" files in "lbann/docs" to see how to get
   certain formatting.

4. When you want to see how your code looks, you have a couple options:

   a. Push your docs to your fork/branch on GitHub and look at how
      the text renders. This is a very simplified look compared to
      Read-the-Docs.

   b. From "lbann/docs" run :bash:`make html` and then
      :bash:`open -a <preferred web browser> _build/html/index.html`.
      This is exactly how the docs will look.

5. Merge your code into "lbann/develop" and then have someone with
   correct permissions on Read-the-Docs update the
   `official docs <http://software.llnl.gov/lbann/>`_.

Making The Build Work
----------------------------------

In order to make :bash:`make html` work, you may need to do a few steps:

1. Run :bash:`pip3 install sphinx breathe sphinx-rtd-theme`.

2. Download Doxygen by going to the
   `Doxygen downloads page <http://www.doxygen.nl/download.html#srcbin>`_,
   downloading "Doxygen-1.8.15.dmg", and
   dragging the app to the "Applications" folder.

3. Determine the directory Doxygen is in by running `which Doxygen`.
   If nothing is returned, see if `doxygen` is in
   "/Applications/Doxygen.app/Contents/Resources" or
   "/Applications/Doxygen.app/Contents/MacOS".

4. Add Doxygen to your path with
   :bash:`PATH="<doxygen directory>:${PATH}"`.
   You may want to add this to your "~/.bash_profile" so your :bash:`PATH` is
   always correct. Run :bash:`source ~.bash_profile` to run that code.

5. Try running :bash:`make html` again.
