## Contributing Applications:

The application directory contains the user-facing code for projects
to use LBANN.  Each project directory should contain the python code
to instantiate the model, run both training and inference, an
experiments directory, as well as utility / helper code to pre- or
post-process data.  In addition to project-specific directories the
directory hierarchy groups together similar projects into broader
categories, such as vision-based networks.

### Directory Structure:

```
applications
└─── ATOM
```

The applications directory has primary __projects__ directories as well
as __categories__ that contain related __projects__.

### Project Directory Structure:

The general structure of a project directory should be:

```
<project>
└─── README.md
└─── <app>.py
└─── lib_<app>.py
└─── experiments
      └─── run_<app>.py
└─── utils

```

* README.md
  * Describe the project, how to run it, etc.
* `<app>.py`
  * Python code that builds the model's compute graph
* `lib_<app>.py`
  * Common Python code that builds common substructurs used by the
    application
* experiments
  * Directory to run an experiment.  Should include launcher scripts,
    etc.
  * `run_<app>.py`
    * Launcher script to run the model in train or inference mode
* utils
  * Directory for holding pre- and post-processing scripts
