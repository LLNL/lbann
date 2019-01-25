from setuptools import setup
import subprocess
import re

def getLBANNVersion():
    try:
        latestTag = subprocess.check_output(["git", "describe", "--abbrev=0"]).decode("utf-8")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    m = re.compile("v(\d+.\d+.\d+)").match(latestTag)
    if m:
        return m.group(1)

    return None

setup(
    name="lbann",
    description="Python wrapper for writing and generating LBANN model prototext files",
    version=getLBANNVersion(),
    url="https://github.com/LLNL/lbann",
    author="Lawrence Livermore National Security, LLC.",
    license="Apache 2.0",
    packages=["lbann"],
    install_requires=["protobuf>=3.0.0",
                      "onnx>=1.3.0",
                      "numpy>=1.16.0",
                      "protobuf>=3.6.1",
                      "nose>=1.3.7"],
    test_suite="nose.collector",
    tests_require=["nose"],
)
