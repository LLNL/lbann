from setuptools import setup
import setuptools.command.build_py
from setuptools.command.install import install
import os.path
import subprocess

from lbann_onnx import getLbannRoot

class installCmd(install):
    def run(self):
        root = getLbannRoot()
        lbannProtoDir = os.path.join(root, "src", "proto")
        lbannProto = os.path.join(lbannProtoDir, "lbann.proto")
        subprocess.check_output(["protoc", "-I={}".format(lbannProtoDir), lbannProto, "--python_out=."])
        install.run(self)

setup(name="lbann-onnx",
      version="0.1.0",
      description="A Python package for the model conversion between LBANN and ONNX",
      packages=["lbann_onnx"],
      install_requires=["onnx>=1.3.0",
                        "numpy>=1.16.0",
                        "protobuf>=3.6.1",
                        "nose>=1.3.7"],
      cmdclass={"install": installCmd},
      test_suite="nose.collector",
      tests_require=["nose"],
      zip_safe=False)
