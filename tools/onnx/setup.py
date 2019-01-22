from setuptools import setup
import setuptools.command.build_py
import setuptools.command.develop
import os.path
import subprocess

from lbann_onnx import getLbannRoot

class build_py(setuptools.command.build_py.build_py):
    def run(self):
        root = getLbannRoot()
        lbannProtoDir = os.path.join(root, "src", "proto")
        lbannProto = os.path.join(lbannProtoDir, "lbann.proto")
        subprocess.check_output(["protoc", "-I={}".format(lbannProtoDir), lbannProto, "--python_out=."])

class develop(setuptools.command.develop.develop):
    def run(self):
        self.run_command('build_py')

setup(name="lbann-onnx",
      version="0.1.0",
      description="A Python package for the model conversion between LBANN and ONNX",
      packages=["lbann_onnx"],
      install_requires=["onnx>=1.3.0",
                        "numpy>=1.16.0",
                        "protobuf>=3.6.1",
                        "nose>=1.3.7"],
      cmdclass={"build_py": build_py,
                "develop": develop},
      test_suite="nose.collector",
      tests_require=["nose"],
      zip_safe=False)
