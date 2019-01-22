import onnx
import numpy as np
import subprocess
import os

ELEM_TYPE = onnx.TensorProto.FLOAT
ELEM_TYPE_NP = np.float32

def getLbannRoot():
    env = os.getenv("LBANN_ROOT")
    if env is not None:
        return env

    return subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).strip().decode("utf-8")
