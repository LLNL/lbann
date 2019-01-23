import os
from lbann_onnx.util import parseBoolEnvVar

DUMPED_MODELS_DIR = "dumped_models"

def isModelDumpEnabled():
    return parseBoolEnvVar("LBANN_ONNX_DUMP_MODELS", False)

def createAndGetDumpedModelsDir():
    if not os.path.exists(DUMPED_MODELS_DIR):
        os.path.mkdir(DUMPED_MODELS_DIR)

    return DUMPED_MODELS_DIR
