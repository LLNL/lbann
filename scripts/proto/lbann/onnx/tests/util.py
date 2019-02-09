import os
import unittest
from lbann.onnx.util import parseBoolEnvVar, list2LbannList

DUMPED_MODELS_DIR = "dumped_models"

def isModelDumpEnabled():
    return parseBoolEnvVar("LBANN_ONNX_DUMP_MODELS", False)

def createAndGetDumpedModelsDir():
    if not os.path.exists(DUMPED_MODELS_DIR):
        os.mkdir(DUMPED_MODELS_DIR)

    return DUMPED_MODELS_DIR

def getLbannVectorField(fields, name):
    if fields.has_vectors:
        return getattr(fields, name)

    else:
        return list2LbannList([getattr(fields, "{}_i".format(name))])
