import os

def parseBoolEnvVar(name, defVal):
    if not name in os.environ.keys():
        return defVal

    v = os.environ[name]
    return v == "1"
