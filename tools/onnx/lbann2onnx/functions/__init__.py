import re
import onnx

import lbann2onnx
import lbann2onnx.util
from lbann2onnx.functions.learnings    import *
from lbann2onnx.functions.math         import *
from lbann2onnx.functions.regularizers import *
from lbann2onnx.functions.transforms   import *
from lbann2onnx.functions.losses       import *
import lbann2onnx.functions as functions

FUNCTIONS = dict(map(lambda x: (x[0].group(1), getattr(functions, x[1])),
                     filter(lambda x: x[0] is not None,
                            map(lambda x: (re.compile("^parse_(.*)$").match(x), x),
                                dir()))))

def parseLbannLayer(l, tensorShapes, knownNodes):
    if any(map(lambda x: l.HasField(x), ["input",
                                         "identity", # LBANN's "identity" does not have outputs
                                         "dummy"])):
        return {}

    lbannInputs = list(map(lambda x: "{}_0".format(x),
                           l.parents.split(" ") if l.parents != "" else []))
    lbannOutputs = l.children.split(" ") if len(l.children) > 0 else []

    for f in FUNCTIONS.keys():
        if l.HasField("split"):
            if l.name not in tensorShapes.keys():
                lbann2onnx.util.printError("The shape of \"{}\" cannot be inferred.".format(l.name) \
                                           + " This error may happen when you set incorret an input tensor name.")
                lbann2onnx.util.printParsingState(l, tensorShapes)
                exit()

            ipt = onnx.helper.make_tensor_value_info(name="{}_0".format(l.name),
                                                     elem_type=lbann2onnx.ELEM_TYPE,
                                                     shape=tensorShapes[l.name])

            return {"inputs": [ipt]}

        if l.HasField(f):
            for i in lbannInputs:
                if not i in tensorShapes.keys():
                    lbann2onnx.util.printError("The shape of \"{}\" cannot be inferred.".format(i))
                    lbann2onnx.util.printParsingState(l, tensorShapes)
                    exit()

            arg = getattr(l, f)
            if f == "unpooling":
                arg = list(filter(lambda x: x.name == l.unpooling.pooling_layer, knownNodes))[0]

            ret = FUNCTIONS[f](arg,
                               list(map(lambda x: tensorShapes[x], lbannInputs)))
            if ret is None:
                return {}

            defVals = {"paramCount": 0,
                       "outputCount": 1,
                       "params": [],
                       "inits": [],
                       "attrs": {}}
            for k in defVals.keys():
                if not k in ret.keys():
                    ret[k] = defVals[k]

            paramNames = list(map(lambda x: "{}_p{}".format(l.name, x), range(ret["paramCount"])))
            inputNames  = lbannInputs + paramNames
            outputNames = list(map(lambda x: "{}_{}".format(l.name, x), range(ret["outputCount"]))) if len(lbannOutputs) == 0 else list(map(lambda x: "{}_0".format(x), lbannOutputs))

            node = onnx.helper.make_node(ret["op"],
                                         inputs=inputNames,
                                         outputs=outputNames,
                                         name=l.name,
                                         lbannOp=f,
                                         lbannDataLayout=l.data_layout,
                                         **ret["attrs"])

            inputs = list(map(lambda x: onnx.helper.make_tensor_value_info(name=paramNames[x],
                                                                           elem_type=lbann2onnx.ELEM_TYPE,
                                                                           shape=ret["params"][x]),
                              range(len(ret["params"]))))

            inits = list(map(lambda x: onnx.helper.make_tensor(name=paramNames[x],
                                                               data_type=ret["inits"][x]["dataType"],
                                                               dims=ret["inits"][x]["shape"],
                                                               vals=ret["inits"][x]["value"],
                                                               raw=True),
                             range(len(ret["inits"]))))

            return {"node": node, "inputs": inputs, "inits": inits}

    lbann2onnx.util.printError("Unimplemented LBANN operator: {}".format(l))

    assert False
