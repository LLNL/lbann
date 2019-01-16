def parse_cross_entropy(lp, inputShapes):
    return {"op": "LbannCrossEntropy"}

def parse_binary_cross_entropy(lp, inputShapes):
    return {"op": "LbannBinaryCrossEntropy"}

def parse_sigmoid_binary_cross_entropy(lp, inputShapes):
    return {"op": "LbannSigmoidBinaryCrossEntropy"}

def parse_categorical_accuracy(lp, inputShapes):
    return {"op": "LbannCategoricalAccuracy"}

def parse_top_k_categorical_accuracy(lp, inputShapes):
    return {"op": "LbannTopKCategoricalAccuracy"}

def parse_mean_absolute_error(lp, inputShapes):
    return {"op": "LbannMeanAbsoluteError"}

def parse_mean_squared_error(lp, inputShapes):
    return {"op": "LbannMeanSquaredError"}

def parse_l2_norm2(lp, inputShapes):
    return {"op": "LbannL2Norm2"}
