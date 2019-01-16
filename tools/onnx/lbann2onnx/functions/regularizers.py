def parse_batch_normalization(lp, inputShapes):
    return {"op": "BatchNormalization",
            "paramCount": 4,
            "attrs": {"epsilon":  lp.epsilon,
                      "momentum": lp.decay,
                      "spatial":  1}}

def parse_dropout(lp, inputShapes):
    return {"op": "Dropout",
            "attrs": {"ratio": 1-lp.keep_prob}}

def parse_local_response_normalization(lp, inputShapes):
    return {"op": "LRN",
            "attrs": {"alpha": lp.lrn_alpha,
                      "beta":  lp.lrn_beta,
                      "bias":  lp.lrn_k,
                      "size":  lp.window_width}}
