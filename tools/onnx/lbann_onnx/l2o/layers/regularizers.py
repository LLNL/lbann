from lbann_onnx.l2o.layers import LbannLayerParser

class LbannLayerParser_batch_normalization(LbannLayerParser):
    def parse(self):
        return {"op": "BatchNormalization",
                "paramCount": 4,
                "attrs": {"epsilon":  self.lp.epsilon,
                          "momentum": self.lp.decay,
                          "spatial":  1}}

class LbannLayerParser_local_response_normalization(LbannLayerParser):
    def parse(self):
        return {"op": "LRN",
                "attrs": {"alpha": self.lp.lrn_alpha,
                          "beta":  self.lp.lrn_beta,
                          "bias":  self.lp.lrn_k,
                          "size":  self.lp.window_width}}

class LbannLayerParser_dropout(LbannLayerParser):
    def parse(self):
        return {"op": "Dropout",
                "attrs": {"ratio": 1-self.lp.keep_prob}}
