# OPTIMIZE: use has_vectors
def parseSpatialAttribute(params, attr, dims):
    ary = getattr(params, attr)
    if ary != "":
        return list(map(int, ary.split(" ")))
    else:
        return [getattr(params, "{}_i".format(attr))]*dims

def parseSpatialAttributes(lp, name, hasDilations):
    spatialParamNameMap = {"kernel_shape": "dims",
                           "pads":         "pads",
                           "strides":      "strides"}
    if hasDilations:
        spatialParamNameMap["dilations"] = "dilations"

    attrs = dict(map(lambda x: (x,
                                parseSpatialAttribute(lp,
                                                      "{}_{}".format(name, spatialParamNameMap[x]),
                                                      lp.num_dims)),
                     spatialParamNameMap.keys()))

    assert len(attrs["pads"]) == lp.num_dims

    attrs["pads"] *= 2
    if hasDilations:
        attrs["dilations"] = list(map(lambda x: 1 if x == 0 else x, attrs["dilations"]))

    return attrs
