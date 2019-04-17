def parseSpatialAttribute(params, attr, dims, defVal):
    if params.has_vectors:
        s = getattr(params, attr)
        if s == "":
            return [defVal]*dims
        return list(map(int, s.split(" ")))
    else:
        return [getattr(params, "{}_i".format(attr))]*dims

def parseSpatialAttributes(lp, name, hasDilations):
    spatialParamNameMap = {"kernel_shape": ("dims", None),
                           "pads":         ("pads", 0),
                           "strides":      ("strides", 1)}
    if hasDilations:
        spatialParamNameMap["dilations"] = ("dilations", 1)

    attrs = dict(map(lambda x: (x,
                                parseSpatialAttribute(lp,
                                                      "{}_{}".format(name, spatialParamNameMap[x][0]),
                                                      lp.num_dims,
                                                      spatialParamNameMap[x][1])),
                     spatialParamNameMap.keys()))

    assert len(attrs["pads"]) == lp.num_dims

    attrs["pads"] *= 2
    if hasDilations:
        attrs["dilations"] = list(map(lambda x: 1 if x == 0 else x, attrs["dilations"]))

    return attrs
