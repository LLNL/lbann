def parserDescriptor(convertedLayers=[], arithmetic=False, stub=False):
    def _parserDescriptor(cl):
        cl.convertedLayers = convertedLayers
        cl.arithmetic = arithmetic
        cl.stub = stub
        return cl

    return _parserDescriptor
