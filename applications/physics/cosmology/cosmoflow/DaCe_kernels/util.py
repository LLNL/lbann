from __future__ import annotations
import copy
from dataclasses import dataclass
from enum import Enum, auto
import sys
import functools
import numpy as np


class DType(Enum):
    FLOAT = auto()
    HALF = auto()
    DOUBLE = auto()

class ConvMode(Enum):
    CROSS_CORRELATION = auto()
    CONVOLUTION = auto()

class ConvType(Enum):
    fwd = auto()
    bwddata = auto()
    bwdfilt = auto()

@dataclass
class TensorDesc:
    shape: list[int]
    strides: list[int]
    dtype: DType = DType.FLOAT

@dataclass
class ConvDesc:
    mode: ConvMode
    pad: list[int]
    stride: list[int]
    dilation: list[int]
    groups: int

@dataclass
class ConvFwdParams:
    x: TensorDesc
    w: TensorDesc
    y: TensorDesc
    conv: ConvDesc
    convtype: ConvType = ConvType.fwd

    def as_cmdline(self, name: str) -> str:
        result = [f'./conv_{self.convtype.name}', name, str(len(self.x.shape))]
        result.extend(list(map(str, self.x.shape)))
        result.extend(list(map(str, self.x.strides)))
        result.extend(list(map(str, self.w.shape)))
        result.extend(list(map(str, self.w.strides)))
        result.extend(list(map(str, self.y.shape)))
        result.extend(list(map(str, self.y.strides)))
        result.extend(list(map(str, self.conv.pad)))
        result.extend(list(map(str, self.conv.stride)))
        result.extend(list(map(str, self.conv.dilation)))
        return ' '.join(result)
    
    @property
    def dims(self) -> int:
        if self.x.shape[-1] == 0:
            if self.x.shape[-2] == 0:
                return 1
            return 2
        return 3

    def as_filename(self) -> str:
        result = [f'conv{self.dims}d']
        result.extend(list(map(str, self.x.shape)))
        result.extend(list(map(str, self.x.strides)))
        result.extend(list(map(str, self.w.shape)))
        # result.extend(list(map(str, self.w.strides)))
        result.extend(list(map(str, self.y.shape)))
        result.extend(list(map(str, self.y.strides)))
        result.extend(list(map(str, self.conv.pad)))
        result.extend(list(map(str, self.conv.stride)))
        result.extend(list(map(str, self.conv.dilation)))
        result.append(str(self.conv.groups))
        result.append(self.convtype.name)
        return '_'.join(result)
    

def prod(x):
    return functools.reduce(lambda a,b: a*b, x, 1)

def read_conv_from_args():
    argv = copy.copy(sys.argv)
    #print(argv)
    try:
        argv.pop(0)  # Skip script name
        if argv[0] == 'explicit':
            explicit = True
            argv.pop(0)
        else:
            explicit = False
        experiment_name = argv.pop(0)

        argv.pop(0)  # Skip "conv#d"
        input_shape = [int(argv.pop(0)) for _ in range(5)]
        input_strides = [int(argv.pop(0)) for _ in range(5)]
        weight_shape = [int(argv.pop(0)) for _ in range(5)]
        #weight_strides = [int(argv.pop(0)) for _ in range(5)]
        weight_strides = [prod(weight_shape[i:]) for i in range(5)]
        output_shape = [int(argv.pop(0)) for _ in range(5)]
        output_strides = [int(argv.pop(0)) for _ in range(5)]
        pads = [int(argv.pop(0)) for _ in range(3)]
        strides = [int(argv.pop(0)) for _ in range(3)]
        dilations = [int(argv.pop(0)) for _ in range(3)]
        groups = 1  # int(argv.pop(0))
    except:
        print(
            'USAGE: generate_conv.py [explicit] <name> <input 5D shape, strides> '
            '<weight 5D shape, strides> <output 5D shape, strides> <padding> '
            '<conv strides> <conv dilations>')
        raise
    
    return experiment_name, explicit, ConvFwdParams(
        TensorDesc(input_shape, input_strides),
        TensorDesc(weight_shape, weight_strides),
        TensorDesc(output_shape, output_strides),
        ConvDesc(ConvMode.CROSS_CORRELATION, pads, strides, dilations, groups),
    )

def conv_inputs():
    # Buffers can stay flat
    x = np.fromfile('inx.bin', dtype=np.float32)
    w = np.fromfile('inw.bin', dtype=np.float32)
    y = np.fromfile('iny.bin', dtype=np.float32)

    return x, w, y


def conv_bwddata_inputs():
    # Buffers can stay flat
    x = np.fromfile('in_dx.bin', dtype=np.float32)
    w = np.fromfile('inw.bin', dtype=np.float32)
    y = np.fromfile('in_dy.bin', dtype=np.float32)

    return x, w, y


def conv_bwdfilt_inputs():
    # Buffers can stay flat
    x = np.fromfile('in_x.bin', dtype=np.float32)
    w = np.fromfile('in_dw.bin', dtype=np.float32)
    y = np.fromfile('in_dy.bin', dtype=np.float32)

    return x, w, y


def _verify(output, reference):
    print('Output   :', output)
    print('Reference:', reference)

    print('Difference:')
    diff = reference - output
    print('  L2  :', np.linalg.norm(diff) / diff.size)
    print('  Linf:', np.max(diff))


def verify_fwd(y):
    ref_y = np.fromfile('out.bin', dtype=np.float32)
    _verify(y, ref_y)


def verify_bwddata(dx):
    ref_dx = np.fromfile('out_dx.bin', dtype=np.float32)
    _verify(dx, ref_dx)


def verify_bwdfilt(dw):
    ref_dw = np.fromfile('out_dw.bin', dtype=np.float32)
    _verify(dw, ref_dw)
