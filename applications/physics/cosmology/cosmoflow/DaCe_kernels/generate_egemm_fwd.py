import math
import numpy as np
import os
import shutil

import dace
from dace.codegen.common import get_gpu_backend
from dace.transformation.auto import auto_optimize as aopt
from dace.transformation import helpers as xfutil
import dace.libraries.blas

import util


######################################################################
# Problem parameters
WORKSPACE_TARGET_MB, _, args = util.read_conv_from_args()
WORKSPACE_TARGET_MB = int(WORKSPACE_TARGET_MB)

pads = args.conv.pad
strides = args.conv.stride
dilation = args.conv.dilation

B, CIN, DIN, HIN, WIN = args.x.shape
B, COUT, DOUT, HOUT, WOUT = args.y.shape

storage = dace.StorageType.GPU_Global
# storage = dace.StorageType.CPU_Heap  # For testing validity
dtype = dace.float32

COUT, CIN, DW, HW, WW = args.w.shape

if B < 0:
    B = dace.symbol('B')

M = B * DOUT * HOUT * WOUT
N = COUT

K = CIN * DW * HW * WW

MB = DOUT * HOUT * WOUT
KB = K//CIN

print('M =', M, 'N =', N, 'K =', K)

# Reduce workspace to target size
wspsize = M * K + M * N
wspsize_mb = wspsize * dtype.bytes / 1024 / 1024
if wspsize_mb > WORKSPACE_TARGET_MB:
    MB_WSTILE = WORKSPACE_TARGET_MB * 1024 * 1024 // dtype.bytes // (K + N) // B
    M_WSTILES = int(math.ceil(MB / MB_WSTILE))
    wspsize_mb = B * (MB_WSTILE * K + MB_WSTILE * N) * dtype.bytes / 1024 / 1024
    print('Required workspace size REDUCED to:', wspsize_mb, 'MiB (tiles:', M_WSTILES, ', tile size:', MB_WSTILE, ')')
    MB_WSTILE_REM = MB % MB_WSTILE
    if MB_WSTILE_REM != 0:
        M_WSTILES -= 1
else:
    M_WSTILES = 1
    MB_WSTILE = MB
    MB_WSTILE_REM = 0
    print('Required workspace size:', wspsize_mb, 'MiB')

######################################################################
# Fast division / modulo implementation

def clz(x):
  for i in range(31, -1, -1):
    if ((1 << i) & x):
        return 31 - i
  return 32

def find_log2(x):
  a = 31 - clz(x)
  if (x & (x - 1)) != 0:
    a += 1 # Round up, add 1 if not a power of 2.
  return a


def fast_divmod(num, denominator: dace.compiletime):
    if denominator == 1:
        return num, 0

    # Precompute fast coefficients for index computations
    # https://gmplib.org/~tege/divcnst-pldi94.pdf
    # Adapted from https://github.com/NVIDIA/cutlass/blob/master/include/cutlass/fast_math.h#L324
    p = dace.inline(31 + find_log2(denominator))
    m: dace.int64 = dace.inline(((1 << (31 + find_log2(denominator))) + denominator - 1) // denominator)

    shift_right = p - 32

    quotient: dace.int64 = 0
    with dace.tasklet(dace.Language.CPP):
        _n << num
        _m << m
        sr << shift_right
        """
        q = __umulhi(_n, _m) >> sr;
        // Equivalent to: q = int(((int64_t)_n * _m) >> 32) >> sr;
        """
        q >> quotient
    # quotient = ((dace.int64(num) * multiplier) >> 32) >> shift_right
    remainder: dace.int64 = num - (quotient * denominator)

    return quotient, remainder

def slow_divmod(num, denominator):
    return num // denominator, num % denominator

divmod = fast_divmod

######################################################################
# Explicit GEMM implementation

x_desc = dace.data.Array(dtype, [B, CIN, DIN, HIN, WIN], strides=args.x.strides, storage=storage)
y_desc = dace.data.Array(dtype, [B, COUT, DOUT, HOUT, WOUT], strides=args.y.strides, storage=storage)
w_desc = dtype[N, K] @ storage

# Optimization if no workspace tiles are used
maybe_unroll = dace.unroll if M_WSTILES == 1 else dace.nounroll

@dace.program
def explicit_gemm(x: x_desc, weights: w_desc, y: y_desc, alpha: dtype, beta: dtype):
    gx = np.empty((B*MB_WSTILE, K), dtype)
    gy = np.empty((B*MB_WSTILE, N), dtype)

    for T in maybe_unroll(range(M_WSTILES)):
        for n, i, k in dace.map[0:B, 0:MB_WSTILE, 0:K]:
            o: dace.int64
            opq_res: dace.int64
            p: dace.int64
            q: dace.int64
            c: dace.int64
            ctrs_res: dace.int64
            t: dace.int64
            trs_res: dace.int64
            r: dace.int64
            s: dace.int64

            o, opq_res = divmod(T * MB_WSTILE + i, HOUT*WOUT)
            p, q = divmod(opq_res, WOUT)

            c, ctrs_res = divmod(k, DW*HW*WW)
            t, trs_res = divmod(ctrs_res, HW*WW)
            r, s = divmod(trs_res, WW)

            d = strides[0]*o + dilation[0]*t - pads[0]
            h = strides[1]*p + dilation[1]*r - pads[1]
            w = strides[2]*q + dilation[2]*s - pads[2]

            elem: dtype
            if h < 0 or h >= HIN or w < 0 or w >= WIN or d < 0 or d >= DIN:
                elem = 0
            else:
                elem = x[n, c, d, h, w]

            gx[i + n*MB_WSTILE, k] = elem

        dace.libraries.blas.Gemm(gx, weights, gy, alpha, beta, trans_a=False, trans_b=True)

        rgy = np.reshape(gy, (B, MB_WSTILE, COUT))
        ry = np.reshape(y, (B, COUT, MB))
        for n, c, j in dace.map[0:B, 0:COUT, 0:MB_WSTILE]:
            ry[n, c, j + T * MB_WSTILE] = rgy[n, j, c] #+ bias[c]

    if MB_WSTILE_REM != 0:
        for n, i, k in dace.map[0:B, 0:MB_WSTILE_REM, 0:K]:
            o: dace.int64
            opq_res: dace.int64
            p: dace.int64
            q: dace.int64
            c: dace.int64
            ctrs_res: dace.int64
            t: dace.int64
            trs_res: dace.int64
            r: dace.int64
            s: dace.int64

            o, opq_res = divmod(M_WSTILES * MB_WSTILE + i, HOUT*WOUT)
            p, q = divmod(opq_res, WOUT)

            c, ctrs_res = divmod(k, DW*HW*WW)
            t, trs_res = divmod(ctrs_res, HW*WW)
            r, s = divmod(trs_res, WW)

            d = strides[0]*o + dilation[0]*t - pads[0]
            h = strides[1]*p + dilation[1]*r - pads[1]
            w = strides[2]*q + dilation[2]*s - pads[2]

            elem: dtype
            if h < 0 or h >= HIN or w < 0 or w >= WIN or d < 0 or d >= DIN:
                elem = 0
            else:
                elem = x[n, c, d, h, w]

            gx[i + n*MB_WSTILE_REM, k] = elem

        dace.libraries.blas.Gemm(gx[:MB_WSTILE_REM*B, :], weights, gy[:MB_WSTILE_REM*B, :], alpha, beta, trans_a=False, trans_b=True)

        rgy_rem = np.reshape(gy[:B*MB_WSTILE_REM, :], (B, MB_WSTILE_REM, COUT))
        ry_rem = np.reshape(y, (B, COUT, MB))
        for n, c, j in dace.map[0:B, 0:COUT, 0:MB_WSTILE_REM]:
            ry_rem[n, c, j + M_WSTILES*MB_WSTILE] = rgy_rem[n, j, c] #+ bias[c]


if __name__ == '__main__':


    dace.Config.set('compiler', 'allow_view_arguments', value=True)
    dace.Config.set('compiler', 'cuda', 'thread_id_type', value='uint64')
    dace.Config.set('compiler', 'cuda', 'max_concurrent_streams', value=1)

    # Create and optimize the SDFG
    sdfg = explicit_gemm.to_sdfg()
    sdfg = aopt.auto_optimize(sdfg, dace.DeviceType.GPU)

    # Change block sizes
    for m, p in sdfg.all_nodes_recursive():
        # Let DiHydrogen manage streams
        if isinstance(m, dace.SDFGState) and storage == dace.StorageType.GPU_Global:
            m.nosync = True

        if isinstance(p, dace.SDFGState) and xfutil.get_parent_map(p, m) is not None:
            continue
        if isinstance(m, dace.nodes.MapEntry) and m.schedule == dace.ScheduleType.GPU_Device:
            m.map.gpu_block_size = (16, 16, 1)

    # Make allocation workspace-based
    if storage == dace.StorageType.GPU_Global:
        for _, _, arr in sdfg.arrays_recursive():
            if arr.lifetime == dace.AllocationLifetime.Persistent:
                arr.lifetime = dace.AllocationLifetime.External

    # Rename output filename
    hash = args.as_filename()
    sdfg.name = 'conv'
    csdfg = sdfg.compile()
    shutil.copyfile(os.path.abspath(os.path.join(sdfg.build_folder, 'build', f'lib{sdfg.name}.so')),
                    f'.dacecache/lib{hash}.so')

    print(f'Compilation complete. Output file: .dacecache/lib{hash}.so')

    if storage == dace.StorageType.CPU_Heap and os.path.exists('out.bin'):
        print('Files found, verifying result')
        x, w, y = util.conv_inputs()
        csdfg(x, w, y, alpha=np.float32(1.0), beta=np.float32(0.0))
        util.verify_fwd(y)
