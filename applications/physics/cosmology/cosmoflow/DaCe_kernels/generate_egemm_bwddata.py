import math
import numpy as np
import os
import shutil

import dace
from dace.transformation.auto import auto_optimize as aopt
from dace.transformation import helpers as xfutil
import dace.libraries.blas

import util


######################################################################
# Problem parameters
WORKSPACE_TARGET_MB, _, args = util.read_conv_from_args()
args.convtype = util.ConvType.bwddata
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
    shrink_factor = int(math.ceil(wspsize_mb / WORKSPACE_TARGET_MB))

    # Shrink B
    BTILE_SIZE = int(math.ceil(B / shrink_factor))
    BTILES = B // BTILE_SIZE
    BTILE_REM = B % BTILE_SIZE
    print("B-tiles:", BTILES, "Size:", BTILE_SIZE, "Remainder tile size:", BTILE_REM)

    MTILE = BTILE_SIZE * DOUT * HOUT * WOUT
    wspsize = MTILE * K + ((MTILE * N) if BTILE_SIZE != 1 else 0)
    wspsize_mb = wspsize * dtype.bytes / 1024 / 1024

    # Shrinking by B alone is not enough
    CITILE_SIZE = CIN
    CITILES = 1
    CITILE_REM = 0
    while CITILE_SIZE > 1 and wspsize_mb > WORKSPACE_TARGET_MB:
        # Shrink by CIN
        CITILE_SIZE -= 1 
        CITILES = CIN // CITILE_SIZE
        CITILE_REM = CIN % CITILE_SIZE
        KTILE = CITILE_SIZE * DW * HW * WW
        wspsize = MTILE * KTILE + ((MTILE * N) if BTILE_SIZE != 1 else 0)
        wspsize_mb = wspsize * dtype.bytes / 1024 / 1024

    print("Cin-tiles:", CITILES, "Size:", CITILE_SIZE, "Remainder tile size:", CITILE_REM)
    
    COTILE_SIZE = COUT
    COTILES = 1
    COTILE_REM = 0
        
    if wspsize_mb > WORKSPACE_TARGET_MB:
        print("WARNING: Cannot reduce workspace further. Current size:", wspsize_mb, 'MiB')
    else:
        print('Required workspace size REDUCED to:', wspsize_mb, 'MiB')
else:
    BTILE_SIZE = B
    BTILE_REM = 0
    BTILES = 1
    CITILE_SIZE = CIN
    CITILES = 1
    CITILE_REM = 0
    COTILE_SIZE = COUT
    COTILES = 1
    COTILE_REM = 0
    print('Required workspace size:', wspsize_mb, 'MiB')


if BTILE_REM != 0:
    BTILES -= 1
if CITILE_REM != 0:
    CITILES -= 1
if COTILE_REM != 0:
    COTILES -= 1

KTILE_SIZE = CITILE_SIZE * DW * HW * WW

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

x_desc = dace.data.Array(dtype, [B, CIN, DIN, HIN, WIN], 
                         strides=args.x.strides, 
                         storage=storage)
y_desc = dace.data.Array(dtype, [B, COUT, DOUT, HOUT, WOUT], 
                         strides=args.y.strides, 
                         storage=storage)

w_desc = dtype[N, K] @ storage
w_desc_atomic = dtype[COUT, CIN, DW, HW, WW] @ storage

def col2im(gdx, dx, BTILE: dace.compiletime, CITILE: dace.compiletime, CIOFFSET, BOFFSET):
    for n, tid in dace.map[0:BTILE, 0:CITILE*DIN*HIN*WIN]:
        c: dace.int64
        copq_res: dace.int64
        ro: dace.int64
        opq_res: dace.int64
        rp: dace.int64
        rq: dace.int64
        
        c, copq_res = divmod(tid, DIN*HIN*WIN)
        ro, opq_res = divmod(copq_res, HIN*WIN)
        rp, rq = divmod(opq_res, WIN)
        
        o: dace.int64 = ro + pads[0]
        p: dace.int64 = rp + pads[1]
        q: dace.int64 = rq + pads[2]

        if o < dilation[0] * (DW - 1) + 1:
            start_d = 0
        else:
            start_d = (o - (dilation[0] * (DW - 1) + 1)) // strides[0] + 1
        end_d = min(DOUT, o // strides[0] + 1)
        if p < dilation[1] * (HW - 1) + 1:
            start_h = 0
        else:
            start_h = (p - (dilation[1] * (HW - 1) + 1)) // strides[1] + 1
        end_h = min(HOUT, p // strides[1] + 1)
        if q < dilation[2] * (WW - 1) + 1:
            start_w = 0
        else:
            start_w = (q - (dilation[2] * (WW - 1) + 1)) // strides[2] + 1
        end_w = min(WOUT, q // strides[2] + 1)

        elem: dtype = 0
        for i in range(start_d, end_d):
            for j in range(start_h, end_h):
                for k in range(start_w, end_w):
                    if ((o - i * strides[0]) % dilation[0] == 0 and
                        (p - j * strides[1]) % dilation[1] == 0 and
                        (q - k * strides[2]) % dilation[2] == 0):
                            ki = (o - i * strides[0]) // dilation[0]
                            kj = (p - j * strides[1]) // dilation[1]
                            kk = (q - k * strides[2]) // dilation[2]

                            elem += gdx[n, i, j, k, c, ki, kj, kk]

        dx[n + BOFFSET, c + CIOFFSET, ro, rp, rq] = elem

def col2im_nobatch(gdx, dx, CITILE: dace.compiletime, CIOFFSET):
    for tid in dace.map[0:CITILE*DIN*HIN*WIN]:
        c: dace.int64
        copq_res: dace.int64
        ro: dace.int64
        opq_res: dace.int64
        rp: dace.int64
        rq: dace.int64
        
        c, copq_res = divmod(tid, DIN*HIN*WIN)
        ro, opq_res = divmod(copq_res, HIN*WIN)
        rp, rq = divmod(opq_res, WIN)
        
        o: dace.int64 = ro + pads[0]
        p: dace.int64 = rp + pads[1]
        q: dace.int64 = rq + pads[2]

        if o < dilation[0] * (DW - 1) + 1:
            start_d = 0
        else:
            start_d = (o - (dilation[0] * (DW - 1) + 1)) // strides[0] + 1
        end_d = min(DOUT, o // strides[0] + 1)
        if p < dilation[1] * (HW - 1) + 1:
            start_h = 0
        else:
            start_h = (p - (dilation[1] * (HW - 1) + 1)) // strides[1] + 1
        end_h = min(HOUT, p // strides[1] + 1)
        if q < dilation[2] * (WW - 1) + 1:
            start_w = 0
        else:
            start_w = (q - (dilation[2] * (WW - 1) + 1)) // strides[2] + 1
        end_w = min(WOUT, q // strides[2] + 1)

        elem: dtype = 0
        for i in range(start_d, end_d):
            for j in range(start_h, end_h):
                for k in range(start_w, end_w):
                    if ((o - i * strides[0]) % dilation[0] == 0 and
                        (p - j * strides[1]) % dilation[1] == 0 and
                        (q - k * strides[2]) % dilation[2] == 0):
                            ki = (o - i * strides[0]) // dilation[0]
                            kj = (p - j * strides[1]) // dilation[1]
                            kk = (q - k * strides[2]) // dilation[2]

                            elem += gdx[c, ki, kj, kk, i, j, k]

        dx[c + CIOFFSET, ro, rp, rq] = elem


def explicit_gemm_tile(w, dx, rdy, alpha, beta, gdx, gdy, BTILE: dace.compiletime, COTILE: dace.compiletime, CITILE: dace.compiletime, KTILE: dace.compiletime, BOFFSET, COOFFSET, CIOFFSET, KOFFSET):
    ggdy = np.reshape(gdy[:BTILE*MB, :COTILE], (BTILE*MB, COTILE))
    rgdy = np.reshape(ggdy, (BTILE, MB, COTILE))
    dy = rdy[BOFFSET, :, :]

    if BTILE > 1:
        # Transpose to NDHWC
        for n, c, j in dace.map[0:BTILE, 0:COTILE, 0:MB]:
            rgdy[n, j, c] = rdy[n + BOFFSET, c + COOFFSET, j]
            
        # GEMM
        dace.libraries.blas.Gemm(ggdy, w, gdx[:BTILE*MB, :KTILE], alpha, beta, trans_a=False, trans_b=False)

        # Col2Im
        rgdx = np.reshape(rrgdx, (BTILE, DOUT, HOUT, WOUT, CIN, DW, HW, WW))
        col2im(rgdx, dx, BTILE, CIN, 0, BOFFSET)
    else:  # Special case for BTILE = 1
        rrgdx = np.reshape(gdx[:MB, :KTILE], (KTILE, MB))

        # GEMM
        dace.libraries.blas.Gemm(w[:, KOFFSET:KOFFSET+KTILE], dy, rrgdx, alpha, beta, trans_a=True, trans_b=False)

        # Col2Im
        rgdx = np.reshape(rrgdx, (CITILE, DW, HW, WW, DOUT, HOUT, WOUT))
        col2im_nobatch(rgdx, dx[BOFFSET], CITILE, CIOFFSET)
    

@dace.program
def explicit_gemm_bwddata(dweights: w_desc, dx: x_desc, dy: y_desc, alpha: dtype, beta: dtype):
    gdx = np.empty((BTILE_SIZE*MB, KTILE_SIZE), dtype)
    gdy = np.empty((BTILE_SIZE*MB, COTILE_SIZE), dtype)
    rdy = np.reshape(dy, (B, COUT, MB))

    for bt in range(BTILES):
        for cit in range(CITILES):
            explicit_gemm_tile(dweights, dx, rdy, alpha, beta, gdx, gdy, BTILE_SIZE, COTILE_SIZE, CITILE_SIZE, KTILE_SIZE, bt*BTILE_SIZE, 0, cit*CITILE_SIZE, cit*KTILE_SIZE)
        if CITILE_REM != 0:
            explicit_gemm_tile(dweights, dx, rdy, alpha, beta, gdx, gdy, BTILE_SIZE, COTILE_SIZE, CITILE_REM, CITILE_REM*DW*HW*WW, bt*BTILE_SIZE, 0, CITILES*CITILE_SIZE, CITILES*KTILE_SIZE)

    if BTILE_REM != 0:
        for cit in range(CITILES):
            explicit_gemm_tile(dweights, dx, rdy, alpha, beta, gdx, gdy, BTILE_REM, COTILE_SIZE,  CITILE_SIZE, KTILE_SIZE, BTILES*BTILE_SIZE, 0, cit*CITILE_SIZE, cit*KTILE_SIZE)
        if CITILE_REM != 0:
            explicit_gemm_tile(dweights, dx, rdy, alpha, beta, gdx, gdy, BTILE_REM, COTILE_SIZE, CITILE_REM, CITILE_REM*DW*HW*WW, BTILES*BTILE_SIZE, 0, CITILES*CITILE_SIZE, CITILES*KTILE_SIZE)

# Direct, atomic version
PADZ, PADY, PADX = pads
SZ, SY, SX = strides
KZ, KY, KX = DW, HW, WW
DZ, DY, DX = dilation

@dace.program#(regenerate_code=False)
def conv3d_bwddata_atomic(dweights: w_desc_atomic, dx: x_desc, dy: y_desc, alpha: dtype, beta: dtype):
    dx *= beta

    for b, cout, cin, k, j, i in dace.map[0:B, 0:COUT, 0:CIN, 0:DOUT, 0:HOUT, 0:WOUT]:
        for kz, ky, kx in dace.map[0:KZ, 0:KY, 0:KX] @ dace.ScheduleType.Sequential:
            with dace.tasklet:
                yin << dy(-1)[b, cout, k, j, i]
                win << dweights(-1)[cout, cin, kz, ky, kx]
                xout >> dx(-1, lambda a, b: a + b)[b, cin, k*SZ + kz*DZ - PADZ, j * SY + ky * DY - PADY, i * SX + kx * DX - PADX]
                if (k * SZ + kz * DZ - PADZ < 0) or (k * SZ + kz * DZ - PADZ >= DIN):
                    continue
                if (j * SY + ky * DY - PADY < 0) or (j * SY + ky * DY - PADY >= HIN):
                    continue
                if (i * SX + kx * DX - PADX < 0) or (i * SX + kx * DX - PADX >= WIN):
                    continue
                xout = yin * win * alpha


if __name__ == '__main__':
    dace.Config.set('compiler', 'allow_view_arguments', value=True)
    dace.Config.set('compiler', 'cuda', 'thread_id_type', value='uint64')
    dace.Config.set('compiler', 'cuda', 'max_concurrent_streams', value=1)

    # Create and optimize the SDFG
    sdfg = explicit_gemm_bwddata.to_sdfg()
    # sdfg = conv3d_bwddata_atomic.to_sdfg()
    sdfg = aopt.auto_optimize(sdfg, dace.DeviceType.GPU)
    
    # Change block sizes
    for m, p in sdfg.all_nodes_recursive():
        # Let DiHydrogen manage streams
        if isinstance(m, dace.SDFGState) and storage == dace.StorageType.GPU_Global:
            m.nosync = True
        
        if isinstance(p, dace.SDFGState) and xfutil.get_parent_map(p, m) is not None:
            continue
        if isinstance(m, dace.nodes.MapEntry) and m.schedule == dace.ScheduleType.GPU_Device:
            m.map.gpu_block_size = (256, 1, 1)
    
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
                    f'lib{hash}.so')

    print(f'Compilation complete. Output file: lib{hash}.so')

    if storage == dace.StorageType.CPU_Heap and os.path.exists('out_dx.bin'):
        print('Files found, verifying result')
        dx, w, dy = util.conv_bwddata_inputs()
        dxin = np.copy(dx)
        csdfg(w, dx, dy, alpha=np.float32(1.0), beta=np.float32(0.0))
        ref_dx = util.verify_bwddata(dx)
        rdx = dx.reshape(*args.x.shape)
        rref = ref_dx.reshape(*args.x.shape)
