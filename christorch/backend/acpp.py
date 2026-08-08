import ctypes
from pathlib import Path

import numpy as np

_lib_path = (
    Path(__file__).resolve().parents[2]
    / "build"
    / "libchristorch_acpp.so"
)

_lib = ctypes.CDLL(str(_lib_path))

_lib.ct_matmul_f32.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t
]

_lib.ct_matmul_f32.restype = ctypes.c_int

def matmul(x, w):
    x = np.ascontiguousarray(x, dtype=np.float32)
    w = np.ascontiguousarray(w, dtype=np.float32)

    if x.ndim != 2 or w.ndim != 2:
        raise ValueError(
            "AdaptiveCpp matmul currently supports 2D arrays only"
        )

    M, K = x.shape
    K2, N = w.shape

    if K != K2:
        raise ValueError(
            f"Incompatible shapes: {x.shape} and {w.shape}"
        )

    out = np.empty(
        (M, N),
        dtype=np.float32
    )

    status = _lib.ct_matmul_f32(
        x.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        ),
        w.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        ),
        out.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        ),
        M,
        K,
        N
    )

    if status != 0:
        raise RuntimeError(
            f"AdapativeCpp matmul failed: {status}"
        )

    return out