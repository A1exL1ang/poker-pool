"""Load only our workspace-local optional accelerator, once at import."""
import ctypes
import os
import sys
from pathlib import Path

LIB = None
LOAD_ERROR = None
if os.environ.get("POKERBOT_PURE_PYTHON") != "1":
    _name = "fast_equity.dll" if sys.platform == "win32" else "fast_equity.dylib" if sys.platform == "darwin" else "fast_equity.so"
    _path = Path(__file__).resolve().parent.parent / "assets" / _name
    if _path.is_file():
        try:
            LIB = ctypes.CDLL(str(_path))
            LIB.bot_api_version.restype = ctypes.c_int
            if LIB.bot_api_version()!=3:
                raise OSError("native helper API mismatch; rebuild with tools/build_native.py")
            LIB.bot_init.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
            LIB.bot_init.restype = None
            LIB.bot_evaluate.argtypes = [ctypes.c_uint64]
            LIB.bot_evaluate.restype = ctypes.c_int
            LIB.bot_river.argtypes = [ctypes.c_uint64,ctypes.c_uint64,ctypes.c_int,ctypes.c_double]
            LIB.bot_river.restype = ctypes.c_double
            LIB.bot_icm.argtypes = [ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
            LIB.bot_icm.restype = ctypes.c_double
            LIB.bot_estimate.argtypes = [ctypes.c_uint64,ctypes.c_uint64,ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_double),ctypes.c_int]
            LIB.bot_estimate.restype = ctypes.c_double
        except (OSError,AttributeError) as exc:
            LOAD_ERROR = str(exc)
            LIB = None
