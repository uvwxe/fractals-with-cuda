"""CUDA-OpenGL interop glue via ctypes against the cudart bundled with cupy.

Registers a GL pixel buffer object with CUDA so a kernel can write pixels
straight into it — the frame never leaves the GPU (no D2H copy, no WDDM
sync wall). This is the classic CUDA fractal-demo presentation path.
"""
import ctypes
import glob
import os


class CudaGLError(RuntimeError):
    pass


_CUDA_SUCCESS = 0
_CUDA_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 1


def _load_cudart():
    # cupy has already loaded the CUDA runtime into the process; LoadLibrary
    # resolves an already-loaded module by base name. Fall back to a wheel scan.
    for name in ("cudart64_13.dll", "cudart64_12.dll", "cudart64_11.dll"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    import cupy
    root = os.path.dirname(os.path.dirname(cupy.__file__))
    for hit in glob.glob(os.path.join(root, "**", "cudart64_*.dll"), recursive=True):
        try:
            return ctypes.CDLL(hit)
        except OSError:
            continue
    raise CudaGLError("cudart64_*.dll not found — import cupy before this module")


_cudart = _load_cudart()

_cudart.cudaGraphicsGLRegisterBuffer.argtypes = [
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_uint]
_cudart.cudaGraphicsGLRegisterBuffer.restype = ctypes.c_int
_cudart.cudaGraphicsMapResources.argtypes = [
    ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
_cudart.cudaGraphicsMapResources.restype = ctypes.c_int
_cudart.cudaGraphicsUnmapResources.argtypes = [
    ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
_cudart.cudaGraphicsUnmapResources.restype = ctypes.c_int
_cudart.cudaGraphicsResourceGetMappedPointer.argtypes = [
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p]
_cudart.cudaGraphicsResourceGetMappedPointer.restype = ctypes.c_int
_cudart.cudaGraphicsUnregisterResource.argtypes = [ctypes.c_void_p]
_cudart.cudaGraphicsUnregisterResource.restype = ctypes.c_int
_cudart.cudaGetErrorString.argtypes = [ctypes.c_int]
_cudart.cudaGetErrorString.restype = ctypes.c_char_p


def _check(code, what):
    if code != _CUDA_SUCCESS:
        msg = _cudart.cudaGetErrorString(code)
        raise CudaGLError(
            f"{what} failed: cudaError {code}"
            f" ({msg.decode(errors='replace') if msg else 'unknown'})")


class GlCudaBuffer:
    """A GL_PIXEL_UNPACK_BUFFER registered with CUDA.

    map() -> device pointer for kernel writes; unmap() hands the buffer back
    to GL so glTexSubImage2D can pull from it GPU-side.
    """

    def __init__(self, gl_buffer_id: int):
        self._res = ctypes.c_void_p()
        _check(
            _cudart.cudaGraphicsGLRegisterBuffer(
                ctypes.byref(self._res), int(gl_buffer_id),
                _CUDA_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD),
            "cudaGraphicsGLRegisterBuffer")
        self._ptr = ctypes.c_void_p(0)
        self._size = ctypes.c_size_t(0)

    def map(self) -> int:
        _check(_cudart.cudaGraphicsMapResources(
            1, ctypes.byref(self._res), None), "cudaGraphicsMapResources")
        _check(_cudart.cudaGraphicsResourceGetMappedPointer(
            ctypes.byref(self._ptr), ctypes.byref(self._size), self._res),
            "cudaGraphicsResourceGetMappedPointer")
        return self._ptr.value

    @property
    def mapped_size(self) -> int:
        return self._size.value

    def unmap(self):
        _check(_cudart.cudaGraphicsUnmapResources(
            1, ctypes.byref(self._res), None), "cudaGraphicsUnmapResources")

    def close(self):
        if self._res:
            _cudart.cudaGraphicsUnregisterResource(self._res)
            self._res = None


def prefer_discrete_gpu():
    """Ask Windows to run this executable on the high-performance GPU.

    Sets the per-app GpuPreference registry value that the Settings app writes.
    Takes effect on the NEXT process launch (Windows reads it at process
    creation). Returns True if the registry write succeeded.
    """
    try:
        import winreg
        exe = os.path.abspath(sys_executable())
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\DirectX\UserGpuPreferences",
            0, winreg.KEY_SET_VALUE)
        with key:
            winreg.SetValueEx(key, exe, 0, winreg.REG_SZ, "GpuPreference=2;")
        return True
    except OSError:
        return False


def sys_executable() -> str:
    import sys
    return sys.executable
