from pathlib import Path
import os
import onnxruntime as ort


def setup_onnx_gpu_runtime(debug: bool = False) -> None:
    """
    Thiết lập DLL search path cho ONNX Runtime GPU trên Windows.

    Mục tiêu:
    - Đưa các thư mục DLL NVIDIA cài qua pip vào vùng tìm kiếm của Windows.
    - Preload CUDA/cuDNN để ONNX Runtime dùng được CUDAExecutionProvider.
    """

    project_root = Path(__file__).resolve().parents[2]

    nvidia_dir = (
        project_root
        / ".venv"
        / "Lib"
        / "site-packages"
        / "nvidia"
    )

    nvidia_dll_dirs = [
        nvidia_dir / "cuda_runtime" / "bin",
        nvidia_dir / "cublas" / "bin",
        nvidia_dir / "cufft" / "bin",
        nvidia_dir / "curand" / "bin",
        nvidia_dir / "cuda_nvrtc" / "bin",
        nvidia_dir / "nvjitlink" / "bin",
        nvidia_dir / "cudnn" / "bin",
    ]

    for dll_dir in nvidia_dll_dirs:
        if dll_dir.exists():
            os.add_dll_directory(str(dll_dir))
            os.environ["PATH"] = (
                str(dll_dir)
                + os.pathsep
                + os.environ.get("PATH", "")
            )

    # Nạp CUDA/cuDNN DLL từ NVIDIA site-packages
    ort.preload_dlls(directory="")

    if debug:
        ort.print_debug_info()