import os
import subprocess
import shutil
from pathlib import Path
import ctypes
import math
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
LIB_PATH = Path(__file__).with_name("libggml_linear.so")


def _compile_shared():
    if LIB_PATH.exists():
        return
    include_dir = REPO_ROOT / "ggml" / "include"
    src_dir = REPO_ROOT / "ggml" / "src"
    cpu_dir = src_dir / "ggml-cpu"
    build_dir = Path(__file__).with_name("ggml_linear_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(exist_ok=True)

    c_sources = [
        Path(__file__).with_suffix(".c"),
        src_dir / "ggml.c",
        src_dir / "ggml-alloc.c",
        src_dir / "ggml-quants.c",
        *cpu_dir.rglob("*.c"),
    ]

    cpp_sources = [
        src_dir / "ggml.cpp",
        src_dir / "ggml-backend.cpp",
        src_dir / "ggml-threading.cpp",
        src_dir / "ggml-opt.cpp",
        src_dir / "gguf.cpp",
        *cpu_dir.rglob("*.cpp"),
    ]

    defs = [
        "-DGGML_USE_CPU",
        '-DGGML_VERSION="0.9.4"',
        '-DGGML_COMMIT="local"',
        "-D_DEFAULT_SOURCE",
        "-D_GNU_SOURCE",
        "-D_POSIX_C_SOURCE=200809L",
    ]

    include_args = [
        f"-I{include_dir}",
        f"-I{include_dir / 'ggml'}",
        f"-I{src_dir}",
        f"-I{cpu_dir}",
    ]

    objects: list[Path] = []
    for src in c_sources:
        if "kleidiai" in src.parts or "arch" in src.parts or "spacemit" in src.parts:
            continue
        rel = src.relative_to(REPO_ROOT).with_suffix("")
        obj = build_dir / f"{rel.as_posix().replace('/', '__')}.o"
        cmd = [
            "gcc",
            "-O3",
            "-std=gnu11",
            "-fPIC",
            *defs,
            *include_args,
            "-c",
            str(src),
            "-o",
            str(obj),
        ]
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        objects.append(obj)

    for src in cpp_sources:
        if "kleidiai" in src.parts or "arch" in src.parts or "spacemit" in src.parts:
            continue
        rel = src.relative_to(REPO_ROOT).with_suffix("")
        obj = build_dir / f"{rel.as_posix().replace('/', '__')}.o"
        cmd = [
            "g++",
            "-O3",
            "-std=gnu++17",
            "-fPIC",
            *defs,
            *include_args,
            "-c",
            str(src),
            "-o",
            str(obj),
        ]
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        objects.append(obj)

    link_cmd = [
        "g++",
        "-shared",
        "-fPIC",
        *defs,
        *include_args,
        *map(str, objects),
        "-o",
        str(LIB_PATH),
        "-lpthread",
        "-lm",
    ]
    subprocess.run(link_cmd, check=True, cwd=REPO_ROOT)


def _load_lib():
    _compile_shared()
    lib = ctypes.CDLL(str(LIB_PATH))
    lib.ggml_linear_forward.restype = ctypes.c_int
    lib.ggml_linear_forward.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    return lib


class GGMLLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, threads: int | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threads = threads or os.cpu_count() or 1
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.use_bias = bias
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in = in_features
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        self._lib = _load_lib()

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear, threads: int | None = None):
        new = cls(linear.in_features, linear.out_features, bias=linear.bias is not None, threads=threads)
        with torch.no_grad():
            new.weight.copy_(linear.weight)
            if linear.bias is not None:
                new.bias.copy_(linear.bias)
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cpu":
            return torch.nn.functional.linear(x, self.weight, self.bias)
        x_contig = x.contiguous()
        batch = x_contig.shape[0]
        out = torch.empty((batch, self.out_features), device="cpu", dtype=torch.float32)
        weight = self.weight.contiguous()
        bias = self.bias if self.use_bias else torch.zeros(self.out_features, dtype=torch.float32)
        ret = self._lib.ggml_linear_forward(
            ctypes.c_void_p(x_contig.data_ptr()),
            ctypes.c_void_p(weight.data_ptr()),
            ctypes.c_void_p(bias.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(batch),
            ctypes.c_int(self.in_features),
            ctypes.c_int(self.out_features),
            ctypes.c_int(self.threads),
        )
        if ret != 0:
            raise RuntimeError("ggml_linear_forward failed")
        return out


__all__ = ["GGMLLinear"]
