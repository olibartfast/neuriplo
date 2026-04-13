# Building neuriplo on Windows

This guide covers building neuriplo natively on Windows using Visual Studio Build Tools 2022 and CMake.

## Detected Environment (this machine)

| Tool | Status | Path |
|------|--------|------|
| VS Build Tools 2022 (MSVC 14.44) | ✓ installed | `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools` |
| MSBuild | ✓ installed | `...\MSBuild\Current\Bin\MSBuild.exe` |
| CMake | ✗ **missing** — install first (see below) |
| Ninja | ✗ not found (optional but faster) |
| Git | ✓ installed | `C:\Program Files\Git\cmd\git.exe` |
| Python | ✓ installed | via WindowsApps |

---

## 1. Install Missing Prerequisites

### CMake (required)

```powershell
winget install Kitware.CMake
# then open a new shell so cmake is on PATH
cmake --version
```

### Ninja (optional, faster builds)

```powershell
winget install Ninja-build.Ninja
```

---

## 2. Open a Developer PowerShell

CMake needs the MSVC compiler on PATH. Use a **Developer PowerShell** (or Developer Command Prompt) so that `cl.exe` and the MSVC environment are available.

### Option A — from Start Menu
Open: **Developer PowerShell for VS 2022**

### Option B — from any PowerShell

```powershell
$vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
& "$vsPath\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64
```

### Option C — import environment manually

```powershell
$vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
cmd /c "`"$vsPath\VC\Auxiliary\Build\vcvars64.bat`" && set" |
  Where-Object { $_ -match "=" } |
  ForEach-Object {
    $k, $v = $_ -split "=", 2
    [System.Environment]::SetEnvironmentVariable($k, $v, "Process")
  }
```

Verify the compiler is on PATH:

```powershell
cl 2>&1 | Select-Object -First 1
# Expected: Microsoft (R) C/C++ Optimizing Compiler Version 19.xx...
```

---

## 3. Install Backend Dependencies

> **Note:** The `scripts/setup_dependencies.sh` and `scripts/setup_*.sh` scripts are bash scripts intended for Linux. On Windows use one of:
> - **Git Bash** (already installed with Git): run `bash scripts/setup_dependencies.sh --backend <BACKEND>`
> - **WSL2**: recommended for full script compatibility
> - **Manual install**: download the pre-built library for the chosen backend and point CMake at it (see Section 4)

Backends that build and run well on Windows CPU-only:

| Backend | CMake flag | Notes |
|---------|-----------|-------|
| `OPENCV_DNN` | `-DDEFAULT_BACKEND=OPENCV_DNN` | Easiest — OpenCV is widely packaged |
| `ONNX_RUNTIME` | `-DDEFAULT_BACKEND=ONNX_RUNTIME` | Pre-built `.zip` from GitHub releases |
| `LIBTORCH` | `-DDEFAULT_BACKEND=LIBTORCH` | Download CPU zip from pytorch.org |
| `OPENVINO` | `-DDEFAULT_BACKEND=OPENVINO` | Use the OpenVINO Windows installer |
| `LIBTENSORFLOW` | `-DDEFAULT_BACKEND=LIBTENSORFLOW` | Pre-built `.zip` from tensorflow.org |

GPU backends (`TENSORRT`, `MIGRAPHX`) require CUDA and additional drivers — not covered here.

### Quick example: ONNX Runtime

```powershell
# Download from https://github.com/microsoft/onnxruntime/releases
# Extract to ~/dependencies/onnxruntime-1.19.2 (matches ONNX_RUNTIME_VERSION in versions.env)
$dest = "$HOME\dependencies\onnxruntime-1.19.2"
```

Or override the path at configure time:
```powershell
cmake -S . -B build -DDEFAULT_BACKEND=ONNX_RUNTIME `
      -DONNX_RUNTIME_DIR="C:\path\to\onnxruntime-1.19.2"
```

---

## 4. Configure and Build

All commands assume a Developer PowerShell (Step 2 complete) and CMake on PATH (Step 1 complete).

### Configure

```powershell
# Example with OpenCV DNN (simplest to start)
cmake -S . -B build `
      -DDEFAULT_BACKEND=OPENCV_DNN `
      -DBUILD_INFERENCE_ENGINE_TESTS=ON

# To use Ninja instead of MSBuild (faster):
cmake -S . -B build -G Ninja `
      -DDEFAULT_BACKEND=OPENCV_DNN `
      -DBUILD_INFERENCE_ENGINE_TESTS=ON
```

### Build

```powershell
cmake --build build --config Release
# Or with MSBuild explicitly:
cmake --build build --config Release --parallel
```

### Test

```powershell
ctest --test-dir build --output-on-failure -C Release
```

---

## 5. Backend-Specific CMake Variables

| Backend | Required CMake variable | Default search path |
|---------|------------------------|---------------------|
| `OPENCV_DNN` | `OpenCV_DIR` | system / vcpkg |
| `ONNX_RUNTIME` | `ONNX_RUNTIME_DIR` | `~/dependencies/onnxruntime-<ver>` |
| `LIBTORCH` | `LIBTORCH_DIR` | `~/dependencies/libtorch-<ver>` |
| `OPENVINO` | `OPENVINO_DIR` | `~/dependencies/openvino-<ver>` |
| `LIBTENSORFLOW` | `TENSORFLOW_DIR` | `~/dependencies/tensorflow-<ver>` |

Version numbers are defined in [versions.env](../versions.env).

---

## 6. OpenCV via vcpkg (recommended for OPENCV_DNN)

vcpkg is the easiest way to get OpenCV on Windows:

```powershell
git clone https://github.com/microsoft/vcpkg "$HOME\vcpkg"
& "$HOME\vcpkg\bootstrap-vcpkg.bat"
& "$HOME\vcpkg\vcpkg" install opencv4:x64-windows

cmake -S . -B build `
      -DDEFAULT_BACKEND=OPENCV_DNN `
      -DCMAKE_TOOLCHAIN_FILE="$HOME\vcpkg\scripts\buildsystems\vcpkg.cmake" `
      -DVCPKG_TARGET_TRIPLET=x64-windows `
      -DBUILD_INFERENCE_ENGINE_TESTS=ON
```

---

## 7. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `cmake` not recognized | Restart shell after `winget install Kitware.CMake`, or add `C:\Program Files\CMake\bin` to PATH manually |
| `cl` not found | Run from Developer PowerShell or apply `vcvars64.bat` (see Step 2) |
| `Cannot find OpenCV` | Set `-DOpenCV_DIR=<path to OpenCVConfig.cmake>` |
| Link errors with MSVC | Ensure all dependencies were built with the same MSVC version (14.44) and `/MD` or `/MT` consistently |
| `LNK2019` on Windows | Some backends expose Linux-only symbols; check backend README for Windows support status |
