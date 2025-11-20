# PowerShell script for setting up neuriplo dependencies on Windows
# This script downloads and installs inference backend dependencies

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("ONNX_RUNTIME", "TENSORRT", "LIBTORCH", "OPENVINO", "LIBTENSORFLOW", "GGML", "OPENCV_DNN")]
    [string]$Backend = "OPENCV_DNN",
    
    [Parameter(Mandatory=$false)]
    [string]$DependencyRoot = "$env:USERPROFILE\dependencies",
    
    [Parameter(Mandatory=$false)]
    [switch]$Force,
    
    [Parameter(Mandatory=$false)]
    [switch]$Verbose
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Color output functions
function Write-StatusMessage {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-SuccessMessage {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-WarningMessage {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-ErrorMessage {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Load versions from versions.env
function Get-VersionsFromEnv {
    $versionsFile = Join-Path $PSScriptRoot "..\versions.env"
    
    if (-not (Test-Path $versionsFile)) {
        Write-ErrorMessage "versions.env file not found at $versionsFile"
        exit 1
    }
    
    $versions = @{}
    Get-Content $versionsFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#")) {
            if ($line -match '^([A-Z_]+)=(.+)$') {
                $key = $Matches[1]
                $value = $Matches[2] -replace '^"(.+)"$', '$1'
                $versions[$key] = $value
            }
        }
    }
    
    return $versions
}

# Check if running as Administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Download file with progress
function Download-File {
    param(
        [string]$Url,
        [string]$OutputPath
    )
    
    Write-StatusMessage "Downloading from $Url..."
    
    try {
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $Url -OutFile $OutputPath -UseBasicParsing
        $ProgressPreference = 'Continue'
        Write-SuccessMessage "Download completed: $OutputPath"
    }
    catch {
        Write-ErrorMessage "Failed to download file: $_"
        throw
    }
}

# Extract archive
function Extract-Archive {
    param(
        [string]$ArchivePath,
        [string]$DestinationPath
    )
    
    Write-StatusMessage "Extracting $ArchivePath to $DestinationPath..."
    
    try {
        Expand-Archive -Path $ArchivePath -DestinationPath $DestinationPath -Force
        Write-SuccessMessage "Extraction completed"
    }
    catch {
        Write-ErrorMessage "Failed to extract archive: $_"
        throw
    }
}

# Setup ONNX Runtime
function Setup-ONNXRuntime {
    param([hashtable]$Versions)
    
    $version = $Versions["ONNX_RUNTIME_VERSION"]
    $installDir = Join-Path $DependencyRoot "onnxruntime-win-x64-gpu-$version"
    
    if ((Test-Path $installDir) -and -not $Force) {
        Write-WarningMessage "ONNX Runtime already installed at $installDir. Use -Force to reinstall."
        return
    }
    
    Write-StatusMessage "Setting up ONNX Runtime $version..."
    
    # Download ONNX Runtime
    $downloadUrl = "https://github.com/microsoft/onnxruntime/releases/download/v$version/onnxruntime-win-x64-gpu-$version.zip"
    $zipFile = Join-Path $env:TEMP "onnxruntime.zip"
    
    Download-File -Url $downloadUrl -OutputPath $zipFile
    
    # Extract
    New-Item -ItemType Directory -Force -Path $DependencyRoot | Out-Null
    Extract-Archive -ArchivePath $zipFile -DestinationPath $DependencyRoot
    
    # Cleanup
    Remove-Item $zipFile -Force
    
    Write-SuccessMessage "ONNX Runtime $version installed at $installDir"
}

# Setup LibTorch
function Setup-LibTorch {
    param([hashtable]$Versions)
    
    $version = $Versions["PYTORCH_VERSION"]
    $installDir = Join-Path $DependencyRoot "libtorch"
    
    if ((Test-Path $installDir) -and -not $Force) {
        Write-WarningMessage "LibTorch already installed at $installDir. Use -Force to reinstall."
        return
    }
    
    Write-StatusMessage "Setting up LibTorch $version..."
    
    # Download LibTorch (CPU version for simplicity, adjust URL for CUDA version)
    $cudaVersion = $Versions["CUDA_VERSION"] -replace '\.', ''
    $downloadUrl = "https://download.pytorch.org/libtorch/cu$cudaVersion/libtorch-win-shared-with-deps-$version%2Bcu$cudaVersion.zip"
    $zipFile = Join-Path $env:TEMP "libtorch.zip"
    
    Write-StatusMessage "Downloading LibTorch with CUDA $($Versions['CUDA_VERSION']) support..."
    Download-File -Url $downloadUrl -OutputPath $zipFile
    
    # Extract
    New-Item -ItemType Directory -Force -Path $DependencyRoot | Out-Null
    Extract-Archive -ArchivePath $zipFile -DestinationPath $DependencyRoot
    
    # Cleanup
    Remove-Item $zipFile -Force
    
    Write-SuccessMessage "LibTorch $version installed at $installDir"
}

# Setup OpenVINO
function Setup-OpenVINO {
    param([hashtable]$Versions)
    
    $version = $Versions["OPENVINO_VERSION"]
    $installDir = Join-Path $DependencyRoot "openvino_$version"
    
    if ((Test-Path $installDir) -and -not $Force) {
        Write-WarningMessage "OpenVINO already installed at $installDir. Use -Force to reinstall."
        return
    }
    
    Write-StatusMessage "Setting up OpenVINO $version..."
    Write-WarningMessage "Please download OpenVINO from Intel's website and extract to: $installDir"
    Write-StatusMessage "Download URL: https://storage.openvinotoolkit.org/repositories/openvino/packages/$version/windows/"
    
    # Note: OpenVINO requires manual download from Intel
    Write-StatusMessage "After downloading, extract the archive to $installDir"
}

# Setup TensorRT
function Setup-TensorRT {
    param([hashtable]$Versions)
    
    $version = $Versions["TENSORRT_VERSION"]
    $installDir = Join-Path $DependencyRoot "TensorRT-$version"
    
    Write-StatusMessage "Setting up TensorRT $version..."
    Write-WarningMessage "TensorRT requires manual download from NVIDIA Developer website"
    Write-StatusMessage "1. Download TensorRT for Windows from: https://developer.nvidia.com/tensorrt"
    Write-StatusMessage "2. Extract to: $installDir"
    Write-StatusMessage "3. Ensure CUDA $($Versions['CUDA_VERSION']) is installed"
}

# Setup GGML
function Setup-GGML {
    param([hashtable]$Versions)
    
    $installDir = Join-Path $DependencyRoot "ggml"
    
    if ((Test-Path $installDir) -and -not $Force) {
        Write-WarningMessage "GGML already installed at $installDir. Use -Force to reinstall."
        return
    }
    
    Write-StatusMessage "Setting up GGML..."
    
    # Clone GGML repository
    if (Get-Command git -ErrorAction SilentlyContinue) {
        New-Item -ItemType Directory -Force -Path $DependencyRoot | Out-Null
        Push-Location $DependencyRoot
        
        if (Test-Path $installDir) {
            Remove-Item -Recurse -Force $installDir
        }
        
        git clone https://github.com/ggerganov/ggml.git
        
        Pop-Location
        
        Write-SuccessMessage "GGML cloned to $installDir"
        Write-StatusMessage "Build GGML using CMake:"
        Write-StatusMessage "  cd $installDir"
        Write-StatusMessage "  cmake -B build -DCMAKE_BUILD_TYPE=Release"
        Write-StatusMessage "  cmake --build build --config Release"
    }
    else {
        Write-ErrorMessage "Git is not installed. Please install Git and try again."
        exit 1
    }
}

# Setup OpenCV
function Setup-OpenCV {
    Write-StatusMessage "Setting up OpenCV..."
    Write-StatusMessage "Install OpenCV via vcpkg:"
    Write-StatusMessage "  vcpkg install opencv[contrib,dnn]:x64-windows"
    Write-StatusMessage ""
    Write-StatusMessage "Or download pre-built binaries from: https://opencv.org/releases/"
}

# Setup glog
function Setup-Glog {
    Write-StatusMessage "Setting up glog..."
    Write-StatusMessage "Install glog via vcpkg:"
    Write-StatusMessage "  vcpkg install glog:x64-windows"
}

# Validate installation
function Test-Installation {
    param(
        [string]$Backend,
        [hashtable]$Versions
    )
    
    Write-StatusMessage "Validating $Backend installation..."
    
    switch ($Backend) {
        "ONNX_RUNTIME" {
            $version = $Versions["ONNX_RUNTIME_VERSION"]
            $installDir = Join-Path $DependencyRoot "onnxruntime-win-x64-gpu-$version"
            $headerFile = Join-Path $installDir "include\onnxruntime_cxx_api.h"
            $libFile = Join-Path $installDir "lib\onnxruntime.lib"
            
            if ((Test-Path $headerFile) -and (Test-Path $libFile)) {
                Write-SuccessMessage "ONNX Runtime validation passed"
            }
            else {
                Write-WarningMessage "ONNX Runtime installation may be incomplete"
            }
        }
        "LIBTORCH" {
            $installDir = Join-Path $DependencyRoot "libtorch"
            $headerFile = Join-Path $installDir "include\torch\torch.h"
            $libFile = Join-Path $installDir "lib\torch.lib"
            
            if ((Test-Path $headerFile) -and (Test-Path $libFile)) {
                Write-SuccessMessage "LibTorch validation passed"
            }
            else {
                Write-WarningMessage "LibTorch installation may be incomplete"
            }
        }
    }
}

# Create environment setup script
function New-EnvironmentSetupScript {
    param([hashtable]$Versions)
    
    $setupScript = Join-Path $DependencyRoot "setup_neuriplo_env.ps1"
    
    $content = @"
# Environment setup script for neuriplo dependencies on Windows
# Generated by setup_dependencies.ps1

`$env:DEPENDENCY_ROOT = "$DependencyRoot"

# Add dependency paths to PATH
`$paths = @()

# ONNX Runtime
`$onnxRuntimeDir = Join-Path `$env:DEPENDENCY_ROOT "onnxruntime-win-x64-gpu-$($Versions['ONNX_RUNTIME_VERSION'])"
if (Test-Path `$onnxRuntimeDir) {
    `$env:ONNX_RUNTIME_DIR = `$onnxRuntimeDir
    `$paths += Join-Path `$onnxRuntimeDir "lib"
}

# LibTorch
`$libtorchDir = Join-Path `$env:DEPENDENCY_ROOT "libtorch"
if (Test-Path `$libtorchDir) {
    `$env:LIBTORCH_DIR = `$libtorchDir
    `$paths += Join-Path `$libtorchDir "lib"
}

# TensorRT
`$tensorrtDir = Join-Path `$env:DEPENDENCY_ROOT "TensorRT-$($Versions['TENSORRT_VERSION'])"
if (Test-Path `$tensorrtDir) {
    `$env:TENSORRT_DIR = `$tensorrtDir
    `$paths += Join-Path `$tensorrtDir "lib"
}

# OpenVINO
`$openvinoDir = Join-Path `$env:DEPENDENCY_ROOT "openvino_$($Versions['OPENVINO_VERSION'])"
if (Test-Path `$openvinoDir) {
    `$env:OPENVINO_DIR = `$openvinoDir
    `$paths += Join-Path `$openvinoDir "runtime\bin\intel64\Release"
}

# GGML
`$ggmlDir = Join-Path `$env:DEPENDENCY_ROOT "ggml"
if (Test-Path `$ggmlDir) {
    `$env:GGML_DIR = `$ggmlDir
    `$paths += Join-Path `$ggmlDir "build\bin\Release"
}

# Update PATH
foreach (`$p in `$paths) {
    if (Test-Path `$p) {
        `$env:Path = "`$p;`$env:Path"
    }
}

Write-Host "neuriplo environment variables set" -ForegroundColor Green
Write-Host "DEPENDENCY_ROOT: `$env:DEPENDENCY_ROOT"
"@

    Set-Content -Path $setupScript -Value $content
    Write-SuccessMessage "Environment setup script created: $setupScript"
    Write-StatusMessage "To use the dependencies, run: . $setupScript"
}

# Main execution
function Main {
    Write-StatusMessage "================================================"
    Write-StatusMessage "neuriplo Windows Dependency Setup Script"
    Write-StatusMessage "================================================"
    Write-StatusMessage "Backend: $Backend"
    Write-StatusMessage "Dependency Root: $DependencyRoot"
    Write-StatusMessage "Force Reinstall: $Force"
    Write-StatusMessage ""
    
    # Check for Administrator privileges for system-wide installations
    if (-not (Test-Administrator)) {
        Write-WarningMessage "Not running as Administrator. Some operations may fail."
    }
    
    # Load versions
    $versions = Get-VersionsFromEnv
    
    # Create dependency root directory
    New-Item -ItemType Directory -Force -Path $DependencyRoot | Out-Null
    
    # Setup system dependencies
    Write-StatusMessage "Installing system dependencies..."
    Setup-OpenCV
    Setup-Glog
    Write-StatusMessage ""
    
    # Setup backend-specific dependencies
    switch ($Backend) {
        "ONNX_RUNTIME" {
            Setup-ONNXRuntime -Versions $versions
        }
        "LIBTORCH" {
            Setup-LibTorch -Versions $versions
        }
        "TENSORRT" {
            Setup-TensorRT -Versions $versions
        }
        "OPENVINO" {
            Setup-OpenVINO -Versions $versions
        }
        "GGML" {
            Setup-GGML -Versions $versions
        }
        "OPENCV_DNN" {
            Write-StatusMessage "OpenCV DNN is included with OpenCV installation"
        }
        "LIBTENSORFLOW" {
            Write-WarningMessage "TensorFlow C++ library setup for Windows is complex"
            Write-StatusMessage "Please refer to TensorFlow documentation for Windows installation"
        }
    }
    
    # Validate installation
    Test-Installation -Backend $Backend -Versions $versions
    
    # Create environment setup script
    New-EnvironmentSetupScript -Versions $versions
    
    Write-StatusMessage ""
    Write-SuccessMessage "Setup completed successfully!"
    Write-StatusMessage ""
    Write-StatusMessage "Next steps:"
    Write-StatusMessage "1. Source the environment setup script: . $DependencyRoot\setup_neuriplo_env.ps1"
    Write-StatusMessage "2. Build neuriplo with CMake:"
    Write-StatusMessage "   cmake -B build -DDEFAULT_BACKEND=$Backend"
    Write-StatusMessage "   cmake --build build --config Release"
}

# Run main function
Main
