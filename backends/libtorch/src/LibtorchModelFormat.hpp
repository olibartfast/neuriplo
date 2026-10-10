#pragma once

#include <algorithm>
#include <cctype>
#include <string>

// The two LibTorch runtimes neuriplo can drive. They read different artifacts
// and are not interchangeable: an AOTInductor package cannot be loaded by
// torch::jit, and a TorchScript archive cannot be loaded by the AOTI loader.
enum class LibtorchModelFormat { TorchScript, AOTInductor };

// Picks the runtime from the model path's extension. ".pt2" (case-insensitive)
// is an AOTInductor package produced by torch._inductor.aoti_compile_and_package;
// everything else keeps the historical TorchScript path, so existing ".pt",
// ".pth" and extension-less models behave exactly as before.
inline LibtorchModelFormat detect_libtorch_model_format(const std::string& model_path) {
    const auto dot = model_path.find_last_of('.');
    const auto slash = model_path.find_last_of("/\\");
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        return LibtorchModelFormat::TorchScript;
    }

    std::string ext = model_path.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    return ext == "pt2" ? LibtorchModelFormat::AOTInductor : LibtorchModelFormat::TorchScript;
}

inline const char* to_string(LibtorchModelFormat format) noexcept {
    return format == LibtorchModelFormat::AOTInductor ? "AOTInductor" : "TorchScript";
}
