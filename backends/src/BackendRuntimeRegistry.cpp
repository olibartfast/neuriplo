#include "BackendRuntimeRegistry.hpp"

#ifdef USE_ONNX_RUNTIME
#include "ORTRuntimeFactory.hpp"
#endif
#ifdef USE_LIBTORCH
#include "LibtorchRuntimeFactory.hpp"
#endif
#ifdef USE_LIBTENSORFLOW
#include "TFRuntimeFactory.hpp"
#endif
#ifdef USE_OPENCV_DNN
#include "OCVDNNRuntimeFactory.hpp"
#endif
#ifdef USE_TENSORRT
#include "TRTRuntimeFactory.hpp"
#endif
#ifdef USE_OPENVINO
#include "OVRuntimeFactory.hpp"
#endif
#ifdef USE_GGML
#include "GGMLRuntimeFactory.hpp"
#endif
#ifdef USE_TVM
#include "TVMRuntimeFactory.hpp"
#endif
#ifdef USE_CACTUS
#include "CactusRuntimeFactory.hpp"
#endif
#ifdef USE_MIGRAPHX
#include "MIGraphXRuntimeFactory.hpp"
#endif
#ifdef USE_LLAMACPP
#include "LlamaCppRuntimeFactory.hpp"
#endif
#ifdef USE_EXECUTORCH
#include "ExecuTorchRuntimeFactory.hpp"
#endif
#ifdef USE_LITERT
#include "LiteRTRuntimeFactory.hpp"
#endif

#include <cstring>
#include <memory>

namespace {

template <typename Factory> std::unique_ptr<IBackendRuntimeFactory> make_factory() {
    return std::make_unique<Factory>();
}

// Single TU holding every compiled-in registration. Backends are appended under
// independent USE_* guards so any subset can coexist in one build; intentionally
// not self-registering static initializers, which the linker may drop from
// static/object libraries.
std::vector<BackendRuntimeRegistration> build_registrations() {
    std::vector<BackendRuntimeRegistration> registrations;
#ifdef USE_ONNX_RUNTIME
    registrations.push_back({"ONNX_RUNTIME", "ONNX Runtime", &make_factory<ORTRuntimeFactory>, false});
#endif
#ifdef USE_LIBTORCH
    registrations.push_back({"LIBTORCH", "LibTorch", &make_factory<LibtorchRuntimeFactory>, false});
#endif
#ifdef USE_LIBTENSORFLOW
    registrations.push_back({"LIBTENSORFLOW", "TensorFlow", &make_factory<TFRuntimeFactory>, false});
#endif
#ifdef USE_OPENCV_DNN
    registrations.push_back({"OPENCV_DNN", "OpenCV DNN", &make_factory<OCVDNNRuntimeFactory>, false});
#endif
#ifdef USE_TENSORRT
    registrations.push_back({"TENSORRT", "TensorRT", &make_factory<TRTRuntimeFactory>, true});
#endif
#ifdef USE_OPENVINO
    registrations.push_back({"OPENVINO", "OpenVINO", &make_factory<OVRuntimeFactory>, false});
#endif
#ifdef USE_GGML
    registrations.push_back({"GGML", "GGML", &make_factory<GGMLRuntimeFactory>, false});
#endif
#ifdef USE_TVM
    registrations.push_back({"TVM", "TVM", &make_factory<TVMRuntimeFactory>, false});
#endif
#ifdef USE_CACTUS
    registrations.push_back({"CACTUS", "Cactus", &make_factory<CactusRuntimeFactory>, false});
#endif
#ifdef USE_MIGRAPHX
    registrations.push_back({"MIGRAPHX", "MIGraphX", &make_factory<MIGraphXRuntimeFactory>, false});
#endif
#ifdef USE_LLAMACPP
    registrations.push_back({"LLAMACPP", "llama.cpp", &make_factory<LlamaCppRuntimeFactory>, false});
#endif
#ifdef USE_EXECUTORCH
    registrations.push_back({"EXECUTORCH", "ExecuTorch", &make_factory<ExecuTorchRuntimeFactory>, false});
#endif
#ifdef USE_LITERT
    registrations.push_back({"LITERT", "LiteRT", &make_factory<LiteRTRuntimeFactory>, false});
#endif
    return registrations;
}

} // namespace

const std::vector<BackendRuntimeRegistration>& get_registered_backends() noexcept {
    static const std::vector<BackendRuntimeRegistration> registrations = build_registrations();
    return registrations;
}

const BackendRuntimeRegistration* find_backend_registration(std::string_view id) noexcept {
    for (const BackendRuntimeRegistration& registration : get_registered_backends()) {
        if (id == registration.id) {
            return &registration;
        }
    }
    return nullptr;
}

const BackendRuntimeRegistration* get_compiled_backend_registration() noexcept {
#ifdef NEURIPLO_DEFAULT_BACKEND
    if (const BackendRuntimeRegistration* registration = find_backend_registration(NEURIPLO_DEFAULT_BACKEND)) {
        return registration;
    }
#endif
    const std::vector<BackendRuntimeRegistration>& registrations = get_registered_backends();
    return registrations.empty() ? nullptr : &registrations.front();
}

const char* compiled_backend_id() noexcept {
    const BackendRuntimeRegistration* registration = get_compiled_backend_registration();
    return registration ? registration->id : "";
}

std::unique_ptr<IBackendRuntimeFactory> create_compiled_backend_factory() {
    const BackendRuntimeRegistration* registration = get_compiled_backend_registration();
    if (!registration || !registration->create_factory) {
        return nullptr;
    }
    return registration->create_factory();
}
