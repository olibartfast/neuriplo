#include "BackendRuntimeRegistry.hpp"

#ifdef USE_ONNX_RUNTIME
#include "ORTRuntimeFactory.hpp"
#elif USE_LIBTORCH
#include "LibtorchRuntimeFactory.hpp"
#elif USE_LIBTENSORFLOW
#include "TFRuntimeFactory.hpp"
#elif USE_OPENCV_DNN
#include "OCVDNNRuntimeFactory.hpp"
#elif USE_TENSORRT
#include "TRTRuntimeFactory.hpp"
#elif USE_OPENVINO
#include "OVRuntimeFactory.hpp"
#elif USE_GGML
#include "GGMLRuntimeFactory.hpp"
#elif USE_TVM
#include "TVMRuntimeFactory.hpp"
#elif USE_CACTUS
#include "CactusRuntimeFactory.hpp"
#elif USE_MIGRAPHX
#include "MIGraphXRuntimeFactory.hpp"
#elif USE_LLAMACPP
#include "LlamaCppRuntimeFactory.hpp"
#elif USE_EXECUTORCH
#include "ExecuTorchRuntimeFactory.hpp"
#elif USE_LITERT
#include "LiteRTRuntimeFactory.hpp"
#endif

#include <memory>

namespace {

template <typename Factory> std::unique_ptr<IBackendRuntimeFactory> make_factory() {
    return std::make_unique<Factory>();
}

const BackendRuntimeRegistration* selected_registration() noexcept {
#ifdef USE_ONNX_RUNTIME
    static const BackendRuntimeRegistration registration{"ONNX_RUNTIME", "ONNX Runtime",
                                                         &make_factory<ORTRuntimeFactory>, false};
#elif USE_LIBTORCH
    static const BackendRuntimeRegistration registration{"LIBTORCH", "LibTorch", &make_factory<LibtorchRuntimeFactory>,
                                                         false};
#elif USE_LIBTENSORFLOW
    static const BackendRuntimeRegistration registration{"LIBTENSORFLOW", "TensorFlow", &make_factory<TFRuntimeFactory>,
                                                         false};
#elif USE_OPENCV_DNN
    static const BackendRuntimeRegistration registration{"OPENCV_DNN", "OpenCV DNN",
                                                         &make_factory<OCVDNNRuntimeFactory>, false};
#elif USE_TENSORRT
    static const BackendRuntimeRegistration registration{"TENSORRT", "TensorRT", &make_factory<TRTRuntimeFactory>,
                                                         true};
#elif USE_OPENVINO
    static const BackendRuntimeRegistration registration{"OPENVINO", "OpenVINO", &make_factory<OVRuntimeFactory>,
                                                         false};
#elif USE_GGML
    static const BackendRuntimeRegistration registration{"GGML", "GGML", &make_factory<GGMLRuntimeFactory>, false};
#elif USE_TVM
    static const BackendRuntimeRegistration registration{"TVM", "TVM", &make_factory<TVMRuntimeFactory>, false};
#elif USE_CACTUS
    static const BackendRuntimeRegistration registration{"CACTUS", "Cactus", &make_factory<CactusRuntimeFactory>,
                                                         false};
#elif USE_MIGRAPHX
    static const BackendRuntimeRegistration registration{"MIGRAPHX", "MIGraphX", &make_factory<MIGraphXRuntimeFactory>,
                                                         false};
#elif USE_LLAMACPP
    static const BackendRuntimeRegistration registration{"LLAMACPP", "llama.cpp", &make_factory<LlamaCppRuntimeFactory>,
                                                         false};
#elif USE_EXECUTORCH
    static const BackendRuntimeRegistration registration{"EXECUTORCH", "ExecuTorch",
                                                         &make_factory<ExecuTorchRuntimeFactory>, false};
#elif USE_LITERT
    static const BackendRuntimeRegistration registration{"LITERT", "LiteRT", &make_factory<LiteRTRuntimeFactory>,
                                                         false};
#else
    return nullptr;
#endif
    return &registration;
}

} // namespace

const BackendRuntimeRegistration* get_compiled_backend_registration() noexcept { return selected_registration(); }

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
