#include "LibtorchRuntimeFactory.hpp"

#include "LibtorchModelFormat.hpp"
#include "testing/TestBlob.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

namespace fs = std::filesystem;

namespace {

// Reads a model path written next to the test binary by model_downloader.py.
std::string model_from_path_file(const std::string& path_file) {
    std::ifstream in(path_file);
    std::string model_path;
    if (in) {
        std::getline(in, model_path);
    }
    return !model_path.empty() && fs::exists(model_path) ? model_path : std::string();
}

template <typename Fn> std::string load_error(Fn&& fn) {
    try {
        fn();
    } catch (const ModelLoadException& e) {
        return e.what();
    }
    return {};
}

const std::vector<std::vector<int64_t>> kResnetInput = {{1, 3, 224, 224}};

} // namespace

TEST(LibtorchModelFormatTest, DetectsFormatFromExtension) {
    EXPECT_EQ(detect_libtorch_model_format("resnet18.pt2"), LibtorchModelFormat::AOTInductor);
    EXPECT_EQ(detect_libtorch_model_format("/models/RESNET18.PT2"), LibtorchModelFormat::AOTInductor);
    EXPECT_EQ(detect_libtorch_model_format("resnet18.pt"), LibtorchModelFormat::TorchScript);
    EXPECT_EQ(detect_libtorch_model_format("resnet18.pth"), LibtorchModelFormat::TorchScript);
    EXPECT_EQ(detect_libtorch_model_format("resnet18"), LibtorchModelFormat::TorchScript);
    EXPECT_EQ(detect_libtorch_model_format("resnet18.pt2.bak"), LibtorchModelFormat::TorchScript);
    EXPECT_EQ(detect_libtorch_model_format("pkgs.pt2/model"), LibtorchModelFormat::TorchScript);
}

TEST(LibtorchRuntimeFactoryTest, RoutesPt2ToAOTInductor) {
    LibtorchRuntimeFactory factory;
    const auto error = load_error([&] { factory.create_backend("missing_model.pt2", false, 1, kResnetInput); });
    ASSERT_FALSE(error.empty()) << "a missing .pt2 must fail with ModelLoadException";
    EXPECT_NE(error.find("AOTInductor"), std::string::npos) << error;
}

TEST(LibtorchRuntimeFactoryTest, RoutesOtherExtensionsToTorchScript) {
    LibtorchRuntimeFactory factory;
    const auto error = load_error([&] { factory.create_backend("missing_model.pt", false, 1, kResnetInput); });
    ASSERT_FALSE(error.empty()) << "a missing .pt must fail with ModelLoadException";
    EXPECT_EQ(error.find("AOTInductor"), std::string::npos) << error;
}

TEST(LibtorchRuntimeFactoryTest, AOTInductorRequiresInputSizes) {
    const auto model_path = model_from_path_file("aoti_model_path.txt");
    if (model_path.empty()) {
        GTEST_SKIP() << "no AOTInductor package available (aoti_model_path.txt)";
    }
    LibtorchRuntimeFactory factory;
    const auto error = load_error([&] { factory.create_backend(model_path, false, 1, {}); });
    EXPECT_NE(error.find("input_sizes"), std::string::npos) << error;
}

TEST(LibtorchRuntimeFactoryTest, AOTInductorRejectsDeviceMismatch) {
    const auto model_path = model_from_path_file("aoti_model_path.txt");
    if (model_path.empty()) {
        GTEST_SKIP() << "no AOTInductor package available (aoti_model_path.txt)";
    }
    // The test package is compiled for CPU; asking for the GPU must fail
    // rather than silently run on the CPU.
    LibtorchRuntimeFactory factory;
    const auto error = load_error([&] { factory.create_backend(model_path, true, 1, kResnetInput); });
    EXPECT_NE(error.find("compiled for CPU"), std::string::npos) << error;
}

TEST(LibtorchRuntimeFactoryTest, AOTInductorMatchesTorchScript) {
    const auto aoti_path = model_from_path_file("aoti_model_path.txt");
    if (aoti_path.empty()) {
        GTEST_SKIP() << "no AOTInductor package available (aoti_model_path.txt)";
    }

    LibtorchRuntimeFactory factory;
    auto aoti = factory.create_backend(aoti_path, false, 1, kResnetInput);
    ASSERT_EQ(aoti->get_inference_metadata().getOutputs().size(), 1U);
    EXPECT_EQ(aoti->get_inference_metadata().getOutputs()[0].shape, (std::vector<int64_t>{1, 1000}));

    const auto input = neuriplo::testing::constant_blob_tensors(neuriplo::testing::kGray128Normalized);
    auto [aoti_out, aoti_shapes] = aoti->get_infer_results(input);
    ASSERT_EQ(aoti_shapes.size(), 1U);
    ASSERT_EQ(aoti_shapes[0], (std::vector<int64_t>{1, 1000}));
    for (const auto& v : aoti_out[0]) {
        ASSERT_TRUE(std::isfinite(std::get<float>(v)));
    }

    // Both test models are exported from the same torchvision ResNet-18
    // weights, so the two runtimes must agree.
    const auto ts_path = model_from_path_file("model_path.txt");
    if (ts_path.empty() || detect_libtorch_model_format(ts_path) != LibtorchModelFormat::TorchScript) {
        return;
    }
    auto ts = factory.create_backend(ts_path, false, 1, kResnetInput);
    auto [ts_out, ts_shapes] = ts->get_infer_results(input);
    ASSERT_EQ(ts_shapes, aoti_shapes);
    for (size_t i = 0; i < ts_out[0].size(); ++i) {
        ASSERT_NEAR(std::get<float>(ts_out[0][i]), std::get<float>(aoti_out[0][i]), 1e-3F) << "logit " << i;
    }
}
