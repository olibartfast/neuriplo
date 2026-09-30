#include "NativeInfer.hpp"
#include "ORTInfer.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr const char* kFixtureEnv = "NEURIPLO_PARITY_FIXTURE";
constexpr size_t kInputElements = 1u * 3u * 224u * 224u;
constexpr size_t kOutputElements = 1000u;
constexpr double kMaxAllowedAbsDiff = 1e-4;

// The fixture path is supplied by ctest through the environment. A missing or
// absent fixture is a hard failure: never a skip, never a mock.
std::string resolve_fixture() {
    const char* from_env = std::getenv(kFixtureEnv);
    if (from_env != nullptr && from_env[0] != '\0') {
        return from_env;
    }
    return "";
}

// One deterministic input shared verbatim by both backends.
std::vector<std::vector<uint8_t>> make_input() {
    std::vector<uint8_t> bytes(kInputElements * sizeof(float));
    for (size_t i = 0; i < kInputElements; ++i) {
        const float value = 0.01F * static_cast<float>((i % 7) + 1);
        std::memcpy(bytes.data() + i * sizeof(float), &value, sizeof(float));
    }
    return {std::move(bytes)};
}

std::vector<float> flatten_floats(const std::vector<std::vector<TensorElement>>& outputs) {
    std::vector<float> values;
    values.reserve(outputs[0].size());
    for (const TensorElement& element : outputs[0]) {
        values.push_back(std::get<float>(element));
    }
    return values;
}

} // namespace

// [T-22]/[V-6]: the same file and the same input through NATIVE and
// ONNX_RUNTIME, compared elementwise. [A-2] bounds the maximum absolute
// difference at 1e-4.
TEST(ParityTest, NativeAndOnnxRuntimeAgreeElementwise) {
    const std::string fixture = resolve_fixture();
    ASSERT_FALSE(fixture.empty()) << kFixtureEnv
                                  << " is not set; the parity fixture must be provisioned before this test runs";
    ASSERT_TRUE(std::filesystem::exists(fixture)) << "parity fixture does not exist: " << fixture;

    const std::vector<std::vector<uint8_t>> input = make_input();

    std::vector<float> native_output;
    try {
        NativeInfer native(fixture, /*use_gpu=*/false);
        auto [outputs, shapes] = native.get_infer_results(input);
        ASSERT_EQ(outputs.size(), 1u) << "NATIVE returned " << outputs.size() << " outputs, expected 1";
        ASSERT_EQ(shapes.size(), 1u) << "NATIVE returned " << shapes.size() << " shapes, expected 1";
        ASSERT_EQ(outputs[0].size(), kOutputElements)
            << "NATIVE returned " << outputs[0].size() << " elements, expected " << kOutputElements;
        native_output = flatten_floats(outputs);
    } catch (const std::exception& ex) {
        FAIL() << "NATIVE inference failed: " << ex.what();
    }

    std::vector<float> ort_output;
    try {
        ORTInfer ort(fixture, /*use_gpu=*/false);
        auto [outputs, shapes] = ort.get_infer_results(input);
        ASSERT_EQ(outputs.size(), 1u) << "ONNX_RUNTIME returned " << outputs.size() << " outputs, expected 1";
        ASSERT_EQ(shapes.size(), 1u) << "ONNX_RUNTIME returned " << shapes.size() << " shapes, expected 1";
        ASSERT_EQ(outputs[0].size(), kOutputElements)
            << "ONNX_RUNTIME returned " << outputs[0].size() << " elements, expected " << kOutputElements;
        ort_output = flatten_floats(outputs);
    } catch (const std::exception& ex) {
        FAIL() << "ONNX_RUNTIME inference failed: " << ex.what();
    }

    ASSERT_EQ(native_output.size(), kOutputElements);
    ASSERT_EQ(ort_output.size(), kOutputElements);

    double max_abs_diff = 0.0;
    for (size_t i = 0; i < kOutputElements; ++i) {
        const double diff = std::fabs(static_cast<double>(native_output[i]) - static_cast<double>(ort_output[i]));
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
        }
    }

    std::cout << "PARITY_MAX_ABS_DIFF=" << max_abs_diff << std::endl;
    EXPECT_LE(max_abs_diff, kMaxAllowedAbsDiff) << "NATIVE and ONNX_RUNTIME diverged: max absolute difference "
                                                << max_abs_diff << " exceeds " << kMaxAllowedAbsDiff;
}
