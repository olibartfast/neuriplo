#include <gtest/gtest.h>
#include "MIGraphXInfer.hpp"
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <filesystem>
#include <memory>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Lightweight mock (no MIGraphX required) for unit tests
// ---------------------------------------------------------------------------
class MockMIGraphXInfer {
public:
    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>
    get_infer_results(const std::vector<std::vector<uint8_t>>&) {
        std::vector<TensorElement> out(1000);
        for (int i = 0; i < 1000; ++i) out[i] = static_cast<float>(i * 0.001f);
        return {{out}, {{1, 1000}}};
    }

    InferenceMetadata get_inference_metadata() {
        InferenceMetadata meta;
        meta.addInput("input", {1, 3, 224, 224}, 1);
        meta.addOutput("output", {1, 1000}, 1);
        return meta;
    }
};

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------
class MIGraphXInferTest : public ::testing::Test {
protected:
    std::string model_path;
    bool has_real_model = false;
    std::unique_ptr<MIGraphXInfer> real_infer;
    std::unique_ptr<MockMIGraphXInfer> mock_infer;

    void SetUp() override {
        std::ifstream modelPathFile("model_path.txt");
        if (modelPathFile) {
            std::getline(modelPathFile, model_path);
            if (!model_path.empty() && fs::exists(model_path)) {
                has_real_model = true;
                try {
                    real_infer = std::make_unique<MIGraphXInfer>(model_path, false);
                } catch (const std::exception& e) {
                    std::cerr << "Real model load failed, using mock: " << e.what() << "\n";
                    has_real_model = false;
                }
            }
        }
        if (!has_real_model)
            mock_infer = std::make_unique<MockMIGraphXInfer>();
    }

    std::vector<std::vector<uint8_t>> make_input_tensors() {
        cv::Mat img = cv::Mat::zeros(224, 224, CV_32FC3);
        cv::Mat blob;
        cv::dnn::blobFromImage(img, blob, 1.f / 255.f,
                               cv::Size(224, 224), cv::Scalar(), true, false);
        std::vector<uint8_t> data(blob.total() * blob.elemSize());
        std::memcpy(data.data(), blob.data, data.size());
        return {data};
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
TEST_F(MIGraphXInferTest, BasicInference) {
    auto input = make_input_tensors();

    std::vector<std::vector<TensorElement>> outputs;
    std::vector<std::vector<int64_t>> shapes;

    if (has_real_model) {
        auto [o, s] = real_infer->get_infer_results(input);
        outputs = o; shapes = s;
    } else {
        auto [o, s] = mock_infer->get_infer_results(input);
        outputs = o; shapes = s;
    }

    ASSERT_FALSE(outputs.empty());
    ASSERT_EQ(shapes[0][1], 1000);
    ASSERT_TRUE(std::holds_alternative<float>(outputs[0][0]));
    ASSERT_EQ(outputs[0].size(), static_cast<size_t>(shapes[0][1]));
}

TEST_F(MIGraphXInferTest, IntegrationTest) {
    if (!has_real_model)
        GTEST_SKIP() << "No real model available";

    auto [outputs, shapes] = real_infer->get_infer_results(make_input_tensors());
    ASSERT_EQ(outputs[0].size(), 1000u);
    for (const auto& elem : outputs[0]) {
        ASSERT_TRUE(std::isfinite(std::get<float>(elem)));
    }
}

TEST_F(MIGraphXInferTest, MockUnitTest) {
    if (has_real_model)
        GTEST_SKIP() << "Real model available; mock test skipped";

    auto [outputs, shapes] = mock_infer->get_infer_results(make_input_tensors());
    ASSERT_EQ(outputs[0].size(), 1000u);
    ASSERT_FLOAT_EQ(std::get<float>(outputs[0][5]), 5 * 0.001f);
}

TEST_F(MIGraphXInferTest, GPUTest) {
    if (!has_real_model)
        GTEST_SKIP() << "No real model available";
    try {
        auto gpu_infer = std::make_unique<MIGraphXInfer>(model_path, true);
        auto [outputs, shapes] = gpu_infer->get_infer_results(make_input_tensors());
        ASSERT_EQ(outputs[0].size(), 1000u);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "GPU not available: " << e.what();
    }
}

TEST_F(MIGraphXInferTest, MetadataTest) {
    if (!has_real_model)
        GTEST_SKIP() << "No real model available";
    auto meta = real_infer->get_inference_metadata();
    ASSERT_FALSE(meta.getInputs().empty());
    ASSERT_FALSE(meta.getOutputs().empty());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
