// Compile coverage for BackendTestTemplate.hpp and MockInferenceInterface.hpp.
//
// Both headers are kept for docs/REFACTOR_DESIGN_PATTERNS.md S6.3 but are
// included by no other translation unit, so nothing ever type-checked them.
// That is how effe518 was able to change what the shared template feeds a
// backend -- gray 128 became zero, and a resized 224x224 case became a real
// 1024x1024 tensor -- without any build noticing. This file closes that gap.

#include "BackendTestTemplate.hpp"
#include "MockInferenceInterface.hpp"

#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

namespace {

// The template takes the backend type it drives. MockInferenceInterface is the
// only one a backend-agnostic test can name, and since no model is discovered
// here the instance is never constructed -- the point is the instantiation.
using TemplateFixture = BackendHybridTestBase<MockInferenceInterface>;

// A class template's member bodies are instantiated only when used, so naming
// the type would compile-check almost nothing. Odr-using each member forces
// its definition to be instantiated and type-checked. Deriving is what gives
// access: they are all protected.
struct TemplateProbe : TemplateFixture {
    static void InstantiateEveryMember() {
        auto create = &TemplateProbe::CreateTestInput;
        auto benchmark = &TemplateProbe::RunPerformanceBenchmark;
        auto leak = &TemplateProbe::DetectMemoryLeak;
        auto edges = &TemplateProbe::TestEdgeCases;
        auto stress = &TemplateProbe::RunStressTest;
        auto skip = &TemplateProbe::SkipIfNoRealModel;
        (void)create;
        (void)benchmark;
        (void)leak;
        (void)edges;
        (void)stress;
        (void)skip;
    }

    using TemplateFixture::CreateTestInput;

    // The two extension points every real backend test must supply. Providing
    // them here is what makes the fixture concrete, and type-checks the
    // signatures an adopter has to match.
    std::unique_ptr<MockInferenceInterface> CreateBackendInstance() override {
        return std::make_unique<MockInferenceInterface>();
    }

    std::string GetBackendName() override { return "mock"; }

    // ::testing::Test leaves TestBody() pure, so a fixture is otherwise
    // abstract. This probe is driven directly rather than by the framework.
    void TestBody() override {}
};

TEST(BackendTestTemplateCompile, EveryMemberInstantiates) {
    TemplateProbe::InstantiateEveryMember();
    SUCCEED() << "BackendTestTemplate.hpp instantiates against MockInferenceInterface";
}

// Guards the specific thing effe518 changed: the template's standard input is
// the gray image the old cv::dnn::blobFromImage call produced, not zeros.
TEST(BackendTestTemplateCompile, CreateTestInputIsTheGrayBlob) {
    TemplateProbe probe;
    const std::vector<std::vector<uint8_t>> input = probe.CreateTestInput();

    ASSERT_EQ(input.size(), 1U) << "the template drives single-input backends";

    // 1x3x224x224 float32, the shape blobFromImage emitted after its resize.
    constexpr std::size_t kElements = 1 * 3 * 224 * 224;
    ASSERT_EQ(input[0].size(), kElements * sizeof(float));

    float first = 0.0F;
    float last = 0.0F;
    std::memcpy(&first, input[0].data(), sizeof(float));
    std::memcpy(&last, input[0].data() + input[0].size() - sizeof(float), sizeof(float));

    EXPECT_FLOAT_EQ(first, neuriplo::testing::kGray128Normalized);
    EXPECT_FLOAT_EQ(last, neuriplo::testing::kGray128Normalized);
    EXPECT_GT(first, 0.0F) << "zeros here would exercise a path the old tests never took";
}

// The other half of the same regression: the zero-input edge case genuinely
// was zeros, and must stay that way.
TEST(BackendTestTemplateCompile, ZeroBlobIsStillZero) {
    const std::vector<std::vector<uint8_t>> zeros = neuriplo::testing::zero_blob_tensors();
    ASSERT_EQ(zeros.size(), 1U);
    EXPECT_EQ(zeros[0].size(), std::size_t{1} * 3 * 224 * 224 * sizeof(float));
    EXPECT_TRUE(std::all_of(zeros[0].begin(), zeros[0].end(), [](uint8_t byte) { return byte == 0; }));
}

} // namespace
