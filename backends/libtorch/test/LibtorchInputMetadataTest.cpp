#include "LibtorchInfer.hpp"
#include "LibtorchTensorUtils.hpp"

#include <cstdlib>
#include <gtest/gtest.h>
#include <string>

// The metadata published for runtime input packing must carry each graph
// input's real scalar type. Regression: the constructor used to advertise
// every input as FP32 while only the internal packing list kept the real
// types, so an INT64 size tensor next to an FP32 image was packed as float.
TEST(LibtorchScalarToDatatype, CoversBackendSupportedKinds) {
    EXPECT_EQ(libtorch_scalar_to_datatype(c10::ScalarType::Float), TensorDataType::Float32);
    EXPECT_EQ(libtorch_scalar_to_datatype(c10::ScalarType::Int), TensorDataType::Int32);
    EXPECT_EQ(libtorch_scalar_to_datatype(c10::ScalarType::Long), TensorDataType::Int64);
    EXPECT_EQ(libtorch_scalar_to_datatype(c10::ScalarType::Byte), TensorDataType::UInt8);
    EXPECT_EQ(libtorch_scalar_to_datatype(c10::ScalarType::Char), TensorDataType::Int8);
    EXPECT_EQ(libtorch_scalar_to_datatype(c10::ScalarType::Bool), TensorDataType::Bool);
}

TEST(LibtorchScalarToDatatype, RejectsKindsWithoutMetadataEquivalent) {
    EXPECT_THROW(libtorch_scalar_to_datatype(c10::ScalarType::Double), std::runtime_error);
    EXPECT_THROW(libtorch_scalar_to_datatype(c10::ScalarType::Half), std::runtime_error);
}

// End-to-end: the published metadata must match the dtypes the TorchScript
// graph itself carries for each input. A traced two-input model (FP32 image +
// INT64 sizes, the RT-DETR/DEIM convention) publishes [Float32, Int64] on a
// runtime that deserializes graph dtypes; on an older runtime that drops them
// (e.g. libtorch 2.0.1 reading a newer file) both sides agree on the explicit
// FP32 default. Either way the metadata must equal the graph truth: the
// regressed behaviour advertised FP32 while the graph said otherwise.
// The model path comes from NEURIPLO_LIBTORCH_TWO_INPUT_MODEL; the test
// skips when no compatible TorchScript file is provided, mirroring the
// existing model_path.txt pattern.
TEST(LibtorchInputMetadata, PublishesGraphInputDtypes) {
    const char* model_path = std::getenv("NEURIPLO_LIBTORCH_TWO_INPUT_MODEL");
    if (model_path == nullptr || std::string(model_path).empty()) {
        GTEST_SKIP() << "Set NEURIPLO_LIBTORCH_TWO_INPUT_MODEL to a two-input TorchScript file";
    }
    torch::jit::script::Module module = torch::jit::load(model_path);
    auto graph = module.get_method("forward").graph();
    auto graph_inputs = graph->inputs();
    ASSERT_GT(graph_inputs.size(), 1U); // first input is the module itself
    std::vector<TensorDataType> expected;
    for (size_t i = 1; i < graph_inputs.size(); ++i) {
        auto tensor_type = graph_inputs[i]->type()->cast<c10::TensorType>();
        ASSERT_TRUE(tensor_type != nullptr) << "graph input " << (i - 1) << " is not a tensor";
        if (tensor_type->scalarType().has_value()) {
            expected.push_back(libtorch_scalar_to_datatype(tensor_type->scalarType().value()));
        } else {
            expected.push_back(TensorDataType::Float32); // documented explicit default
        }
    }
    LibtorchInfer infer(model_path, false, 1, {{1, 3, 32, 32}, {1, 2}});
    const auto inputs = infer.get_inference_metadata().getInputs();
    ASSERT_EQ(inputs.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(inputs[i].datatype, expected[i]) << "metadata dtype mismatch on input " << i;
    }
    // The fixture's second input is an INT64 size tensor whenever the runtime
    // preserves graph dtypes; guard the real regression where it does.
    if (expected.size() == 2U && expected[1] != TensorDataType::Float32) {
        EXPECT_EQ(inputs[1].datatype, TensorDataType::Int64);
    }
}
