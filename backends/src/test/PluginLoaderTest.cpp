// Plugin loader integration tests. Built only when NEURIPLO_PLUGIN_BACKENDS is
// non-empty; NEURIPLO_TEST_PLUGIN_DIR points at the build's plugins/ output.
// The expected configuration is a host WITHOUT built-in ONNX Runtime plus the
// ONNX_RUNTIME plugin, proving a dlopen()ed backend serves real inference next
// to the compiled-in default.

#include "plugin/PluginLoader.hpp"

#include "BackendRuntimeRegistry.hpp"
#include "InferenceBackendSetup.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<uint8_t>;

void append_varint(Bytes& bytes, uint64_t value) {
    while (value >= 0x80) {
        bytes.push_back(static_cast<uint8_t>(value | 0x80));
        value >>= 7;
    }
    bytes.push_back(static_cast<uint8_t>(value));
}

void append_key(Bytes& bytes, int field_number, int wire_type) {
    append_varint(bytes, (static_cast<uint64_t>(field_number) << 3U) | static_cast<uint64_t>(wire_type));
}

void append_int64(Bytes& bytes, int field_number, int64_t value) {
    append_key(bytes, field_number, 0);
    append_varint(bytes, static_cast<uint64_t>(value));
}

void append_string(Bytes& bytes, int field_number, const std::string& value) {
    append_key(bytes, field_number, 2);
    append_varint(bytes, value.size());
    bytes.insert(bytes.end(), value.begin(), value.end());
}

void append_message(Bytes& bytes, int field_number, const Bytes& message) {
    append_key(bytes, field_number, 2);
    append_varint(bytes, message.size());
    bytes.insert(bytes.end(), message.begin(), message.end());
}

Bytes tensor_shape(const std::vector<int64_t>& shape) {
    Bytes bytes;
    for (const int64_t dimension : shape) {
        Bytes dim_bytes;
        append_int64(dim_bytes, 1, dimension);
        append_message(bytes, 1, dim_bytes);
    }
    return bytes;
}

Bytes value_info(const std::string& name, const std::vector<int64_t>& shape) {
    Bytes tensor_type;
    append_int64(tensor_type, 1, 1); // elem_type FLOAT
    append_message(tensor_type, 2, tensor_shape(shape));
    Bytes type_proto;
    append_message(type_proto, 1, tensor_type);
    Bytes bytes;
    append_string(bytes, 1, name);
    append_message(bytes, 2, type_proto);
    return bytes;
}

Bytes identity_onnx_model() {
    Bytes node;
    append_string(node, 1, "input");
    append_string(node, 2, "output");
    append_string(node, 3, "identity");
    append_string(node, 4, "Identity");

    const std::vector<int64_t> shape{1, 3, 4, 4};
    Bytes graph;
    append_message(graph, 1, node);
    append_string(graph, 2, "identity_graph");
    append_message(graph, 11, value_info("input", shape));
    append_message(graph, 12, value_info("output", shape));

    Bytes opset;
    append_int64(opset, 2, 13);

    Bytes model;
    append_int64(model, 1, 7); // ir_version
    append_string(model, 2, "neuriplo-plugin-test");
    append_message(model, 7, graph);
    append_message(model, 8, opset);
    return model;
}

std::string write_identity_model() {
    const auto path = std::filesystem::temp_directory_path() / "neuriplo-plugin-identity.onnx";
    std::ofstream output(path, std::ios::binary);
    const Bytes model = identity_onnx_model();
    output.write(reinterpret_cast<const char*>(model.data()), static_cast<std::streamsize>(model.size()));
    return path.string();
}

std::string plugin_dir() { return NEURIPLO_TEST_PLUGIN_DIR; }

} // namespace

TEST(PluginLoaderTest, DiscoversPluginsInDirectory) {
    load_backend_plugins(plugin_dir());
    const PluginBackendDescriptor* descriptor = find_plugin_backend("ONNX_RUNTIME");
    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(descriptor->display_name, "ONNX Runtime");
    EXPECT_FALSE(descriptor->force_gpu);
    EXPECT_NE(descriptor->api, nullptr);
}

TEST(PluginLoaderTest, PluginBackendIsNotCompiledIn) {
    // The test host build keeps ONNX Runtime plugin-only, so the plugin path
    // is genuinely exercised rather than shadowed by a built-in registration.
    EXPECT_EQ(find_backend_registration("ONNX_RUNTIME"), nullptr);
}

TEST(PluginLoaderTest, AvailableIdsMergeBuiltinsAndPlugins) {
    const auto ids = available_backend_ids(plugin_dir());
    EXPECT_NE(std::find(ids.begin(), ids.end(), NEURIPLO_DEFAULT_BACKEND), ids.end());
    EXPECT_NE(std::find(ids.begin(), ids.end(), "ONNX_RUNTIME"), ids.end());
}

TEST(PluginLoaderTest, LoadIsIdempotent) {
    load_backend_plugins(plugin_dir());
    const size_t count = get_plugin_backends().size();
    EXPECT_EQ(load_backend_plugins(plugin_dir()), 0u);
    EXPECT_EQ(get_plugin_backends().size(), count);
}

TEST(PluginLoaderTest, ServesIdentityModelThroughPlugin) {
    EngineOptions options;
    options.model_path = write_identity_model();
    options.backend_id = "ONNX_RUNTIME";
    options.plugin_dir = plugin_dir();

    auto engine = setup_inference_engine(options);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->state(), BackendState::Ready);

    const InferenceMetadata metadata = engine->get_inference_metadata();
    ASSERT_EQ(metadata.getInputs().size(), 1u);
    ASSERT_EQ(metadata.getOutputs().size(), 1u);
    EXPECT_EQ(metadata.getInputs()[0].name, "input");
    EXPECT_EQ(metadata.getOutputs()[0].name, "output");

    constexpr size_t element_count = 1 * 3 * 4 * 4;
    std::vector<float> values(element_count);
    for (size_t i = 0; i < element_count; ++i) {
        values[i] = static_cast<float>(i) * 0.25F;
    }
    std::vector<uint8_t> input_bytes(element_count * sizeof(float));
    std::memcpy(input_bytes.data(), values.data(), input_bytes.size());

    auto [outputs, shapes] = engine->get_infer_results({input_bytes});
    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_EQ(shapes.size(), 1u);
    EXPECT_EQ(shapes[0], (std::vector<int64_t>{1, 3, 4, 4}));
    ASSERT_EQ(outputs[0].size(), element_count);
    for (size_t i = 0; i < element_count; ++i) {
        EXPECT_FLOAT_EQ(std::get<float>(outputs[0][i]), values[i]) << "element " << i;
    }
}

TEST(PluginLoaderTest, UnknownBackendStillFailsCleanly) {
    EngineOptions options;
    options.model_path = "/nonexistent/model";
    options.backend_id = "NOT_A_BACKEND";
    options.plugin_dir = plugin_dir();
    EXPECT_EQ(setup_inference_engine(options), nullptr);
}
