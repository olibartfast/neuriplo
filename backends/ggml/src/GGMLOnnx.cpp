#include "GGMLOnnx.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ggml_onnx {
namespace {

// ---------------------------------------------------------------------------
// Minimal ONNX protobuf reader.
// ---------------------------------------------------------------------------

class Reader {
  public:
    Reader(const uint8_t* data, size_t size) : p_(data), end_(data + size) {}

    bool empty() const { return p_ == end_; }

    uint64_t varint() {
        uint64_t value = 0;
        for (int shift = 0; shift < 64; shift += 7) {
            require(1);
            const uint8_t byte = *p_++;
            value |= static_cast<uint64_t>(byte & 0x7f) << shift;
            if ((byte & 0x80) == 0) {
                return value;
            }
        }
        throw std::runtime_error("truncated varint");
    }

    uint32_t fixed32() {
        require(4);
        uint32_t value = 0;
        std::memcpy(&value, p_, sizeof(value));
        p_ += sizeof(value);
        return value;
    }

    uint64_t fixed64() {
        require(8);
        uint64_t value = 0;
        std::memcpy(&value, p_, sizeof(value));
        p_ += sizeof(value);
        return value;
    }

    Reader submessage() {
        const uint64_t length = varint();
        require(static_cast<size_t>(length));
        Reader reader(p_, static_cast<size_t>(length));
        p_ += length;
        return reader;
    }

    const uint8_t* begin() const { return p_; }
    const uint8_t* end() const { return end_; }

    std::string string_field() {
        Reader reader = submessage();
        return std::string(reinterpret_cast<const char*>(reader.p_), reader.remaining());
    }

    void skip(uint32_t wire) {
        switch (wire) {
        case 0:
            (void)varint();
            break;
        case 1:
            require(8);
            p_ += 8;
            break;
        case 2: {
            const uint64_t length = varint();
            require(static_cast<size_t>(length));
            p_ += length;
            break;
        }
        case 5:
            require(4);
            p_ += 4;
            break;
        default:
            throw std::runtime_error("unsupported protobuf wire type");
        }
    }

  private:
    size_t remaining() const { return static_cast<size_t>(end_ - p_); }

    void require(size_t count) const {
        if (count > remaining()) {
            throw std::runtime_error("truncated protobuf message");
        }
    }

    const uint8_t* p_ = nullptr;
    const uint8_t* end_ = nullptr;
};

struct TensorProto {
    std::string name;
    int32_t data_type = 0;
    std::vector<int64_t> dims;
    std::vector<float> float_data;
    std::vector<int32_t> int32_data;
    std::vector<int64_t> int64_data;
    std::vector<double> double_data;
    std::vector<uint8_t> raw_data;
    bool external = false;
};

struct ValueInfo {
    std::string name;
    int32_t elem_type = 0;
    std::vector<int64_t> dims;
    bool static_shape = false;
};

struct Attribute {
    std::string name;
    float f = 0.0f;
    int64_t i = 0;
    std::string s;
    std::vector<float> floats;
    std::vector<int64_t> ints;
    std::vector<std::string> strings;
    std::optional<TensorProto> tensor;
};

struct Node {
    std::string op_type;
    std::string name;
    std::string domain;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<Attribute> attributes;
};

struct GraphProto {
    std::vector<Node> nodes;
    std::vector<TensorProto> initializers;
    std::vector<ValueInfo> inputs;
    std::vector<ValueInfo> outputs;
    std::vector<ValueInfo> value_info;
};

struct ModelProto {
    GraphProto graph;
};

void read_int64s(Reader& reader, uint32_t wire, std::vector<int64_t>& values) {
    if (wire == 0) {
        values.push_back(static_cast<int64_t>(reader.varint()));
        return;
    }
    if (wire == 2) {
        Reader packed = reader.submessage();
        while (!packed.empty()) {
            values.push_back(static_cast<int64_t>(packed.varint()));
        }
        return;
    }
    throw std::runtime_error("bad wire type for int64 field");
}

void read_int32s(Reader& reader, uint32_t wire, std::vector<int32_t>& values) {
    if (wire == 0) {
        values.push_back(static_cast<int32_t>(reader.varint()));
        return;
    }
    if (wire == 2) {
        Reader packed = reader.submessage();
        while (!packed.empty()) {
            values.push_back(static_cast<int32_t>(packed.varint()));
        }
        return;
    }
    throw std::runtime_error("bad wire type for int32 field");
}

void read_floats(Reader& reader, uint32_t wire, std::vector<float>& values) {
    if (wire == 5) {
        const uint32_t bits = reader.fixed32();
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        values.push_back(value);
        return;
    }
    if (wire == 2) {
        Reader packed = reader.submessage();
        while (!packed.empty()) {
            const uint32_t bits = packed.fixed32();
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            values.push_back(value);
        }
        return;
    }
    throw std::runtime_error("bad wire type for float field");
}

void read_doubles(Reader& reader, uint32_t wire, std::vector<double>& values) {
    if (wire == 1) {
        const uint64_t bits = reader.fixed64();
        double value = 0.0;
        std::memcpy(&value, &bits, sizeof(value));
        values.push_back(value);
        return;
    }
    if (wire == 2) {
        Reader packed = reader.submessage();
        while (!packed.empty()) {
            const uint64_t bits = packed.fixed64();
            double value = 0.0;
            std::memcpy(&value, &bits, sizeof(value));
            values.push_back(value);
        }
        return;
    }
    throw std::runtime_error("bad wire type for double field");
}

TensorProto parse_tensor(Reader reader) {
    TensorProto tensor;
    while (!reader.empty()) {
        const uint64_t tag = reader.varint();
        const uint32_t field = static_cast<uint32_t>(tag >> 3);
        const uint32_t wire = static_cast<uint32_t>(tag & 0x7);
        switch (field) {
        case 1:
            read_int64s(reader, wire, tensor.dims);
            break;
        case 2:
            tensor.data_type = static_cast<int32_t>(reader.varint());
            break;
        case 4:
            read_floats(reader, wire, tensor.float_data);
            break;
        case 5:
            read_int32s(reader, wire, tensor.int32_data);
            break;
        case 7:
            read_int64s(reader, wire, tensor.int64_data);
            break;
        case 8:
            tensor.name = reader.string_field();
            break;
        case 9: {
            Reader raw = reader.submessage();
            tensor.raw_data.assign(raw.begin(), raw.end());
            break;
        }
        case 10:
            read_doubles(reader, wire, tensor.double_data);
            break;
        case 13:
            (void)reader.string_field();
            tensor.external = true;
            break;
        case 14:
            tensor.external = tensor.external || reader.varint() != 0;
            break;
        default:
            reader.skip(wire);
            break;
        }
    }
    return tensor;
}

void parse_tensor_type(Reader reader, ValueInfo& info) {
    while (!reader.empty()) {
        const uint64_t tag = reader.varint();
        const uint32_t field = static_cast<uint32_t>(tag >> 3);
        const uint32_t wire = static_cast<uint32_t>(tag & 0x7);
        switch (field) {
        case 1:
            info.elem_type = static_cast<int32_t>(reader.varint());
            break;
        case 2: {
            Reader shape = reader.submessage();
            info.static_shape = true;
            while (!shape.empty()) {
                const uint64_t dim_tag = shape.varint();
                const uint32_t dim_field = static_cast<uint32_t>(dim_tag >> 3);
                const uint32_t dim_wire = static_cast<uint32_t>(dim_tag & 0x7);
                if (dim_field != 1) {
                    shape.skip(dim_wire);
                    continue;
                }
                Reader dim = shape.submessage();
                bool has_value = false;
                bool symbolic = false;
                int64_t value = -1;
                while (!dim.empty()) {
                    const uint64_t value_tag = dim.varint();
                    const uint32_t value_field = static_cast<uint32_t>(value_tag >> 3);
                    const uint32_t value_wire = static_cast<uint32_t>(value_tag & 0x7);
                    if (value_field == 1) {
                        value = static_cast<int64_t>(dim.varint());
                        has_value = true;
                    } else if (value_field == 2) {
                        (void)dim.string_field();
                        symbolic = true;
                    } else {
                        dim.skip(value_wire);
                    }
                }
                if (!has_value || symbolic) {
                    info.static_shape = false;
                    value = -1;
                }
                info.dims.push_back(value);
            }
            break;
        }
        default:
            reader.skip(wire);
            break;
        }
    }
}

void parse_type(Reader reader, ValueInfo& info) {
    while (!reader.empty()) {
        const uint64_t tag = reader.varint();
        const uint32_t field = static_cast<uint32_t>(tag >> 3);
        const uint32_t wire = static_cast<uint32_t>(tag & 0x7);
        if (field == 1) {
            parse_tensor_type(reader.submessage(), info);
        } else {
            reader.skip(wire);
        }
    }
}

ValueInfo parse_value_info(Reader reader) {
    ValueInfo info;
    while (!reader.empty()) {
        const uint64_t tag = reader.varint();
        const uint32_t field = static_cast<uint32_t>(tag >> 3);
        const uint32_t wire = static_cast<uint32_t>(tag & 0x7);
        switch (field) {
        case 1:
            info.name = reader.string_field();
            break;
        case 2:
            parse_type(reader.submessage(), info);
            break;
        default:
            reader.skip(wire);
            break;
        }
    }
    return info;
}

Attribute parse_attribute(Reader reader) {
    Attribute attribute;
    while (!reader.empty()) {
        const uint64_t tag = reader.varint();
        const uint32_t field = static_cast<uint32_t>(tag >> 3);
        const uint32_t wire = static_cast<uint32_t>(tag & 0x7);
        switch (field) {
        case 1:
            attribute.name = reader.string_field();
            break;
        case 2: {
            const uint32_t bits = reader.fixed32();
            std::memcpy(&attribute.f, &bits, sizeof(attribute.f));
            break;
        }
        case 3:
            attribute.i = static_cast<int64_t>(reader.varint());
            break;
        case 4:
            attribute.s = reader.string_field();
            break;
        case 5:
            attribute.tensor = parse_tensor(reader.submessage());
            break;
        case 7:
            read_floats(reader, wire, attribute.floats);
            break;
        case 8:
            read_int64s(reader, wire, attribute.ints);
            break;
        case 9:
            attribute.strings.push_back(reader.string_field());
            break;
        default:
            reader.skip(wire);
            break;
        }
    }
    return attribute;
}

Node parse_node(Reader reader) {
    Node node;
    while (!reader.empty()) {
        const uint64_t tag = reader.varint();
        const uint32_t field = static_cast<uint32_t>(tag >> 3);
        const uint32_t wire = static_cast<uint32_t>(tag & 0x7);
        switch (field) {
        case 1:
            node.inputs.push_back(reader.string_field());
            break;
        case 2:
            node.outputs.push_back(reader.string_field());
            break;
        case 3:
            node.name = reader.string_field();
            break;
        case 4:
            node.op_type = reader.string_field();
            break;
        case 5:
            node.attributes.push_back(parse_attribute(reader.submessage()));
            break;
        case 7:
            node.domain = reader.string_field();
            break;
        default:
            reader.skip(wire);
            break;
        }
    }
    return node;
}

GraphProto parse_graph(Reader reader) {
    GraphProto graph;
    while (!reader.empty()) {
        const uint64_t tag = reader.varint();
        const uint32_t field = static_cast<uint32_t>(tag >> 3);
        const uint32_t wire = static_cast<uint32_t>(tag & 0x7);
        switch (field) {
        case 1:
            graph.nodes.push_back(parse_node(reader.submessage()));
            break;
        case 5:
            graph.initializers.push_back(parse_tensor(reader.submessage()));
            break;
        case 11:
            graph.inputs.push_back(parse_value_info(reader.submessage()));
            break;
        case 12:
            graph.outputs.push_back(parse_value_info(reader.submessage()));
            break;
        case 13:
            graph.value_info.push_back(parse_value_info(reader.submessage()));
            break;
        case 15:
            throw std::runtime_error("sparse initializers are not supported");
        default:
            reader.skip(wire);
            break;
        }
    }
    return graph;
}

ModelProto parse_model(const uint8_t* data, size_t size) {
    Reader reader(data, size);
    ModelProto model;
    while (!reader.empty()) {
        const uint64_t tag = reader.varint();
        const uint32_t field = static_cast<uint32_t>(tag >> 3);
        const uint32_t wire = static_cast<uint32_t>(tag & 0x7);
        if (field == 7) {
            model.graph = parse_graph(reader.submessage());
        } else {
            reader.skip(wire);
        }
    }
    return model;
}

// ---------------------------------------------------------------------------
// Tensor decoding.
// ---------------------------------------------------------------------------

size_t element_count(const std::vector<int64_t>& dims) {
    size_t count = 1;
    for (int64_t dim : dims) {
        if (dim <= 0) {
            throw std::invalid_argument("tensor has a non-static dimension");
        }
        if (static_cast<uint64_t>(dim) > SIZE_MAX / count) {
            throw std::invalid_argument("tensor element count overflow");
        }
        count *= static_cast<size_t>(dim);
    }
    return count;
}

float half_to_float(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000) << 16;
    const uint32_t exponent = (value >> 10) & 0x1f;
    uint32_t mantissa = value & 0x3ff;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x3ff;
            bits = sign | static_cast<uint32_t>(127 - 15 - shift) << 23 | mantissa << 13;
        }
    } else if (exponent == 0x1f) {
        bits = sign | 0x7f800000u | mantissa << 13;
    } else {
        bits = sign | (exponent + 127 - 15) << 23 | mantissa << 13;
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::vector<float> tensor_f32(const TensorProto& tensor) {
    const size_t count = element_count(tensor.dims);
    switch (tensor.data_type) {
    case 1: { // FLOAT
        if (tensor.float_data.size() == count) {
            return tensor.float_data;
        }
        if (tensor.raw_data.size() == count * sizeof(float)) {
            std::vector<float> values(count);
            std::memcpy(values.data(), tensor.raw_data.data(), tensor.raw_data.size());
            return values;
        }
        break;
    }
    case 11: { // DOUBLE
        std::vector<double> source = tensor.double_data;
        if (source.empty() && tensor.raw_data.size() == count * sizeof(double)) {
            source.resize(count);
            std::memcpy(source.data(), tensor.raw_data.data(), tensor.raw_data.size());
        }
        if (source.size() == count) {
            std::vector<float> values(count);
            std::transform(source.begin(), source.end(), values.begin(),
                           [](double value) { return static_cast<float>(value); });
            return values;
        }
        break;
    }
    case 10: { // FLOAT16
        if (tensor.raw_data.size() == count * sizeof(uint16_t)) {
            std::vector<float> values(count);
            for (size_t i = 0; i < count; ++i) {
                uint16_t half = 0;
                std::memcpy(&half, tensor.raw_data.data() + i * sizeof(half), sizeof(half));
                values[i] = half_to_float(half);
            }
            return values;
        }
        break;
    }
    default:
        break;
    }
    throw std::invalid_argument("tensor '" + tensor.name + "' is not decodable as FLOAT");
}

std::vector<int64_t> tensor_i64(const TensorProto& tensor) {
    const size_t count = element_count(tensor.dims);
    if (tensor.data_type != 7) {
        throw std::invalid_argument("tensor '" + tensor.name + "' is not INT64");
    }
    if (tensor.int64_data.size() == count) {
        return tensor.int64_data;
    }
    if (tensor.raw_data.size() == count * sizeof(int64_t)) {
        std::vector<int64_t> values(count);
        std::memcpy(values.data(), tensor.raw_data.data(), tensor.raw_data.size());
        return values;
    }
    throw std::invalid_argument("tensor '" + tensor.name + "' has no INT64 payload");
}

std::vector<int32_t> tensor_i32(const TensorProto& tensor) {
    const size_t count = element_count(tensor.dims);
    switch (tensor.data_type) {
    case 6: { // INT32
        if (tensor.int32_data.size() == count) {
            return tensor.int32_data;
        }
        if (tensor.raw_data.size() == count * sizeof(int32_t)) {
            std::vector<int32_t> values(count);
            std::memcpy(values.data(), tensor.raw_data.data(), tensor.raw_data.size());
            return values;
        }
        break;
    }
    case 7: { // INT64 narrowed through the graph's I32 constant path
        const std::vector<int64_t> values = tensor_i64(tensor);
        std::vector<int32_t> narrowed(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            if (values[i] < INT32_MIN || values[i] > INT32_MAX) {
                throw std::invalid_argument("INT64 tensor '" + tensor.name + "' does not fit INT32");
            }
            narrowed[i] = static_cast<int32_t>(values[i]);
        }
        return narrowed;
    }
    default:
        break;
    }
    throw std::invalid_argument("tensor '" + tensor.name + "' is not decodable as INT32");
}

// ---------------------------------------------------------------------------
// Graph emission.
// ---------------------------------------------------------------------------

struct Shape {
    int32_t elem_type = 0;
    std::vector<int64_t> dims; // ONNX order
};

const Attribute* find_attribute(const Node& node, const char* name) {
    for (const auto& attribute : node.attributes) {
        if (attribute.name == name) {
            return &attribute;
        }
    }
    return nullptr;
}

int64_t attribute_int(const Node& node, const char* name, int64_t fallback) {
    const Attribute* attribute = find_attribute(node, name);
    return attribute == nullptr ? fallback : attribute->i;
}

float attribute_float(const Node& node, const char* name, float fallback) {
    const Attribute* attribute = find_attribute(node, name);
    return attribute == nullptr ? fallback : attribute->f;
}

std::vector<int64_t> attribute_ints(const Node& node, const char* name) {
    const Attribute* attribute = find_attribute(node, name);
    return attribute == nullptr ? std::vector<int64_t>{} : attribute->ints;
}

bool is_float_type(int32_t elem_type) { return elem_type == 1 || elem_type == 10 || elem_type == 11; }

void require_supported_rank(const Shape& shape, const Node& node) {
    if (shape.dims.size() > 4) {
        throw std::invalid_argument("op '" + node.op_type + "' rank exceeds 4 at node '" + node.name + "'");
    }
}

ggml_graph::DType input_dtype(int32_t elem_type) {
    switch (elem_type) {
    case 1:
        return ggml_graph::DType::F32;
    case 6:
        return ggml_graph::DType::I32;
    case 7:
        return ggml_graph::DType::I64;
    default:
        throw std::invalid_argument("unsupported graph input element type " + std::to_string(elem_type));
    }
}

class Builder {
  public:
    explicit Builder(ModelProto model) : proto_(std::move(model)) {}

    std::unique_ptr<Model> build() {
        result_ = std::make_unique<Model>();
        for (const auto& initializer : proto_.graph.initializers) {
            constants_[initializer.name] = initializer;
            shapes_[initializer.name] = {initializer.data_type, initializer.dims};
            logical_dims_[initializer.name] = initializer.dims;
            axis_offsets_[initializer.name] = 0;
        }
        add_graph_inputs();
        for (const auto& value : proto_.graph.value_info) {
            if (shapes_.find(value.name) == shapes_.end() && value.static_shape) {
                shapes_[value.name] = {value.elem_type, value.dims};
            }
        }
        for (const auto& node : proto_.graph.nodes) {
            build_node(node);
        }
        add_graph_outputs();
        return std::move(result_);
    }

  private:
    void add_graph_inputs() {
        for (const auto& input : proto_.graph.inputs) {
            if (constants_.find(input.name) != constants_.end()) {
                continue;
            }
            if (!input.static_shape) {
                throw std::invalid_argument("graph input '" + input.name + "' has a dynamic shape");
            }
            const auto dims = ggml_dims(input.dims);
            const ggml_graph::DType dtype = input_dtype(input.elem_type);
            const int id = result_->graph.add_input(input.name, dtype, dims);
            ids_[input.name] = id;
            id_dtypes_[id] = dtype;
            shapes_[input.name] = {input.elem_type, input.dims};
            logical_dims_[input.name] = input.dims;
            axis_offsets_[input.name] = 0;
            result_->inputs.push_back({input.name, dtype, input.dims});
        }
    }

    void add_graph_outputs() {
        for (const auto& output : proto_.graph.outputs) {
            const int id = ensure(output.name);
            result_->graph.mark_output(id);
            result_->output_names.push_back(output.name);
        }
    }

    Shape shape_of(const std::string& name) const {
        const auto it = shapes_.find(name);
        if (it == shapes_.end()) {
            throw std::invalid_argument("no static shape for tensor '" + name + "'");
        }
        return it->second;
    }

    std::vector<int64_t> logical_dims_of(const std::string& name) const {
        const auto it = logical_dims_.find(name);
        if (it != logical_dims_.end()) {
            return it->second;
        }
        return shape_of(name).dims;
    }

    size_t axis_offset_of(const std::string& name) const {
        const auto it = axis_offsets_.find(name);
        return it == axis_offsets_.end() ? 0 : it->second;
    }

    void set_shape(const std::string& name, int32_t elem_type, std::vector<int64_t> logical_dims) {
        size_t dropped = 0;
        while (logical_dims.size() - dropped > 4 && logical_dims[dropped] == 1) {
            ++dropped;
        }
        if (logical_dims.size() - dropped > 4) {
            throw std::invalid_argument("tensor '" + name + "' rank exceeds 4 after dropping leading ones");
        }
        std::vector<int64_t> graph_dims;
        if (logical_dims.size() > dropped) {
            graph_dims.assign(logical_dims.begin() + static_cast<int64_t>(dropped), logical_dims.end());
        }
        if (graph_dims.empty() && !logical_dims.empty()) {
            graph_dims = {1};
        }
        shapes_[name] = {elem_type, graph_dims};
        logical_dims_[name] = std::move(logical_dims);
        axis_offsets_[name] = dropped;
    }

    std::vector<int64_t> ggml_dims(const std::vector<int64_t>& onnx_dims) const {
        if (onnx_dims.size() > 4) {
            throw std::invalid_argument("tensor rank " + std::to_string(onnx_dims.size()) + " exceeds 4");
        }
        if (onnx_dims.empty()) {
            return {1};
        }
        std::vector<int64_t> dims(onnx_dims.rbegin(), onnx_dims.rend());
        for (int64_t dim : dims) {
            if (dim <= 0) {
                throw std::invalid_argument("tensor has a non-static dimension");
            }
        }
        return dims;
    }

    int ensure(const std::string& name) {
        if (name.empty()) {
            throw std::invalid_argument("missing optional tensor used as a required input");
        }
        const auto existing = ids_.find(name);
        if (existing != ids_.end()) {
            return existing->second;
        }
        const auto constant = constants_.find(name);
        if (constant == constants_.end()) {
            throw std::invalid_argument("unknown tensor '" + name + "'");
        }
        return materialize(constant->second);
    }

    int materialize(const TensorProto& tensor) {
        if (tensor.external) {
            throw std::invalid_argument("external tensor data is not supported ('" + tensor.name + "')");
        }
        const std::vector<int64_t> dims = ggml_dims(tensor.dims);
        int id = -1;
        switch (tensor.data_type) {
        case 1:
        case 10:
        case 11:
            id = result_->graph.add_const_f32(tensor.name, dims, tensor_f32(tensor));
            id_dtypes_[id] = ggml_graph::DType::F32;
            break;
        case 6:
            id = result_->graph.add_const_i32(tensor.name, dims, tensor_i32(tensor));
            id_dtypes_[id] = ggml_graph::DType::I32;
            break;
        case 7:
            id = result_->graph.add_const_i32(tensor.name, dims, tensor_i32(tensor));
            id_dtypes_[id] = ggml_graph::DType::I32;
            break;
        default:
            throw std::invalid_argument("unsupported initializer element type " + std::to_string(tensor.data_type) +
                                        " ('" + tensor.name + "')");
        }
        ids_[tensor.name] = id;
        return id;
    }

    int scalar_f32(const std::string& name, float value) {
        const int id = result_->graph.add_const_f32(name + "#scalar", {1}, {value});
        id_dtypes_[id] = ggml_graph::DType::F32;
        return id;
    }

    int transpose_2d(int id) {
        return result_->graph.add_op(ggml_graph::Op::Transpose, {id}, {1.0f, 0.0f, 2.0f, 3.0f});
    }

    int reshape_ggml(int id, const std::vector<int64_t>& dims) {
        std::vector<float> attrs;
        attrs.reserve(dims.size());
        for (int64_t dim : dims) {
            attrs.push_back(static_cast<float>(dim));
        }
        return result_->graph.add_op(ggml_graph::Op::Reshape, {id}, attrs);
    }

    int to_f32(int id) {
        switch (result_->graph.dtype(id)) {
        case ggml_graph::DType::F32:
            return id;
        case ggml_graph::DType::I32: {
            const int cast = result_->graph.add_op(ggml_graph::Op::CastI32ToF32, {id});
            id_dtypes_[cast] = ggml_graph::DType::F32;
            return cast;
        }
        case ggml_graph::DType::I64: {
            const int narrowed = result_->graph.add_op(ggml_graph::Op::CastI64ToI32, {id});
            id_dtypes_[narrowed] = ggml_graph::DType::I32;
            const int cast = result_->graph.add_op(ggml_graph::Op::CastI32ToF32, {narrowed});
            id_dtypes_[cast] = ggml_graph::DType::F32;
            return cast;
        }
        }
        throw std::invalid_argument("unsupported dtype");
    }

    std::vector<float> ggml_axes_for_perm(const std::vector<int64_t>& perm) const {
        const int rank = static_cast<int>(perm.size());
        std::vector<int64_t> inverse(perm.size());
        for (size_t i = 0; i < perm.size(); ++i) {
            inverse[static_cast<size_t>(perm[i])] = static_cast<int64_t>(i);
        }
        std::vector<float> axes(4);
        for (int j = 0; j < 4; ++j) {
            if (j < rank) {
                axes[static_cast<size_t>(j)] =
                    static_cast<float>((rank - 1) - inverse[static_cast<size_t>(rank - 1 - j)]);
            } else {
                axes[static_cast<size_t>(j)] = static_cast<float>(j);
            }
        }
        return axes;
    }

    int transpose_onnx(int id, const std::vector<int64_t>& perm) {
        return result_->graph.add_op(ggml_graph::Op::Transpose, {id}, ggml_axes_for_perm(perm));
    }

    std::vector<int64_t> broadcast_dims(const std::vector<int64_t>& a, const std::vector<int64_t>& b) const {
        const size_t rank = std::max(a.size(), b.size());
        std::vector<int64_t> result(rank, 1);
        for (size_t i = 0; i < rank; ++i) {
            const int64_t left = i < a.size() ? a[a.size() - 1 - i] : 1;
            const int64_t right = i < b.size() ? b[b.size() - 1 - i] : 1;
            if (left == right || right == 1) {
                result[rank - 1 - i] = left;
            } else if (left == 1) {
                result[rank - 1 - i] = right;
            } else {
                throw std::invalid_argument("incompatible broadcast shapes");
            }
        }
        return result;
    }

    std::vector<int64_t> ggml_ne(const std::vector<int64_t>& onnx_dims) const {
        std::vector<int64_t> ne(4, 1);
        for (size_t i = 0; i < onnx_dims.size() && i < 4; ++i) {
            ne[i] = onnx_dims[onnx_dims.size() - 1 - i];
        }
        return ne;
    }

    bool can_repeat(const std::vector<int64_t>& target, const std::vector<int64_t>& source) const {
        const std::vector<int64_t> target_ne = ggml_ne(target);
        const std::vector<int64_t> source_ne = ggml_ne(source);
        for (size_t i = 0; i < 4; ++i) {
            if (source_ne[i] != 1 && source_ne[i] != target_ne[i]) {
                return false;
            }
        }
        return true;
    }

    int repeat_to(int id, const Shape& shape, const std::vector<int64_t>& target) {
        if (result_->graph.dtype(id) == ggml_graph::DType::I64) {
            id = result_->graph.add_op(ggml_graph::Op::CastI64ToI32, {id});
            id_dtypes_[id] = ggml_graph::DType::I32;
        }
        const size_t offset = target.size() - shape.dims.size();
        for (size_t i = 0; i < shape.dims.size(); ++i) {
            if (shape.dims[i] != 1 && target[offset + i] % shape.dims[i] != 0) {
                throw std::invalid_argument("cannot repeat tensor to the broadcast shape");
            }
        }
        std::vector<int64_t> dims = ggml_dims(target);
        std::vector<float> attrs(4, 1.0f);
        for (size_t i = 0; i < dims.size(); ++i) {
            attrs[i] = static_cast<float>(dims[i]);
        }
        return result_->graph.add_op(ggml_graph::Op::Repeat, {id}, attrs);
    }

    void record_output(const Node& node, size_t index, int id, int32_t elem_type, std::vector<int64_t> dims) {
        if (index >= node.outputs.size() || node.outputs[index].empty()) {
            throw std::invalid_argument("node '" + node.name + "' has no output " + std::to_string(index));
        }
        ids_[node.outputs[index]] = id;
        id_dtypes_[id] =
            elem_type == 1 || elem_type == 10 || elem_type == 11 ? ggml_graph::DType::F32 : ggml_graph::DType::I32;
        set_shape(node.outputs[index], elem_type, std::move(dims));
    }

    void record(const Node& node, int id, int32_t elem_type, std::vector<int64_t> dims) {
        if (node.outputs.empty() || node.outputs[0].empty()) {
            throw std::invalid_argument("node '" + node.name + "' has no primary output");
        }
        for (size_t i = 1; i < node.outputs.size(); ++i) {
            if (!node.outputs[i].empty()) {
                throw std::invalid_argument("multi-output ONNX op '" + node.op_type + "' is not supported");
            }
        }
        record_output(node, 0, id, elem_type, std::move(dims));
    }

    void require_count(const Node& node, size_t min_inputs, size_t max_inputs) const {
        if (node.inputs.size() < min_inputs || node.inputs.size() > max_inputs) {
            throw std::invalid_argument("op '" + node.op_type + "' has an unsupported input count at node '" +
                                        node.name + "'");
        }
    }

    void build_binary(const Node& node, ggml_graph::Op op, bool commutative) {
        require_count(node, 2, 2);
        const Shape left = shape_of(node.inputs[0]);
        const Shape right = shape_of(node.inputs[1]);
        if (op == ggml_graph::Op::Add || op == ggml_graph::Op::Sub || op == ggml_graph::Op::Mul ||
            op == ggml_graph::Op::Div) {
            const auto left_values = known_i64(node.inputs[0]);
            const auto right_values = known_i64(node.inputs[1]);
            if (left_values.has_value() && right_values.has_value()) {
                const size_t count = std::max(left_values->size(), right_values->size());
                if ((left_values->size() != 1 && left_values->size() != count) ||
                    (right_values->size() != 1 && right_values->size() != count)) {
                    throw std::invalid_argument("integer broadcast sizes do not match at node '" + node.name + "'");
                }
                std::vector<int64_t> values(count);
                for (size_t i = 0; i < count; ++i) {
                    const int64_t a = left_values->size() == 1 ? left_values->front() : (*left_values)[i];
                    const int64_t b = right_values->size() == 1 ? right_values->front() : (*right_values)[i];
                    if (op == ggml_graph::Op::Add) {
                        values[i] = a + b;
                    } else if (op == ggml_graph::Op::Sub) {
                        values[i] = a - b;
                    } else if (op == ggml_graph::Op::Mul) {
                        values[i] = a * b;
                    } else {
                        if (b == 0) {
                            throw std::invalid_argument("integer division by zero at node '" + node.name + "'");
                        }
                        values[i] = a / b;
                    }
                }
                register_i64(node.outputs.at(0), std::move(values), broadcast_dims(left.dims, right.dims));
                return;
            }
        }
        if (!is_float_type(left.elem_type) || !is_float_type(right.elem_type)) {
            if (!is_float_type(left.elem_type) && !is_float_type(right.elem_type) &&
                (op == ggml_graph::Op::Add || op == ggml_graph::Op::Sub || op == ggml_graph::Op::Mul ||
                 op == ggml_graph::Op::Div)) {
                const std::vector<int64_t> logical =
                    broadcast_dims(logical_dims_of(node.inputs[0]), logical_dims_of(node.inputs[1]));
                int a = ensure(node.inputs[0]);
                int b = ensure(node.inputs[1]);
                const int64_t left_count = static_cast<int64_t>(element_count(left.dims));
                const int64_t right_count = static_cast<int64_t>(element_count(right.dims));
                if (right_count != 1 && right_count != left_count) {
                    throw std::invalid_argument("integer broadcast is not supported at node '" + node.name + "'");
                }
                const int code = op == ggml_graph::Op::Add   ? 0
                                 : op == ggml_graph::Op::Sub ? 1
                                 : op == ggml_graph::Op::Mul ? 2
                                                             : 3;
                const int id = result_->graph.add_op(ggml_graph::Op::IntegerBinary, {a, b}, {static_cast<float>(code)});
                id_dtypes_[id] = ggml_graph::DType::I32;
                record(node, id, left.elem_type, logical);
                return;
            }
            throw std::invalid_argument("op '" + node.op_type + "' requires FLOAT or integer operands at node '" +
                                        node.name + "'");
        }
        const std::vector<int64_t> logical_dims =
            broadcast_dims(logical_dims_of(node.inputs[0]), logical_dims_of(node.inputs[1]));
        const std::vector<int64_t> dims = broadcast_dims(left.dims, right.dims);
        int a = ensure(node.inputs[0]);
        int b = ensure(node.inputs[1]);
        if (result_->graph.dtype(a) != ggml_graph::DType::F32 || result_->graph.dtype(b) != ggml_graph::DType::F32) {
            throw std::invalid_argument("float binary op has integer inputs at node '" + node.name + "'");
        }
        if (result_->graph.dtype(a) != result_->graph.dtype(b)) {
            throw std::invalid_argument("binary dtype mismatch at node '" + node.name + "'");
        }
        Shape a_shape = left;
        Shape b_shape = right;
        if (commutative && a_shape.dims != dims && b_shape.dims == dims) {
            std::swap(a, b);
            std::swap(a_shape, b_shape);
        }
        if (a_shape.dims != dims) {
            a = repeat_to(a, a_shape, dims);
        }
        if (b_shape.dims != dims) {
            if (!can_repeat(dims, b_shape.dims)) {
                throw std::invalid_argument("op '" + node.op_type + "' cannot broadcast its right operand");
            }
            b = repeat_to(b, b_shape, dims);
        }
        record(node, result_->graph.add_op(op, {a, b}, {4.0f}), 1, logical_dims);
    }

    void build_unary(const Node& node, ggml_graph::Op op) {
        require_count(node, 1, 1);
        record(node, result_->graph.add_op(op, {ensure(node.inputs[0])}), shape_of(node.inputs[0]).elem_type,
               logical_dims_of(node.inputs[0]));
    }

    int build_linear(const std::string& x_name, const std::string& weight_name, bool transpose_weight,
                     const std::string& bias_name, float alpha, int64_t bias_features) {
        int x = ensure(x_name);
        int weight = ensure(weight_name);
        if (transpose_weight) {
            weight = transpose_2d(weight);
        }
        int bias = -1;
        if (!bias_name.empty()) {
            bias = ensure(bias_name);
        } else {
            bias = result_->graph.add_const_f32(weight_name + "#zero_bias", {bias_features},
                                                std::vector<float>(static_cast<size_t>(bias_features), 0.0f));
        }
        int output = result_->graph.add_op(ggml_graph::Op::Linear, {x, weight, bias});
        if (alpha != 1.0f) {
            const int scale = scalar_f32(weight_name + "#alpha", alpha);
            if (result_->graph.dtype(output) != ggml_graph::DType::F32 ||
                result_->graph.dtype(scale) != ggml_graph::DType::F32) {
                throw std::invalid_argument("Linear alpha scale requires F32 at weight '" + weight_name + "'");
            }
            output = result_->graph.add_op(ggml_graph::Op::Mul, {output, scale}, {1.0f});
        }
        return output;
    }

    void build_matmul(const Node& node) {
        require_count(node, 2, 2);
        const Shape a = shape_of(node.inputs[0]);
        const Shape b = shape_of(node.inputs[1]);
        const bool a_vector = a.dims.size() == 1;
        const bool b_vector = b.dims.size() == 1;
        if (a_vector || b_vector) {
            if (a.dims.size() > 2 || b.dims.size() > 2) {
                throw std::invalid_argument("MatMul with rank-1 operands supports rank <= 2 only at node '" +
                                            node.name + "'");
            }
            const std::vector<int64_t> a_dims = a_vector ? std::vector<int64_t>{1, a.dims[0]} : a.dims;
            const std::vector<int64_t> b_dims = b_vector ? std::vector<int64_t>{b.dims[0], 1} : b.dims;
            const int64_t rows = a_dims[0];
            const int64_t inner = a_dims[1];
            const int64_t inner_b = b_dims[0];
            const int64_t columns = b_dims[1];
            if (inner != inner_b) {
                throw std::invalid_argument("MatMul inner dimensions do not match at node '" + node.name + "'");
            }
            int a_id = ensure(node.inputs[0]);
            int b_id = ensure(node.inputs[1]);
            if (a_vector) {
                a_id = reshape_ggml(a_id, {inner, 1});
            }
            if (b_vector) {
                b_id = reshape_ggml(b_id, {inner, 1});
            } else {
                b_id = transpose_2d(b_id);
            }
            std::vector<int64_t> dims;
            if (!a_vector) {
                dims.push_back(rows);
            }
            if (!b_vector) {
                dims.push_back(columns);
            }
            record(node, result_->graph.add_op(ggml_graph::Op::MatMul, {a_id, b_id}), 1, dims);
            return;
        }
        if (a.dims.size() < 2 || b.dims.size() < 2) {
            throw std::invalid_argument("MatMul operands must have rank >= 2 at node '" + node.name + "'");
        }
        const size_t rank = std::max(a.dims.size(), b.dims.size());
        std::vector<int64_t> a_dims(rank, 1);
        std::vector<int64_t> b_dims(rank, 1);
        std::copy(a.dims.begin(), a.dims.end(), a_dims.begin() + static_cast<int64_t>(rank - a.dims.size()));
        std::copy(b.dims.begin(), b.dims.end(), b_dims.begin() + static_cast<int64_t>(rank - b.dims.size()));
        const int64_t rows = a_dims[rank - 2];
        const int64_t inner = a_dims[rank - 1];
        const int64_t inner_b = b_dims[rank - 2];
        const int64_t columns = b_dims[rank - 1];
        if (inner != inner_b) {
            throw std::invalid_argument("MatMul inner dimensions do not match at node '" + node.name + "'");
        }
        std::vector<int64_t> dims;
        dims.reserve(rank);
        for (size_t i = 0; i + 2 < rank; ++i) {
            if (a_dims[i] != b_dims[i] && a_dims[i] != 1 && b_dims[i] != 1) {
                throw std::invalid_argument("MatMul batch dimensions do not match at node '" + node.name + "'");
            }
            dims.push_back(std::max(a_dims[i], b_dims[i]));
        }
        dims.push_back(rows);
        dims.push_back(columns);
        if (rank == 2) {
            const int id = build_linear(node.inputs[0], node.inputs[1], true, "", 1.0f, columns);
            record(node, id, 1, dims);
            return;
        }
        std::vector<int64_t> perm(b.dims.size());
        for (size_t i = 0; i < perm.size(); ++i) {
            perm[i] = static_cast<int64_t>(i);
        }
        std::swap(perm[perm.size() - 2], perm[perm.size() - 1]);
        const int weight = transpose_onnx(ensure(node.inputs[1]), perm);
        int64_t a_ne2 = rank >= 3 ? a_dims[rank - 3] : 1;
        int64_t a_ne3 = rank >= 4 ? a_dims[rank - 4] : 1;
        int64_t b_ne2 = rank >= 3 ? b_dims[rank - 3] : 1;
        int64_t b_ne3 = rank >= 4 ? b_dims[rank - 4] : 1;
        const bool b_over_a = a_ne2 % b_ne2 == 0 && a_ne3 % b_ne3 == 0;
        const bool a_over_b = b_ne2 % a_ne2 == 0 && b_ne3 % a_ne3 == 0;
        try {
            if (b_over_a) {
                record(node, result_->graph.add_op(ggml_graph::Op::BatchedMatMul, {weight, ensure(node.inputs[0])}), 1,
                       dims);
            } else if (a_over_b) {
                record(node,
                       result_->graph.add_op(ggml_graph::Op::BatchedMatMulTransposed, {ensure(node.inputs[0]), weight}),
                       1, dims);
            } else {
                throw std::invalid_argument("batch broadcast cannot be expressed in ggml");
            }
        } catch (const std::exception& error) {
            throw std::invalid_argument("MatMul '" + node.name + "': " + error.what());
        }
    }

    void build_gemm(const Node& node) {
        require_count(node, 2, 3);
        const Shape a = shape_of(node.inputs[0]);
        const Shape b = shape_of(node.inputs[1]);
        if (a.dims.size() != 2 || b.dims.size() != 2) {
            throw std::invalid_argument("Gemm supports rank-2 operands only");
        }
        const bool trans_a = attribute_int(node, "transA", 0) != 0;
        const bool trans_b = attribute_int(node, "transB", 0) != 0;
        const int64_t rows = trans_a ? a.dims[1] : a.dims[0];
        const int64_t inner = trans_a ? a.dims[0] : a.dims[1];
        const int64_t inner_b = trans_b ? b.dims[1] : b.dims[0];
        const int64_t columns = trans_b ? b.dims[0] : b.dims[1];
        if (inner != inner_b) {
            throw std::invalid_argument("Gemm inner dimensions do not match");
        }
        int x = ensure(node.inputs[0]);
        if (trans_a) {
            x = transpose_2d(x);
        }
        int weight = ensure(node.inputs[1]);
        if (!trans_b) {
            weight = transpose_2d(weight);
        }
        const float alpha = attribute_float(node, "alpha", 1.0f);
        const float beta = attribute_float(node, "beta", 1.0f);
        int bias = -1;
        int scaled_bias = -1;
        bool fused_bias = false;
        if (node.inputs.size() == 3 && !node.inputs[2].empty()) {
            scaled_bias = to_f32(ensure(node.inputs[2]));
            if (beta != 1.0f) {
                const int scale = scalar_f32(node.name + "#beta", beta);
                if (result_->graph.dtype(scaled_bias) != ggml_graph::DType::F32 ||
                    result_->graph.dtype(scale) != ggml_graph::DType::F32) {
                    throw std::invalid_argument("Gemm beta scale requires F32 at node '" + node.name + "'");
                }
                scaled_bias = result_->graph.add_op(ggml_graph::Op::Mul, {scaled_bias, scale}, {2.0f});
            }
            fused_bias = alpha == 1.0f;
        }
        bias = fused_bias ? scaled_bias
                          : result_->graph.add_const_f32(node.name + "#zero_bias", {columns},
                                                         std::vector<float>(static_cast<size_t>(columns), 0.0f));
        int output = result_->graph.add_op(ggml_graph::Op::Linear, {x, weight, bias});
        if (alpha != 1.0f) {
            const int scale = scalar_f32(node.name + "#alpha", alpha);
            if (result_->graph.dtype(output) != ggml_graph::DType::F32 ||
                result_->graph.dtype(scale) != ggml_graph::DType::F32) {
                throw std::invalid_argument("Gemm alpha scale requires F32 at node '" + node.name + "'");
            }
            output = result_->graph.add_op(ggml_graph::Op::Mul, {output, scale}, {3.0f});
        }
        if (scaled_bias >= 0 && !fused_bias) {
            output = result_->graph.add_op(ggml_graph::Op::Add, {output, scaled_bias});
        }
        record(node, output, 1, {rows, columns});
    }

    std::vector<int64_t> lookup_i64(const std::string& name) {
        if (const auto values = known_i64(name); values.has_value()) {
            return *values;
        }
        const auto constant = constants_.find(name);
        if (constant != constants_.end()) {
            if (constant->second.data_type == 6) {
                const std::vector<int32_t> values = tensor_i32(constant->second);
                return std::vector<int64_t>(values.begin(), values.end());
            }
            if (constant->second.data_type == 1 || constant->second.data_type == 10 ||
                constant->second.data_type == 11) {
                const std::vector<float> values = tensor_f32(constant->second);
                std::vector<int64_t> converted;
                converted.reserve(values.size());
                for (float value : values) {
                    if (value != std::floor(value)) {
                        throw std::invalid_argument("shape tensor '" + name + "' has a non-integral value");
                    }
                    converted.push_back(static_cast<int64_t>(value));
                }
                return converted;
            }
        }
        throw std::invalid_argument("tensor '" + name + "' is not a static shape value");
    }

    std::optional<std::vector<int64_t>> known_i64(const std::string& name) {
        const auto cached = i64_values_.find(name);
        if (cached != i64_values_.end()) {
            return cached->second;
        }
        const auto constant = constants_.find(name);
        if (constant == constants_.end() || constant->second.data_type != 7) {
            return std::nullopt;
        }
        const std::vector<int64_t> values = tensor_i64(constant->second);
        i64_values_[name] = values;
        return values;
    }

    void register_i64(const std::string& name, std::vector<int64_t> values, const std::vector<int64_t>& dims) {
        TensorProto tensor;
        tensor.name = name;
        tensor.data_type = 7;
        tensor.dims = dims;
        tensor.int64_data = values;
        i64_values_[name] = std::move(values);
        constants_[name] = std::move(tensor);
        set_shape(name, 7, dims);
    }

    void register_f32(const std::string& name, std::vector<float> values, const std::vector<int64_t>& dims) {
        TensorProto tensor;
        tensor.name = name;
        tensor.data_type = 1;
        tensor.dims = dims;
        tensor.float_data = std::move(values);
        constants_[name] = std::move(tensor);
        set_shape(name, 1, dims);
    }

    std::vector<int64_t> slice_indices(int64_t start, int64_t end, int64_t step, int64_t dim) {
        if (step == 0) {
            throw std::invalid_argument("Slice step must not be zero");
        }
        if (start < 0) {
            start += dim;
        }
        if (end < 0) {
            end += dim;
        }
        std::vector<int64_t> indices;
        if (step > 0) {
            start = std::max<int64_t>(0, std::min(start, dim));
            end = std::max<int64_t>(0, std::min(end, dim));
            for (int64_t index = start; index < end; index += step) {
                indices.push_back(index);
            }
        } else {
            start = std::max<int64_t>(-1, std::min(start, dim - 1));
            end = std::max<int64_t>(-1, std::min(end, dim - 1));
            for (int64_t index = start; index > end; index += step) {
                indices.push_back(index);
            }
        }
        return indices;
    }

    std::vector<int64_t> slice_values(const std::vector<int64_t>& values, const std::vector<int64_t>& dims,
                                      const std::vector<std::vector<int64_t>>& indices) {
        std::vector<int64_t> output_dims(indices.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            output_dims[i] = static_cast<int64_t>(indices[i].size());
            if (output_dims[i] == 0) {
                return {};
            }
        }
        const int64_t count = static_cast<int64_t>(element_count(output_dims));
        std::vector<int64_t> output;
        output.reserve(static_cast<size_t>(count));
        std::vector<int64_t> strides(dims.size(), 1);
        for (size_t i = dims.size(); i > 1; --i) {
            strides[i - 2] = strides[i - 1] * dims[i - 1];
        }
        for (int64_t flat = 0; flat < count; ++flat) {
            int64_t remainder = flat;
            int64_t offset = 0;
            for (size_t axis = dims.size(); axis > 0; --axis) {
                const size_t index = static_cast<size_t>(axis - 1);
                const int64_t coordinate = remainder % output_dims[index];
                remainder /= output_dims[index];
                offset += indices[index][static_cast<size_t>(coordinate)] * strides[index];
            }
            output.push_back(values[static_cast<size_t>(offset)]);
        }
        return output;
    }

    std::vector<int64_t> reshape_target(const Node& node, const Shape& input) {
        if (node.inputs.size() < 2) {
            throw std::invalid_argument("Reshape requires a shape input");
        }
        std::vector<int64_t> target = lookup_i64(node.inputs[1]);
        const bool allow_zero = attribute_int(node, "allowzero", 0) != 0;
        int64_t known = 1;
        int infer_index = -1;
        for (size_t i = 0; i < target.size(); ++i) {
            if (target[i] == 0 && !allow_zero) {
                if (i >= input.dims.size()) {
                    throw std::invalid_argument("Reshape zero index exceeds the input rank");
                }
                target[i] = input.dims[i];
            }
            if (target[i] == -1) {
                if (infer_index >= 0) {
                    throw std::invalid_argument("Reshape has more than one inferred dimension");
                }
                infer_index = static_cast<int>(i);
            } else if (target[i] > 0) {
                known *= target[i];
            } else {
                throw std::invalid_argument("Reshape has an unsupported target dimension");
            }
        }
        const int64_t input_elements = static_cast<int64_t>(element_count(input.dims));
        if (infer_index >= 0) {
            if (known <= 0 || input_elements % known != 0) {
                throw std::invalid_argument("Reshape cannot infer the target dimension");
            }
            target[static_cast<size_t>(infer_index)] = input_elements / known;
        }
        return target;
    }

    void build_reshape(const Node& node) {
        require_count(node, 1, 2);
        const Shape input = shape_of(node.inputs[0]);
        std::vector<int64_t> target;
        try {
            target = reshape_target(node, input);
        } catch (const std::exception& error) {
            throw std::invalid_argument("Reshape '" + node.name + "': " + error.what());
        }
        if (const auto values = known_i64(node.inputs[0]); values.has_value()) {
            register_i64(node.outputs.at(0), *values, target);
            return;
        }
        std::vector<int64_t> graph_target = target;
        while (graph_target.size() > 4 && graph_target.front() == 1) {
            graph_target.erase(graph_target.begin());
        }
        if (graph_target.size() > 4) {
            throw std::invalid_argument("Reshape graph rank exceeds 4 at node '" + node.name + "'");
        }
        const std::vector<int64_t> dims = ggml_dims(graph_target);
        std::vector<float> attrs;
        attrs.reserve(dims.size());
        for (int64_t dim : dims) {
            attrs.push_back(static_cast<float>(dim));
        }
        record(node, result_->graph.add_op(ggml_graph::Op::Reshape, {ensure(node.inputs[0])}, attrs), 1, target);
    }

    void build_transpose(const Node& node) {
        require_count(node, 1, 1);
        const Shape input = shape_of(node.inputs[0]);
        const std::vector<int64_t> logical = logical_dims_of(node.inputs[0]);
        const size_t offset = axis_offset_of(node.inputs[0]);
        std::vector<int64_t> perm = attribute_ints(node, "perm");
        if (perm.empty()) {
            for (size_t i = 0; i < logical.size(); ++i) {
                perm.push_back(static_cast<int64_t>(logical.size() - 1 - i));
            }
        }
        if (perm.size() != logical.size()) {
            throw std::invalid_argument("Transpose perm rank does not match the input");
        }
        std::vector<int64_t> output_logical(perm.size());
        for (size_t i = 0; i < perm.size(); ++i) {
            if (perm[i] < 0 || perm[i] >= static_cast<int64_t>(logical.size())) {
                throw std::invalid_argument("Transpose perm is out of range");
            }
            output_logical[i] = logical[static_cast<size_t>(perm[i])];
        }
        size_t output_offset = 0;
        while (output_logical.size() - output_offset > 4 && output_logical[output_offset] == 1) {
            ++output_offset;
        }
        if (output_logical.size() - output_offset > 4) {
            throw std::invalid_argument("Transpose output rank exceeds 4 at node '" + node.name + "'");
        }
        const size_t graph_rank = logical.size() - offset;
        const size_t output_graph_rank = output_logical.size() - output_offset;
        if (graph_rank != output_graph_rank) {
            throw std::invalid_argument("Transpose changes the graph rank at node '" + node.name + "'");
        }
        std::vector<int64_t> graph_perm(output_graph_rank);
        std::vector<bool> seen(graph_rank, false);
        for (size_t i = 0; i < output_graph_rank; ++i) {
            const int64_t source = perm[output_offset + i];
            if (source < static_cast<int64_t>(offset)) {
                throw std::invalid_argument("Transpose moves a dropped leading axis into the graph");
            }
            const size_t graph_axis = static_cast<size_t>(source) - offset;
            if (seen[graph_axis]) {
                throw std::invalid_argument("Transpose perm is not a permutation");
            }
            seen[graph_axis] = true;
            graph_perm[i] = static_cast<int64_t>(graph_axis);
        }
        if (const auto values = known_i64(node.inputs[0]); values.has_value()) {
            std::vector<int64_t> folded = transpose_values(*values, logical, perm);
            register_i64(node.outputs.at(0), std::move(folded), output_logical);
            return;
        }
        bool identity = true;
        for (size_t i = 0; i < graph_perm.size(); ++i) {
            identity = identity && graph_perm[i] == static_cast<int64_t>(i);
        }
        const int data = ensure(node.inputs[0]);
        const int id = identity ? data : transpose_onnx(data, graph_perm);
        record(node, id, input.elem_type, output_logical);
    }

    void build_concat(const Node& node) {
        require_count(node, 1, static_cast<size_t>(-1));
        const std::vector<int64_t> logical = logical_dims_of(node.inputs[0]);
        const size_t offset = axis_offset_of(node.inputs[0]);
        const int64_t rank = static_cast<int64_t>(logical.size());
        int64_t axis = attribute_int(node, "axis", 0);
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || axis >= rank) {
            throw std::invalid_argument("Concat axis is out of range");
        }
        if (static_cast<size_t>(axis) < offset) {
            throw std::invalid_argument("Concat on a dropped leading axis is not supported");
        }
        if (node.inputs.size() == 1) {
            if (const auto values = known_i64(node.inputs[0]); values.has_value()) {
                register_i64(node.outputs.at(0), *values, logical);
                return;
            }
            record(node, ensure(node.inputs[0]), shape_of(node.inputs[0]).elem_type, logical);
            return;
        }
        if (rank == 1) {
            std::vector<int64_t> folded;
            bool all_known = true;
            for (const auto& name : node.inputs) {
                const auto values = known_i64(name);
                if (!values.has_value()) {
                    all_known = false;
                    break;
                }
                folded.insert(folded.end(), values->begin(), values->end());
            }
            if (all_known) {
                const int64_t count = static_cast<int64_t>(folded.size());
                register_i64(node.outputs.at(0), std::move(folded), {count});
                return;
            }
        }
        std::vector<int64_t> dims = logical;
        dims[static_cast<size_t>(axis)] = 0;
        std::vector<int> inputs;
        for (const auto& name : node.inputs) {
            if (axis_offset_of(name) != offset) {
                throw std::invalid_argument("Concat input axis metadata does not match at node '" + node.name + "'");
            }
            const std::vector<int64_t> shape = logical_dims_of(name);
            if (shape.size() != logical.size()) {
                throw std::invalid_argument("Concat input ranks do not match at node '" + node.name + "' (input '" +
                                            name + "')");
            }
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i != static_cast<size_t>(axis) && shape[i] != logical[i]) {
                    throw std::invalid_argument("Concat input shapes do not match at node '" + node.name + "'");
                }
            }
            dims[static_cast<size_t>(axis)] += shape[static_cast<size_t>(axis)];
            inputs.push_back(ensure(name));
        }
        const size_t graph_rank = logical.size() - offset;
        const int ggml_axis = static_cast<int>(graph_rank - 1 - (static_cast<size_t>(axis) - offset));
        record(node, result_->graph.add_op(ggml_graph::Op::Concat, inputs, {static_cast<float>(ggml_axis)}), 1, dims);
    }

    void build_gather(const Node& node) {
        require_count(node, 2, 2);
        const Shape data = shape_of(node.inputs[0]);
        const Shape indices = shape_of(node.inputs[1]);
        int64_t axis = attribute_int(node, "axis", 0);
        if (axis < 0) {
            axis += static_cast<int64_t>(logical_dims_of(node.inputs[0]).size());
        }
        if (indices.dims.empty()) {
            const auto index_values = known_i64(node.inputs[1]);
            if (!index_values.has_value() || index_values->size() != 1) {
                throw std::invalid_argument("Gather with a scalar index requires a constant index at node '" +
                                            node.name + "'");
            }
            const std::vector<int64_t> logical = logical_dims_of(node.inputs[0]);
            const size_t offset = axis_offset_of(node.inputs[0]);
            if (axis < 0 || axis >= static_cast<int64_t>(logical.size())) {
                throw std::invalid_argument("Gather axis is out of range at node '" + node.name + "'");
            }
            int64_t index = index_values->front();
            if (index < 0) {
                index += logical[static_cast<size_t>(axis)];
            }
            if (index < 0 || index >= logical[static_cast<size_t>(axis)]) {
                throw std::invalid_argument("Gather index is out of range at node '" + node.name + "'");
            }
            std::vector<int64_t> output_logical = logical;
            output_logical.erase(output_logical.begin() + axis);
            if (const auto data_values = known_i64(node.inputs[0]); data_values.has_value()) {
                std::vector<std::vector<int64_t>> gather_indices(logical.size());
                for (size_t i = 0; i < logical.size(); ++i) {
                    for (int64_t value = 0; value < logical[i]; ++value) {
                        gather_indices[i].push_back(value);
                    }
                }
                gather_indices[static_cast<size_t>(axis)] = {index};
                std::vector<int64_t> folded = slice_values(*data_values, logical, gather_indices);
                register_i64(node.outputs.at(0), std::move(folded), output_logical);
                return;
            }
            if (static_cast<size_t>(axis) < offset) {
                if (index != 0) {
                    throw std::invalid_argument("Gather on a dropped leading axis must select index 0");
                }
                record(node, ensure(node.inputs[0]), data.elem_type, output_logical);
                return;
            }
            const size_t graph_axis = static_cast<size_t>(axis) - offset;
            std::vector<float> attrs(8, 0.0f);
            for (size_t ggml_axis = 0; ggml_axis < 4; ++ggml_axis) {
                attrs[2 * ggml_axis] = 0.0f;
                attrs[2 * ggml_axis + 1] = 1.0f;
                if (ggml_axis < data.dims.size()) {
                    const size_t onnx_axis = data.dims.size() - 1 - ggml_axis;
                    attrs[2 * ggml_axis] = onnx_axis == graph_axis ? static_cast<float>(index) : 0.0f;
                    attrs[2 * ggml_axis + 1] = static_cast<float>(onnx_axis == graph_axis ? 1 : data.dims[onnx_axis]);
                }
            }
            const int slice = result_->graph.add_op(ggml_graph::Op::Slice, {ensure(node.inputs[0])}, attrs);
            std::vector<int64_t> graph_output = data.dims;
            graph_output.erase(graph_output.begin() + static_cast<int64_t>(graph_axis));
            const int reshaped = reshape_ggml(slice, ggml_dims(graph_output));
            record(node, reshaped, data.elem_type, output_logical);
            return;
        }
        if (axis == 0 && data.dims.size() == 1) {
            const auto data_values = known_i64(node.inputs[0]);
            const auto index_values = known_i64(node.inputs[1]);
            if (data_values.has_value() && index_values.has_value()) {
                std::vector<int64_t> folded;
                folded.reserve(index_values->size());
                for (int64_t index : *index_values) {
                    if (index < 0) {
                        index += data.dims[0];
                    }
                    if (index < 0 || index >= data.dims[0]) {
                        throw std::invalid_argument("Gather index is out of range at node '" + node.name + "'");
                    }
                    folded.push_back((*data_values)[static_cast<size_t>(index)]);
                }
                register_i64(node.outputs.at(0), std::move(folded), indices.dims);
                return;
            }
        }
        if (axis != 0 || data.dims.size() != 2 || indices.dims.size() != 1) {
            throw std::invalid_argument("Gather supports rank-2 data with axis=0 and rank-1 indices at node '" +
                                        node.name + "'");
        }
        int index_id = ensure(node.inputs[1]);
        const ggml_graph::DType dtype = result_->graph.dtype(index_id);
        if (dtype == ggml_graph::DType::I64) {
            index_id = result_->graph.add_op(ggml_graph::Op::CastI64ToI32, {index_id});
            id_dtypes_[index_id] = ggml_graph::DType::I32;
        } else if (dtype != ggml_graph::DType::I32) {
            throw std::invalid_argument("Gather indices must be INT32 or INT64");
        }
        record(node, result_->graph.add_op(ggml_graph::Op::Gather, {ensure(node.inputs[0]), index_id}), 1,
               {indices.dims[0], data.dims[1]});
    }

    float constant_scalar_f32(const std::string& name) {
        const auto constant = constants_.find(name);
        if (constant == constants_.end()) {
            throw std::invalid_argument("tensor '" + name + "' is not a static FLOAT constant");
        }
        const std::vector<float> values = tensor_f32(constant->second);
        if (values.size() != 1) {
            throw std::invalid_argument("tensor '" + name + "' is not a scalar FLOAT constant");
        }
        return values[0];
    }

    void build_reduce(const Node& node, ggml_graph::Op op) {
        require_count(node, 1, 2);
        const Shape input = shape_of(node.inputs[0]);
        const std::vector<int64_t> logical = logical_dims_of(node.inputs[0]);
        const bool keepdims = attribute_int(node, "keepdims", 1) != 0;
        if (logical.empty()) {
            if (keepdims) {
                const int id = result_->graph.add_op(ggml_graph::Op::Reshape, {ensure(node.inputs[0])}, {1.0f});
                record(node, id, input.elem_type, {1});
            } else {
                record(node, ensure(node.inputs[0]), input.elem_type, {});
            }
            return;
        }
        std::vector<int64_t> axes = attribute_ints(node, "axes");
        if (axes.empty() && node.inputs.size() == 2 && !node.inputs[1].empty()) {
            axes = lookup_i64(node.inputs[1]);
        }
        if (axes.empty()) {
            for (size_t i = 0; i < logical.size(); ++i) {
                axes.push_back(static_cast<int64_t>(i));
            }
        }
        if (axes.size() != 1) {
            throw std::invalid_argument("reduction supports one axis only");
        }
        int64_t axis = axes[0];
        if (axis < 0) {
            axis += static_cast<int64_t>(logical.size());
        }
        if (axis != static_cast<int64_t>(logical.size()) - 1) {
            throw std::invalid_argument("reduction supports the last axis only");
        }
        std::vector<int64_t> dims = logical;
        if (keepdims) {
            dims.back() = 1;
        } else {
            dims.pop_back();
        }
        int id = result_->graph.add_op(op, {ensure(node.inputs[0])});
        if (!keepdims && !dims.empty()) {
            std::vector<int64_t> graph_dims = dims;
            while (graph_dims.size() > 4 && graph_dims.front() == 1) {
                graph_dims.erase(graph_dims.begin());
            }
            if (graph_dims.size() > 4) {
                throw std::invalid_argument("reduction graph rank exceeds 4 at node '" + node.name + "'");
            }
            const std::vector<int64_t> ggml = ggml_dims(graph_dims);
            std::vector<float> attrs;
            attrs.reserve(ggml.size());
            for (int64_t dim : ggml) {
                attrs.push_back(static_cast<float>(dim));
            }
            id = result_->graph.add_op(ggml_graph::Op::Reshape, {id}, attrs);
        }
        record(node, id, input.elem_type, dims);
    }

    void build_pow(const Node& node) {
        require_count(node, 2, 2);
        const float exponent = constant_scalar_f32(node.inputs[1]);
        const Shape input = shape_of(node.inputs[0]);
        if (exponent == 1.0f) {
            record(node, ensure(node.inputs[0]), input.elem_type, input.dims);
        } else if (exponent == 2.0f) {
            record(node, result_->graph.add_op(ggml_graph::Op::Square, {ensure(node.inputs[0])}), input.elem_type,
                   input.dims);
        } else if (exponent == 0.5f) {
            record(node, result_->graph.add_op(ggml_graph::Op::Sqrt, {ensure(node.inputs[0])}), input.elem_type,
                   input.dims);
        } else {
            throw std::invalid_argument("Pow supports exponents 1, 2 and 0.5 only");
        }
    }

    std::vector<int64_t> transpose_values(const std::vector<int64_t>& values, const std::vector<int64_t>& dims,
                                          const std::vector<int64_t>& perm) {
        std::vector<int64_t> output_dims(perm.size());
        for (size_t i = 0; i < perm.size(); ++i) {
            output_dims[i] = dims[static_cast<size_t>(perm[i])];
        }
        std::vector<int64_t> strides(dims.size(), 1);
        for (size_t i = dims.size(); i > 1; --i) {
            strides[i - 2] = strides[i - 1] * dims[i - 1];
        }
        const int64_t count = static_cast<int64_t>(element_count(output_dims));
        std::vector<int64_t> output;
        output.reserve(static_cast<size_t>(count));
        std::vector<int64_t> coordinates(dims.size(), 0);
        for (int64_t flat = 0; flat < count; ++flat) {
            int64_t remainder = flat;
            for (size_t axis = output_dims.size(); axis > 0; --axis) {
                const size_t index = axis - 1;
                coordinates[static_cast<size_t>(perm[index])] = remainder % output_dims[index];
                remainder /= output_dims[index];
            }
            int64_t offset = 0;
            for (size_t axis = 0; axis < dims.size(); ++axis) {
                offset += coordinates[axis] * strides[axis];
            }
            output.push_back(values[static_cast<size_t>(offset)]);
        }
        return output;
    }

    void build_split(const Node& node) {
        require_count(node, 1, 2);
        const Shape input = shape_of(node.inputs[0]);
        const std::vector<int64_t> logical = logical_dims_of(node.inputs[0]);
        const size_t offset = axis_offset_of(node.inputs[0]);
        int64_t axis = attribute_int(node, "axis", 0);
        if (axis < 0) {
            axis += static_cast<int64_t>(logical.size());
        }
        if (axis < 0 || axis >= static_cast<int64_t>(logical.size())) {
            throw std::invalid_argument("Split axis is out of range");
        }
        if (static_cast<size_t>(axis) < offset) {
            throw std::invalid_argument("Split on a dropped leading axis is not supported");
        }
        if (node.outputs.empty()) {
            throw std::invalid_argument("Split has no outputs");
        }
        std::vector<int64_t> sizes = attribute_ints(node, "split");
        if (sizes.empty() && node.inputs.size() == 2 && !node.inputs[1].empty()) {
            sizes = lookup_i64(node.inputs[1]);
        }
        if (sizes.empty()) {
            const int64_t count = static_cast<int64_t>(node.outputs.size());
            if (logical[static_cast<size_t>(axis)] % count != 0) {
                throw std::invalid_argument("Split cannot divide the axis evenly");
            }
            sizes.assign(static_cast<size_t>(count), logical[static_cast<size_t>(axis)] / count);
        }
        if (sizes.size() != node.outputs.size()) {
            throw std::invalid_argument("Split size count does not match the output count");
        }
        int64_t total = 0;
        for (int64_t size : sizes) {
            if (size <= 0) {
                throw std::invalid_argument("Split sizes must be positive");
            }
            total += size;
        }
        if (total != logical[static_cast<size_t>(axis)]) {
            throw std::invalid_argument("Split sizes do not cover the axis");
        }

        if (const auto values = known_i64(node.inputs[0]); values.has_value()) {
            if (logical.size() != 1) {
                throw std::invalid_argument("integer Split supports rank-1 values only");
            }
            size_t position = 0;
            for (size_t i = 0; i < sizes.size(); ++i) {
                std::vector<int64_t> part(values->begin() + static_cast<int64_t>(position),
                                          values->begin() + static_cast<int64_t>(position + sizes[i]));
                position += static_cast<size_t>(sizes[i]);
                register_i64(node.outputs[i], std::move(part), {sizes[i]});
            }
            return;
        }

        const size_t graph_axis = static_cast<size_t>(axis) - offset;
        const int data = ensure(node.inputs[0]);
        int64_t start = 0;
        for (size_t i = 0; i < sizes.size(); ++i) {
            std::vector<int64_t> dims = logical;
            dims[static_cast<size_t>(axis)] = sizes[i];
            std::vector<float> attrs(8, 0.0f);
            for (size_t ggml_axis = 0; ggml_axis < 4; ++ggml_axis) {
                attrs[2 * ggml_axis] = 0.0f;
                attrs[2 * ggml_axis + 1] = 1.0f;
                if (ggml_axis < input.dims.size()) {
                    const size_t onnx_axis = input.dims.size() - 1 - ggml_axis;
                    attrs[2 * ggml_axis] = onnx_axis == graph_axis ? static_cast<float>(start) : 0.0f;
                    attrs[2 * ggml_axis + 1] =
                        static_cast<float>(onnx_axis == graph_axis ? sizes[i] : input.dims[onnx_axis]);
                }
            }
            record_output(node, i, result_->graph.add_op(ggml_graph::Op::Slice, {data}, attrs), input.elem_type, dims);
            start += sizes[i];
        }
    }

    void build_topk(const Node& node) {
        require_count(node, 2, 2);
        const Shape input = shape_of(node.inputs[0]);
        const std::vector<int64_t> logical = logical_dims_of(node.inputs[0]);
        const std::vector<int64_t> k_values = lookup_i64(node.inputs[1]);
        if (k_values.size() != 1 || k_values[0] <= 0) {
            throw std::invalid_argument("TopK k must be a positive scalar");
        }
        int64_t axis = attribute_int(node, "axis", -1);
        if (axis < 0) {
            axis += static_cast<int64_t>(logical.size());
        }
        if (axis != static_cast<int64_t>(logical.size()) - 1) {
            throw std::invalid_argument("TopK supports the last axis only");
        }
        if (attribute_int(node, "largest", 1) == 0) {
            throw std::invalid_argument("TopK supports largest=1 only");
        }
        if (node.outputs.size() != 2) {
            throw std::invalid_argument("TopK requires two outputs");
        }
        const int64_t k = k_values[0];
        if (k > logical.back()) {
            throw std::invalid_argument("TopK k exceeds the axis size");
        }
        std::vector<int64_t> dims = logical;
        dims.back() = k;
        const int values =
            result_->graph.add_op(ggml_graph::Op::TopKValues, {ensure(node.inputs[0])}, {static_cast<float>(k)});
        const int indices =
            result_->graph.add_op(ggml_graph::Op::TopKIndices, {ensure(node.inputs[0])}, {static_cast<float>(k)});
        record_output(node, 0, values, input.elem_type, dims);
        record_output(node, 1, indices, 7, dims);
    }

    void build_expand(const Node& node) {
        require_count(node, 2, 2);
        const Shape input = shape_of(node.inputs[0]);
        const std::vector<int64_t> logical = logical_dims_of(node.inputs[0]);
        std::vector<int64_t> target = lookup_i64(node.inputs[1]);
        if (target.size() < logical.size()) {
            target.insert(target.begin(), logical.size() - target.size(), 1);
        }
        const std::vector<int64_t> dims = broadcast_dims(logical, target);
        std::vector<int64_t> graph_target = dims;
        while (graph_target.size() > 4 && graph_target.front() == 1) {
            graph_target.erase(graph_target.begin());
        }
        if (graph_target.size() > 4) {
            throw std::invalid_argument("Expand graph rank exceeds 4 at node '" + node.name + "'");
        }
        if (dims == logical) {
            record(node, ensure(node.inputs[0]), input.elem_type, logical);
            return;
        }
        int data = ensure(node.inputs[0]);
        if (result_->graph.dtype(data) == ggml_graph::DType::I64) {
            data = result_->graph.add_op(ggml_graph::Op::CastI64ToI32, {data});
            id_dtypes_[data] = ggml_graph::DType::I32;
        }
        const Shape graph_shape{input.elem_type, input.dims};
        record(node, repeat_to(data, graph_shape, graph_target), input.elem_type, dims);
    }

    void build_tile(const Node& node) {
        require_count(node, 2, 2);
        const Shape input = shape_of(node.inputs[0]);
        const std::vector<int64_t> logical = logical_dims_of(node.inputs[0]);
        const std::vector<int64_t> repeats = lookup_i64(node.inputs[1]);
        if (repeats.size() != logical.size()) {
            throw std::invalid_argument("Tile repeats rank does not match the input");
        }
        std::vector<int64_t> dims(logical.size());
        bool unchanged = true;
        for (size_t i = 0; i < dims.size(); ++i) {
            if (repeats[i] <= 0) {
                throw std::invalid_argument("Tile repeats must be positive");
            }
            dims[i] = logical[i] * repeats[i];
            unchanged = unchanged && repeats[i] == 1;
        }
        std::vector<int64_t> graph_target = dims;
        while (graph_target.size() > 4 && graph_target.front() == 1) {
            graph_target.erase(graph_target.begin());
        }
        if (graph_target.size() > 4) {
            throw std::invalid_argument("Tile graph rank exceeds 4 at node '" + node.name + "'");
        }
        if (unchanged) {
            record(node, ensure(node.inputs[0]), input.elem_type, logical);
            return;
        }
        int data = ensure(node.inputs[0]);
        if (result_->graph.dtype(data) == ggml_graph::DType::I64) {
            data = result_->graph.add_op(ggml_graph::Op::CastI64ToI32, {data});
            id_dtypes_[data] = ggml_graph::DType::I32;
        }
        const Shape graph_shape{input.elem_type, input.dims};
        record(node, repeat_to(data, graph_shape, graph_target), input.elem_type, dims);
    }

    void build_gather_elements(const Node& node) {
        require_count(node, 2, 2);
        const Shape data = shape_of(node.inputs[0]);
        const Shape indices = shape_of(node.inputs[1]);
        int64_t axis = attribute_int(node, "axis", 0);
        if (axis < 0) {
            axis += static_cast<int64_t>(data.dims.size());
        }
        if (data.dims.size() != 3 || axis != 1) {
            throw std::invalid_argument("GatherElements supports rank-3 input with axis=1 only");
        }
        if (data.dims.size() != indices.dims.size()) {
            throw std::invalid_argument("GatherElements input ranks do not match");
        }
        int index_id = ensure(node.inputs[1]);
        const ggml_graph::DType dtype = result_->graph.dtype(index_id);
        if (dtype == ggml_graph::DType::I64) {
            index_id = result_->graph.add_op(ggml_graph::Op::CastI64ToI32, {index_id});
            id_dtypes_[index_id] = ggml_graph::DType::I32;
        } else if (dtype != ggml_graph::DType::I32) {
            throw std::invalid_argument("GatherElements indices must be INT32 or INT64");
        }
        record(node, result_->graph.add_op(ggml_graph::Op::GatherElements, {ensure(node.inputs[0]), index_id}),
               data.elem_type, indices.dims);
    }

    void build_slice(const Node& node) {
        require_count(node, 3, 5);
        const Shape input = shape_of(node.inputs[0]);
        const std::vector<int64_t> logical = logical_dims_of(node.inputs[0]);
        const size_t offset = axis_offset_of(node.inputs[0]);
        if (input.dims.size() != logical.size() - offset) {
            throw std::invalid_argument("Slice axis metadata is inconsistent at node '" + node.name + "'");
        }
        const std::vector<int64_t> starts = lookup_i64(node.inputs[1]);
        const std::vector<int64_t> ends = lookup_i64(node.inputs[2]);
        std::vector<int64_t> axes;
        if (node.inputs.size() >= 4 && !node.inputs[3].empty()) {
            axes = lookup_i64(node.inputs[3]);
        } else {
            for (size_t i = 0; i < starts.size(); ++i) {
                axes.push_back(static_cast<int64_t>(i));
            }
        }
        std::vector<int64_t> steps;
        if (node.inputs.size() >= 5 && !node.inputs[4].empty()) {
            steps = lookup_i64(node.inputs[4]);
        } else {
            steps.assign(starts.size(), 1);
        }
        if (starts.size() != ends.size() || axes.size() != starts.size() || steps.size() != starts.size()) {
            throw std::invalid_argument("Slice parameter sizes do not match at node '" + node.name + "'");
        }

        std::set<int64_t> used_axes;
        std::vector<std::vector<int64_t>> indices(logical.size());
        for (size_t axis = 0; axis < logical.size(); ++axis) {
            for (int64_t index = 0; index < logical[axis]; ++index) {
                indices[axis].push_back(index);
            }
        }
        for (size_t i = 0; i < axes.size(); ++i) {
            int64_t axis = axes[i];
            if (axis < 0) {
                axis += static_cast<int64_t>(logical.size());
            }
            if (axis < 0 || axis >= static_cast<int64_t>(logical.size())) {
                throw std::invalid_argument("Slice axis is out of range at node '" + node.name + "'");
            }
            if (!used_axes.insert(axis).second) {
                throw std::invalid_argument("Slice axis is duplicated at node '" + node.name + "'");
            }
            std::vector<int64_t> selected =
                slice_indices(starts[i], ends[i], steps[i], logical[static_cast<size_t>(axis)]);
            if (static_cast<size_t>(axis) < offset) {
                if (selected != std::vector<int64_t>{0}) {
                    throw std::invalid_argument("Slice on a dropped leading axis must select the full axis");
                }
                indices[static_cast<size_t>(axis)] = std::move(selected);
                continue;
            }
            indices[static_cast<size_t>(axis)] = std::move(selected);
        }

        std::vector<int64_t> output_dims(indices.size());
        bool has_empty_axis = false;
        for (size_t axis = 0; axis < indices.size(); ++axis) {
            output_dims[axis] = static_cast<int64_t>(indices[axis].size());
            has_empty_axis = has_empty_axis || indices[axis].empty();
        }

        if (const auto values = known_i64(node.inputs[0]); values.has_value()) {
            std::vector<int64_t> folded = slice_values(*values, logical, indices);
            register_i64(node.outputs.at(0), std::move(folded), output_dims);
            return;
        }
        if (has_empty_axis) {
            throw std::invalid_argument("empty Slice is not supported at node '" + node.name + "'");
        }

        bool steps_are_one = true;
        for (size_t i = 0; i < axes.size(); ++i) {
            int64_t axis = axes[i];
            if (axis < 0) {
                axis += static_cast<int64_t>(logical.size());
            }
            if (static_cast<size_t>(axis) >= offset && steps[i] != 1) {
                steps_are_one = false;
            }
        }
        const int data_id = ensure(node.inputs[0]);
        if (!steps_are_one) {
            int64_t reverse_axis = -1;
            int64_t reverse_start = 0;
            for (size_t i = 0; i < axes.size(); ++i) {
                int64_t axis = axes[i];
                if (axis < 0) {
                    axis += static_cast<int64_t>(logical.size());
                }
                if (static_cast<size_t>(axis) >= offset && steps[i] != 1) {
                    if (reverse_axis >= 0) {
                        throw std::invalid_argument("Slice supports one reversed axis at node '" + node.name + "'");
                    }
                    reverse_axis = axis - static_cast<int64_t>(offset);
                    reverse_start = indices[static_cast<size_t>(axis)].front();
                }
            }
            const bool reverse_axis_zero =
                input.dims.size() == 2 && reverse_axis == 0 &&
                indices[static_cast<size_t>(reverse_axis + static_cast<int64_t>(offset))].size() ==
                    static_cast<size_t>(input.dims[0]) &&
                reverse_start == input.dims[0] - 1 &&
                indices[static_cast<size_t>(reverse_axis + static_cast<int64_t>(offset))].back() == 0;
            if (!reverse_axis_zero) {
                throw std::invalid_argument("Slice step != 1 is only supported for a full axis-0 reverse");
            }
            std::vector<int32_t> reversed(static_cast<size_t>(input.dims[0]));
            for (size_t i = 0; i < reversed.size(); ++i) {
                reversed[i] = static_cast<int32_t>(reversed.size() - 1 - i);
            }
            const int reverse_indices =
                result_->graph.add_const_i32(node.name + "#reverse_indices", {input.dims[0]}, reversed);
            record(node, result_->graph.add_op(ggml_graph::Op::Gather, {data_id, reverse_indices}), input.elem_type,
                   output_dims);
            return;
        }

        std::vector<int64_t> graph_starts(input.dims.size(), 0);
        std::vector<int64_t> graph_sizes = input.dims;
        for (size_t i = 0; i < axes.size(); ++i) {
            int64_t axis = axes[i];
            if (axis < 0) {
                axis += static_cast<int64_t>(logical.size());
            }
            if (static_cast<size_t>(axis) < offset) {
                continue;
            }
            const size_t graph_axis = static_cast<size_t>(axis) - offset;
            graph_starts[graph_axis] = indices[static_cast<size_t>(axis)].front();
            graph_sizes[graph_axis] = static_cast<int64_t>(indices[static_cast<size_t>(axis)].size());
        }
        if (graph_starts == std::vector<int64_t>(input.dims.size(), 0) && graph_sizes == input.dims) {
            record(node, data_id, input.elem_type, output_dims);
            return;
        }
        std::vector<float> attrs(8, 0.0f);
        for (size_t ggml_axis = 0; ggml_axis < 4; ++ggml_axis) {
            attrs[2 * ggml_axis] = 0.0f;
            attrs[2 * ggml_axis + 1] = 1.0f;
            if (ggml_axis < input.dims.size()) {
                const size_t onnx_axis = input.dims.size() - 1 - ggml_axis;
                attrs[2 * ggml_axis] = static_cast<float>(graph_starts[onnx_axis]);
                attrs[2 * ggml_axis + 1] = static_cast<float>(graph_sizes[onnx_axis]);
            }
        }
        record(node, result_->graph.add_op(ggml_graph::Op::Slice, {data_id}, attrs), input.elem_type, output_dims);
    }

    void build_softmax(const Node& node) {
        require_count(node, 1, 1);
        const Shape input = shape_of(node.inputs[0]);
        const std::vector<int64_t> logical = logical_dims_of(node.inputs[0]);
        int64_t axis = attribute_int(node, "axis", -1);
        if (axis < 0) {
            axis += static_cast<int64_t>(logical.size());
        }
        if (axis != static_cast<int64_t>(logical.size()) - 1) {
            throw std::invalid_argument("Softmax supports the last axis only");
        }
        record(node, result_->graph.add_op(ggml_graph::Op::Softmax, {ensure(node.inputs[0])}), input.elem_type,
               logical);
    }

    void build_layer_norm(const Node& node) {
        require_count(node, 2, 3);
        const Shape input = shape_of(node.inputs[0]);
        const std::vector<int64_t> logical = logical_dims_of(node.inputs[0]);
        int64_t axis = attribute_int(node, "axis", -1);
        if (axis < 0) {
            axis += static_cast<int64_t>(logical.size());
        }
        if (axis != static_cast<int64_t>(logical.size()) - 1) {
            throw std::invalid_argument("LayerNormalization supports the last axis only");
        }
        int bias = -1;
        if (node.inputs.size() == 3 && !node.inputs[2].empty()) {
            bias = ensure(node.inputs[2]);
        } else {
            bias = result_->graph.add_const_f32(node.name + "#zero_bias", {logical.back()},
                                                std::vector<float>(static_cast<size_t>(logical.back()), 0.0f));
        }
        const float epsilon = attribute_float(node, "epsilon", 1e-5f);
        record(node,
               result_->graph.add_op(ggml_graph::Op::LayerNorm, {ensure(node.inputs[0]), ensure(node.inputs[1]), bias},
                                     {epsilon}),
               input.elem_type, logical);
    }

    void build_cast(const Node& node) {
        require_count(node, 1, 1);
        const Shape input = shape_of(node.inputs[0]);
        const std::vector<int64_t> logical = logical_dims_of(node.inputs[0]);
        const int64_t target = attribute_int(node, "to", 0);
        if (const auto values = known_i64(node.inputs[0]); values.has_value() && (target == 6 || target == 7)) {
            register_i64(node.outputs.at(0), *values, logical);
            return;
        }
        if (const auto values = known_i64(node.inputs[0]); values.has_value() && target == 1) {
            std::vector<float> converted;
            converted.reserve(values->size());
            for (int64_t value : *values) {
                converted.push_back(static_cast<float>(value));
            }
            register_f32(node.outputs.at(0), std::move(converted), logical);
            record(node, ensure(node.outputs.at(0)), 1, logical);
            return;
        }
        const int id = ensure(node.inputs[0]);
        const ggml_graph::DType source = result_->graph.dtype(id);
        if (target == 6) { // INT32
            if (source == ggml_graph::DType::I64) {
                const int cast = result_->graph.add_op(ggml_graph::Op::CastI64ToI32, {id});
                id_dtypes_[cast] = ggml_graph::DType::I32;
                record(node, cast, 6, input.dims);
                return;
            }
            if (source == ggml_graph::DType::I32) {
                record(node, id, 6, input.dims);
                return;
            }
        } else if (target == 1 && source == ggml_graph::DType::F32) {
            record(node, id, 1, input.dims);
            return;
        } else if (target == 1 && source == ggml_graph::DType::I32) {
            int cast = -1;
            try {
                cast = result_->graph.add_op(ggml_graph::Op::CastI32ToF32, {id});
            } catch (const std::exception& error) {
                throw std::invalid_argument("Cast '" + node.name + "': " + error.what());
            }
            id_dtypes_[cast] = ggml_graph::DType::F32;
            record(node, cast, 1, input.dims);
            return;
        } else if (target == 1 && source == ggml_graph::DType::I64) {
            const int narrowed = result_->graph.add_op(ggml_graph::Op::CastI64ToI32, {id});
            id_dtypes_[narrowed] = ggml_graph::DType::I32;
            int cast = -1;
            try {
                cast = result_->graph.add_op(ggml_graph::Op::CastI32ToF32, {narrowed});
            } catch (const std::exception& error) {
                throw std::invalid_argument("Cast '" + node.name + "': " + error.what());
            }
            id_dtypes_[cast] = ggml_graph::DType::F32;
            record(node, cast, 1, input.dims);
            return;
        } else if (target == 7 && source == ggml_graph::DType::I32) {
            // Shape arithmetic is carried as I32 inside the graph; keep the
            // ONNX-visible dtype so downstream shape ops accept it.
            record(node, id, 7, input.dims);
            return;
        } else if (target == 7 && source == ggml_graph::DType::I64) {
            record(node, id, 7, input.dims);
            return;
        }
        throw std::invalid_argument("Cast to element type " + std::to_string(target) + " from dtype " +
                                    std::to_string(static_cast<int>(source)) + " is not supported at node '" +
                                    node.name + "'");
    }

    void build_squeeze(const Node& node, bool unsqueeze) {
        require_count(node, 1, 2);
        const std::vector<int64_t> logical = logical_dims_of(node.inputs[0]);
        const Shape input = shape_of(node.inputs[0]);
        std::vector<int64_t> axes = attribute_ints(node, "axes");
        if (axes.empty() && node.inputs.size() == 2 && !node.inputs[1].empty()) {
            axes = lookup_i64(node.inputs[1]);
        }
        std::vector<int64_t> dims;
        if (unsqueeze) {
            const int64_t output_rank = static_cast<int64_t>(logical.size() + axes.size());
            if (axes.empty()) {
                throw std::invalid_argument("Unsqueeze requires axes");
            }
            std::set<int64_t> normalized;
            for (int64_t axis : axes) {
                if (axis < 0) {
                    axis += output_rank;
                }
                if (axis < 0 || axis >= output_rank || !normalized.insert(axis).second) {
                    throw std::invalid_argument("Unsqueeze axes are out of range or duplicated");
                }
            }
            dims.reserve(static_cast<size_t>(output_rank));
            size_t source = 0;
            for (int64_t i = 0; i < output_rank; ++i) {
                if (normalized.count(i) != 0) {
                    dims.push_back(1);
                } else {
                    dims.push_back(logical[source++]);
                }
            }
        } else {
            std::set<int64_t> normalized;
            for (int64_t axis : axes) {
                if (axis < 0) {
                    axis += static_cast<int64_t>(logical.size());
                }
                if (axis < 0 || axis >= static_cast<int64_t>(logical.size()) ||
                    logical[static_cast<size_t>(axis)] != 1) {
                    throw std::invalid_argument("Squeeze axis is out of range or not one");
                }
                normalized.insert(axis);
            }
            for (size_t i = 0; i < logical.size(); ++i) {
                if (normalized.empty() ? logical[i] != 1 : normalized.count(static_cast<int64_t>(i)) == 0) {
                    dims.push_back(logical[i]);
                }
            }
        }
        if (const auto values = known_i64(node.inputs[0]); values.has_value()) {
            register_i64(node.outputs.at(0), *values, dims);
            return;
        }
        std::vector<int64_t> graph_dims = dims;
        while (graph_dims.size() > 4 && graph_dims.front() == 1) {
            graph_dims.erase(graph_dims.begin());
        }
        if (graph_dims.size() > 4) {
            throw std::invalid_argument("Squeeze/Unsqueeze graph rank exceeds 4 at node '" + node.name + "'");
        }
        const std::vector<int64_t> ggml = ggml_dims(graph_dims);
        std::vector<float> attrs;
        attrs.reserve(ggml.size());
        for (int64_t dim : ggml) {
            attrs.push_back(static_cast<float>(dim));
        }
        record(node, result_->graph.add_op(ggml_graph::Op::Reshape, {ensure(node.inputs[0])}, attrs), input.elem_type,
               dims);
    }

    void build_flatten(const Node& node) {
        require_count(node, 1, 1);
        const Shape input = shape_of(node.inputs[0]);
        int64_t axis = attribute_int(node, "axis", 1);
        if (axis < 0) {
            axis += static_cast<int64_t>(input.dims.size());
        }
        if (axis < 0 || axis > static_cast<int64_t>(input.dims.size())) {
            throw std::invalid_argument("Flatten axis is out of range");
        }
        int64_t before = 1;
        int64_t after = 1;
        for (size_t i = 0; i < input.dims.size(); ++i) {
            if (static_cast<int64_t>(i) < axis) {
                before *= input.dims[i];
            } else {
                after *= input.dims[i];
            }
        }
        const std::vector<int64_t> dims = {before, after};
        const std::vector<int64_t> ggml = ggml_dims(dims);
        std::vector<float> attrs;
        attrs.reserve(ggml.size());
        for (int64_t dim : ggml) {
            attrs.push_back(static_cast<float>(dim));
        }
        record(node, result_->graph.add_op(ggml_graph::Op::Reshape, {ensure(node.inputs[0])}, attrs), 1, dims);
    }

    void build_conv(const Node& node) {
        require_count(node, 2, 3);
        const Shape input = shape_of(node.inputs[0]);
        const Shape weight = shape_of(node.inputs[1]);
        if (input.dims.size() != 4 || weight.dims.size() != 4) {
            throw std::invalid_argument("Conv supports rank-4 NCHW input and weight only");
        }
        const std::string auto_pad =
            find_attribute(node, "auto_pad") == nullptr ? "NOTSET" : find_attribute(node, "auto_pad")->s;
        if (auto_pad != "NOTSET") {
            throw std::invalid_argument("Conv auto_pad is not supported");
        }
        std::vector<int64_t> kernel = attribute_ints(node, "kernel_shape");
        if (kernel.size() != 2) {
            kernel = {weight.dims[2], weight.dims[3]};
        }
        std::vector<int64_t> strides = attribute_ints(node, "strides");
        if (strides.empty()) {
            strides = {1, 1};
        }
        std::vector<int64_t> pads = attribute_ints(node, "pads");
        if (pads.empty()) {
            pads = {0, 0, 0, 0};
        }
        std::vector<int64_t> dilations = attribute_ints(node, "dilations");
        if (dilations.empty()) {
            dilations = {1, 1};
        }
        if (strides.size() != 2 || pads.size() != 4 || dilations.size() != 2 || kernel.size() != 2) {
            throw std::invalid_argument("Conv attribute rank is invalid");
        }
        if (pads[0] != pads[2] || pads[1] != pads[3]) {
            throw std::invalid_argument("Conv supports symmetric padding only");
        }
        if (kernel[0] != weight.dims[2] || kernel[1] != weight.dims[3]) {
            throw std::invalid_argument("Conv kernel_shape does not match the weight");
        }
        const int64_t group = attribute_int(node, "group", 1);
        const int64_t channels = input.dims[1];
        const int64_t out_channels = weight.dims[0];
        if (weight.dims[1] * group != channels || out_channels % group != 0) {
            throw std::invalid_argument("Conv group/channel dimensions do not match");
        }
        const bool depthwise = group == channels && weight.dims[1] == 1;
        if (!depthwise && group != 1) {
            throw std::invalid_argument("Conv supports group=1 or depthwise convolutions only");
        }
        const int64_t out_h = (input.dims[2] + pads[0] + pads[2] - dilations[0] * (kernel[0] - 1) - 1) / strides[0] + 1;
        const int64_t out_w = (input.dims[3] + pads[1] + pads[3] - dilations[1] * (kernel[1] - 1) - 1) / strides[1] + 1;
        if (out_h <= 0 || out_w <= 0) {
            throw std::invalid_argument("Conv output shape is empty");
        }
        const int x = ensure(node.inputs[0]);
        const int w = ensure(node.inputs[1]);
        int bias = -1;
        if (node.inputs.size() == 3 && !node.inputs[2].empty()) {
            bias = ensure(node.inputs[2]);
        } else {
            bias = result_->graph.add_const_f32(node.name + "#zero_bias", {out_channels},
                                                std::vector<float>(static_cast<size_t>(out_channels), 0.0f));
        }
        const auto op = depthwise ? ggml_graph::Op::Conv2dDepthwise : ggml_graph::Op::Conv2d;
        const std::vector<float> attrs = {static_cast<float>(strides[1]),   static_cast<float>(strides[0]),
                                          static_cast<float>(pads[1]),      static_cast<float>(pads[0]),
                                          static_cast<float>(dilations[1]), static_cast<float>(dilations[0])};
        record(node, result_->graph.add_op(op, {x, w, bias}, attrs), 1, {input.dims[0], out_channels, out_h, out_w});
    }

    void build_max_pool(const Node& node) {
        require_count(node, 1, 1);
        const Shape input = shape_of(node.inputs[0]);
        if (input.dims.size() != 4) {
            throw std::invalid_argument("MaxPool supports rank-4 NCHW input only");
        }
        std::vector<int64_t> kernel = attribute_ints(node, "kernel_shape");
        std::vector<int64_t> strides = attribute_ints(node, "strides");
        std::vector<int64_t> pads = attribute_ints(node, "pads");
        if (strides.empty()) {
            strides = {1, 1};
        }
        if (pads.empty()) {
            pads = {0, 0, 0, 0};
        }
        if (kernel.size() != 2 || strides.size() != 2 || pads.size() != 4) {
            throw std::invalid_argument("MaxPool attribute rank is invalid");
        }
        if (pads[0] != pads[2] || pads[1] != pads[3]) {
            throw std::invalid_argument("MaxPool supports symmetric padding only");
        }
        const int64_t out_h = (input.dims[2] + pads[0] + pads[2] - kernel[0]) / strides[0] + 1;
        const int64_t out_w = (input.dims[3] + pads[1] + pads[3] - kernel[1]) / strides[1] + 1;
        if (out_h <= 0 || out_w <= 0) {
            throw std::invalid_argument("MaxPool output shape is empty");
        }
        if (attribute_int(node, "ceil_mode", 0) != 0) {
            const int64_t ceil_h = (input.dims[2] + pads[0] + pads[2] - kernel[0] + strides[0] - 1) / strides[0] + 1;
            const int64_t ceil_w = (input.dims[3] + pads[1] + pads[3] - kernel[1] + strides[1] - 1) / strides[1] + 1;
            if (ceil_h != out_h || ceil_w != out_w) {
                throw std::invalid_argument("MaxPool ceil_mode changes the output shape; not supported");
            }
        }
        const std::vector<float> attrs = {static_cast<float>(kernel[1]),  static_cast<float>(kernel[0]),
                                          static_cast<float>(strides[1]), static_cast<float>(strides[0]),
                                          static_cast<float>(pads[1]),    static_cast<float>(pads[0])};
        record(node, result_->graph.add_op(ggml_graph::Op::MaxPool2d, {ensure(node.inputs[0])}, attrs), 1,
               {input.dims[0], input.dims[1], out_h, out_w});
    }

    void build_constant_of_shape(const Node& node) {
        require_count(node, 1, 1);
        const std::vector<int64_t> dims = lookup_i64(node.inputs[0]);
        TensorProto scalar;
        const Attribute* value = find_attribute(node, "value");
        if (value != nullptr && value->tensor.has_value()) {
            scalar = *value->tensor;
        } else {
            scalar.data_type = 1;
            scalar.float_data = {0.0f};
        }
        const size_t count = element_count(dims);
        TensorProto tensor;
        tensor.dims = dims;
        tensor.name = node.outputs.at(0);
        tensor.data_type = scalar.data_type;
        switch (scalar.data_type) {
        case 1:
        case 10:
        case 11:
            tensor.float_data.assign(count, tensor_f32(scalar)[0]);
            tensor.data_type = 1;
            break;
        case 6:
            tensor.int32_data.assign(count, tensor_i32(scalar)[0]);
            break;
        case 7:
            tensor.int64_data.assign(count, tensor_i64(scalar)[0]);
            i64_values_[tensor.name] = tensor.int64_data;
            break;
        default:
            throw std::invalid_argument("ConstantOfShape value type is not supported");
        }
        constants_[tensor.name] = tensor;
        set_shape(tensor.name, tensor.data_type, dims);
    }

    void build_grid_sample(const Node& node) {
        require_count(node, 2, 2);
        const Shape input = shape_of(node.inputs[0]);
        const Shape grid = shape_of(node.inputs[1]);
        if (input.dims.size() != 4 || grid.dims.size() != 4) {
            throw std::invalid_argument("GridSample supports rank-4 NCHW input and grid");
        }
        if (grid.dims[0] != input.dims[0] || grid.dims[3] != 2) {
            throw std::invalid_argument("GridSample grid shape does not match the input");
        }
        const std::string mode = find_attribute(node, "mode") == nullptr ? "bilinear" : find_attribute(node, "mode")->s;
        const std::string padding =
            find_attribute(node, "padding_mode") == nullptr ? "zeros" : find_attribute(node, "padding_mode")->s;
        if ((mode != "bilinear" && mode != "linear") || padding != "zeros") {
            throw std::invalid_argument("GridSample supports bilinear/linear mode with zero padding only");
        }
        const int64_t align_corners = attribute_int(node, "align_corners", 0);
        if (align_corners != 0 && align_corners != 1) {
            throw std::invalid_argument("GridSample align_corners must be 0 or 1");
        }
        record(node,
               result_->graph.add_op(ggml_graph::Op::GridSample, {ensure(node.inputs[0]), ensure(node.inputs[1])},
                                     {static_cast<float>(align_corners)}),
               input.elem_type, {input.dims[0], input.dims[1], grid.dims[1], grid.dims[2]});
    }

    void build_pad(const Node& node) {
        require_count(node, 2, 3);
        const Shape input = shape_of(node.inputs[0]);
        const std::vector<int64_t> logical = logical_dims_of(node.inputs[0]);
        const size_t offset = axis_offset_of(node.inputs[0]);
        const std::string mode = find_attribute(node, "mode") == nullptr ? "constant" : find_attribute(node, "mode")->s;
        if (mode != "constant") {
            throw std::invalid_argument("Pad supports constant mode only");
        }
        if (node.inputs.size() == 3 && !node.inputs[2].empty()) {
            if (constant_scalar_f32(node.inputs[2]) != 0.0f) {
                throw std::invalid_argument("Pad supports a zero constant value only");
            }
        }
        const std::vector<int64_t> pads = lookup_i64(node.inputs[1]);
        const size_t rank = logical.size();
        if (pads.size() != 2 * rank) {
            throw std::invalid_argument("Pad pads size does not match the input rank");
        }
        std::vector<int64_t> begin(rank, 0);
        std::vector<int64_t> end(rank, 0);
        std::vector<int64_t> output_dims = logical;
        for (size_t axis = 0; axis < rank; ++axis) {
            begin[axis] = pads[axis];
            end[axis] = pads[rank + axis];
            if (begin[axis] < 0 || end[axis] < 0) {
                throw std::invalid_argument("Pad cropping is not supported");
            }
            if (begin[axis] != 0) {
                throw std::invalid_argument("Pad supports begin padding of zero only");
            }
            if (axis < offset && end[axis] != 0) {
                throw std::invalid_argument("Pad on a dropped leading axis is not supported");
            }
            output_dims[axis] += end[axis];
        }
        bool no_padding = true;
        for (size_t axis = 0; axis < rank; ++axis) {
            no_padding = no_padding && begin[axis] == 0 && end[axis] == 0;
        }
        const int data = ensure(node.inputs[0]);
        if (no_padding) {
            record(node, data, input.elem_type, logical);
            return;
        }
        const size_t graph_rank = logical.size() - offset;
        std::vector<float> attrs(4, 0.0f);
        for (size_t ggml_axis = 0; ggml_axis < 4; ++ggml_axis) {
            if (ggml_axis < graph_rank) {
                const size_t graph_axis = graph_rank - 1 - ggml_axis;
                attrs[ggml_axis] = static_cast<float>(end[offset + graph_axis]);
            }
        }
        record(node, result_->graph.add_op(ggml_graph::Op::Pad, {data}, attrs), input.elem_type, output_dims);
    }

    std::vector<float> constant_f32_values(const std::string& name) {
        const auto constant = constants_.find(name);
        if (constant == constants_.end()) {
            throw std::invalid_argument("tensor '" + name + "' is not a static FLOAT constant");
        }
        return tensor_f32(constant->second);
    }

    void build_resize(const Node& node) {
        require_count(node, 3, 4);
        const Shape input = shape_of(node.inputs[0]);
        const std::vector<int64_t> logical = logical_dims_of(node.inputs[0]);
        const std::string mode = find_attribute(node, "mode") == nullptr ? "nearest" : find_attribute(node, "mode")->s;
        const std::string coordinate = find_attribute(node, "coordinate_transformation_mode") == nullptr
                                           ? "half_pixel"
                                           : find_attribute(node, "coordinate_transformation_mode")->s;
        const std::string nearest = find_attribute(node, "nearest_mode") == nullptr
                                        ? "round_prefer_floor"
                                        : find_attribute(node, "nearest_mode")->s;
        if (mode != "nearest" || coordinate != "asymmetric" || nearest != "floor") {
            throw std::invalid_argument("Resize supports nearest/asymmetric/floor only");
        }
        std::vector<int64_t> target = logical;
        if (node.inputs.size() >= 4 && !node.inputs[3].empty()) {
            target = lookup_i64(node.inputs[3]);
        } else if (node.inputs.size() >= 3 && !node.inputs[2].empty()) {
            const std::vector<float> scales = constant_f32_values(node.inputs[2]);
            if (scales.size() != logical.size()) {
                throw std::invalid_argument("Resize scales rank does not match the input");
            }
            for (size_t i = 0; i < scales.size(); ++i) {
                target[i] = static_cast<int64_t>(std::floor(static_cast<float>(logical[i]) * scales[i]));
            }
        } else {
            throw std::invalid_argument("Resize requires scales or sizes");
        }
        if (target.size() != logical.size()) {
            throw std::invalid_argument("Resize target rank does not match the input");
        }
        bool unchanged = true;
        for (size_t i = 0; i < target.size(); ++i) {
            if (target[i] <= 0 || target[i] > 4 * logical[i]) {
                throw std::invalid_argument("Resize target dimension is out of range");
            }
            unchanged = unchanged && target[i] == logical[i];
        }
        if (unchanged) {
            record(node, ensure(node.inputs[0]), input.elem_type, logical);
            return;
        }
        std::vector<int64_t> graph_target = target;
        while (graph_target.size() > 4 && graph_target.front() == 1) {
            graph_target.erase(graph_target.begin());
        }
        if (graph_target.size() > 4) {
            throw std::invalid_argument("Resize graph rank exceeds 4 at node '" + node.name + "'");
        }
        const std::vector<int64_t> ggml = ggml_dims(graph_target);
        std::vector<float> attrs(4, 1.0f);
        for (size_t i = 0; i < ggml.size(); ++i) {
            attrs[i] = static_cast<float>(ggml[i]);
        }
        record(node, result_->graph.add_op(ggml_graph::Op::Interpolate, {ensure(node.inputs[0])}, attrs),
               input.elem_type, target);
    }

    void build_clip(const Node& node) {
        require_count(node, 1, 3);
        const Shape input = shape_of(node.inputs[0]);
        require_supported_rank(input, node);
        float min_value = -INFINITY;
        float max_value = INFINITY;
        if (node.inputs.size() >= 2 && !node.inputs[1].empty()) {
            const auto constant = constants_.find(node.inputs[1]);
            if (constant == constants_.end()) {
                throw std::invalid_argument("Clip min must be a scalar constant");
            }
            const std::vector<float> values = tensor_f32(constant->second);
            if (values.size() != 1) {
                throw std::invalid_argument("Clip min must be a scalar constant");
            }
            min_value = values[0];
        }
        if (node.inputs.size() >= 3 && !node.inputs[2].empty()) {
            const auto constant = constants_.find(node.inputs[2]);
            if (constant == constants_.end()) {
                throw std::invalid_argument("Clip max must be a scalar constant");
            }
            const std::vector<float> values = tensor_f32(constant->second);
            if (values.size() != 1) {
                throw std::invalid_argument("Clip max must be a scalar constant");
            }
            max_value = values[0];
        }
        if (find_attribute(node, "min") != nullptr) {
            min_value = attribute_float(node, "min", min_value);
        }
        if (find_attribute(node, "max") != nullptr) {
            max_value = attribute_float(node, "max", max_value);
        }
        record(node, result_->graph.add_op(ggml_graph::Op::Clip, {ensure(node.inputs[0])}, {min_value, max_value}),
               input.elem_type, input.dims);
    }

    void build_constant(const Node& node) {
        const Attribute* value = find_attribute(node, "value");
        TensorProto tensor;
        if (value != nullptr && value->tensor.has_value()) {
            tensor = *value->tensor;
        } else if (find_attribute(node, "value_float") != nullptr) {
            tensor.data_type = 1;
            tensor.float_data = {attribute_float(node, "value_float", 0.0f)};
        } else if (find_attribute(node, "value_int") != nullptr) {
            tensor.data_type = 7;
            tensor.int64_data = {attribute_int(node, "value_int", 0)};
        } else if (find_attribute(node, "value_floats") != nullptr) {
            tensor.data_type = 1;
            tensor.float_data = find_attribute(node, "value_floats")->floats;
            tensor.dims = {static_cast<int64_t>(tensor.float_data.size())};
        } else if (find_attribute(node, "value_ints") != nullptr) {
            tensor.data_type = 7;
            tensor.int64_data = find_attribute(node, "value_ints")->ints;
            tensor.dims = {static_cast<int64_t>(tensor.int64_data.size())};
        } else {
            throw std::invalid_argument("Constant node has no supported value attribute");
        }
        tensor.name = node.outputs.at(0);
        if (tensor.data_type == 7) {
            i64_values_[tensor.name] = tensor_i64(tensor);
        }
        constants_[tensor.name] = tensor;
        set_shape(tensor.name, tensor.data_type, tensor.dims);
    }

    void build_shape(const Node& node) {
        require_count(node, 1, 1);
        const std::vector<int64_t> dimensions = logical_dims_of(node.inputs[0]);
        i64_values_[node.outputs[0]] = dimensions;
        std::vector<int32_t> values(dimensions.begin(), dimensions.end());
        const int id = result_->graph.add_const_i32(node.outputs[0], {static_cast<int64_t>(values.size())}, values);
        id_dtypes_[id] = ggml_graph::DType::I32;
        ids_[node.outputs[0]] = id;
        set_shape(node.outputs[0], 7, {static_cast<int64_t>(dimensions.size())});
    }

    void build_node(const Node& node) {
        if (!node.domain.empty() && node.domain != "ai.onnx" && node.domain != "onnx") {
            throw std::invalid_argument("unsupported ONNX domain '" + node.domain + "'");
        }
        const std::string& op = node.op_type;
        if (op == "Add") {
            build_binary(node, ggml_graph::Op::Add, true);
        } else if (op == "Sub") {
            build_binary(node, ggml_graph::Op::Sub, false);
        } else if (op == "Mul") {
            build_binary(node, ggml_graph::Op::Mul, true);
        } else if (op == "Div") {
            build_binary(node, ggml_graph::Op::Div, false);
        } else if (op == "Sqrt") {
            build_unary(node, ggml_graph::Op::Sqrt);
        } else if (op == "Relu") {
            build_unary(node, ggml_graph::Op::Relu);
        } else if (op == "Sigmoid") {
            build_unary(node, ggml_graph::Op::Sigmoid);
        } else if (op == "Abs") {
            if (const auto values = known_i64(node.inputs[0]); values.has_value()) {
                std::vector<int64_t> folded = *values;
                for (int64_t& value : folded) {
                    if (value < 0) {
                        value = -value;
                    }
                }
                register_i64(node.outputs.at(0), std::move(folded), logical_dims_of(node.inputs[0]));
            } else {
                build_unary(node, ggml_graph::Op::Abs);
            }
        } else if (op == "Log") {
            build_unary(node, ggml_graph::Op::Log);
        } else if (op == "Erf") {
            build_unary(node, ggml_graph::Op::Erf);
        } else if (op == "Pow") {
            build_pow(node);
        } else if (op == "ReduceMean") {
            build_reduce(node, ggml_graph::Op::ReduceMean);
        } else if (op == "ReduceSum") {
            build_reduce(node, ggml_graph::Op::ReduceSum);
        } else if (op == "ReduceMax") {
            build_reduce(node, ggml_graph::Op::ReduceMax);
        } else if (op == "Clip") {
            build_clip(node);
        } else if (op == "Pad") {
            build_pad(node);
        } else if (op == "GridSample") {
            build_grid_sample(node);
        } else if (op == "Resize") {
            build_resize(node);
        } else if (op == "Conv") {
            build_conv(node);
        } else if (op == "MaxPool") {
            build_max_pool(node);
        } else if (op == "MatMul") {
            build_matmul(node);
        } else if (op == "Gemm") {
            build_gemm(node);
        } else if (op == "Softmax") {
            build_softmax(node);
        } else if (op == "LayerNormalization") {
            build_layer_norm(node);
        } else if (op == "Gather") {
            build_gather(node);
        } else if (op == "Slice") {
            build_slice(node);
        } else if (op == "Split") {
            build_split(node);
        } else if (op == "TopK") {
            build_topk(node);
        } else if (op == "Expand") {
            build_expand(node);
        } else if (op == "Tile") {
            build_tile(node);
        } else if (op == "GatherElements") {
            build_gather_elements(node);
        } else if (op == "Reshape") {
            build_reshape(node);
        } else if (op == "Transpose") {
            build_transpose(node);
        } else if (op == "Concat") {
            build_concat(node);
        } else if (op == "Squeeze") {
            build_squeeze(node, false);
        } else if (op == "Unsqueeze") {
            build_squeeze(node, true);
        } else if (op == "Flatten") {
            build_flatten(node);
        } else if (op == "Cast") {
            build_cast(node);
        } else if (op == "Identity") {
            require_count(node, 1, 1);
            const Shape input = shape_of(node.inputs[0]);
            if (const auto values = known_i64(node.inputs[0]); values.has_value()) {
                register_i64(node.outputs.at(0), *values, input.dims);
                return;
            }
            record(node, ensure(node.inputs[0]), input.elem_type, input.dims);
        } else if (op == "ConstantOfShape") {
            build_constant_of_shape(node);
        } else if (op == "Constant") {
            build_constant(node);
        } else if (op == "Shape") {
            build_shape(node);
        } else {
            const std::string label = node.name.empty() ? op : op + " (" + node.name + ")";
            throw std::invalid_argument("unsupported ONNX op '" + label + "'");
        }
    }

    ModelProto proto_;
    std::unique_ptr<Model> result_;
    std::map<std::string, TensorProto> constants_;
    std::map<std::string, Shape> shapes_;
    std::map<std::string, std::vector<int64_t>> logical_dims_;
    std::map<std::string, size_t> axis_offsets_;
    std::map<std::string, int> ids_;
    std::map<int, ggml_graph::DType> id_dtypes_;
    std::map<std::string, std::vector<int64_t>> i64_values_;
};

} // namespace

std::unique_ptr<Model> parse(const uint8_t* data, size_t size, const std::string& label) {
    try {
        Builder builder(parse_model(data, size));
        return builder.build();
    } catch (const std::exception& error) {
        throw std::invalid_argument(label + ": " + error.what());
    }
}

std::unique_ptr<Model> parse(const std::vector<uint8_t>& bytes, const std::string& label) {
    return parse(bytes.data(), bytes.size(), label);
}

std::unique_ptr<Model> load(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::invalid_argument("cannot open ONNX file '" + path + "'");
    }
    const std::string bytes((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    return parse(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), path);
}

} // namespace ggml_onnx
