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

    int transpose_onnx(int id, const std::vector<int64_t>& perm) {
        const int rank = static_cast<int>(perm.size());
        std::vector<float> axes(4);
        for (int j = 0; j < 4; ++j) {
            if (j < rank) {
                axes[static_cast<size_t>(j)] = static_cast<float>((rank - 1) - perm[static_cast<size_t>(rank - 1 - j)]);
            } else {
                axes[static_cast<size_t>(j)] = static_cast<float>(j);
            }
        }
        return result_->graph.add_op(ggml_graph::Op::Transpose, {id}, axes);
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

    void record(const Node& node, int id, int32_t elem_type, std::vector<int64_t> dims) {
        if (node.outputs.empty() || node.outputs[0].empty()) {
            throw std::invalid_argument("node '" + node.name + "' has no primary output");
        }
        for (size_t i = 1; i < node.outputs.size(); ++i) {
            if (!node.outputs[i].empty()) {
                throw std::invalid_argument("multi-output ONNX op '" + node.op_type + "' is not supported");
            }
        }
        ids_[node.outputs[0]] = id;
        id_dtypes_[id] = elem_type == 1 || elem_type == 10 || elem_type == 11
                             ? ggml_graph::DType::F32
                             : (elem_type == 7 ? ggml_graph::DType::I64 : ggml_graph::DType::I32);
        shapes_[node.outputs[0]] = {elem_type, std::move(dims)};
    }

    void require_count(const Node& node, size_t min_inputs, size_t max_inputs) const {
        if (node.inputs.size() < min_inputs || node.inputs.size() > max_inputs) {
            throw std::invalid_argument("op '" + node.op_type + "' has an unsupported input count");
        }
    }

    void build_binary(const Node& node, ggml_graph::Op op, bool commutative) {
        require_count(node, 2, 2);
        const Shape left = shape_of(node.inputs[0]);
        const Shape right = shape_of(node.inputs[1]);
        if (!is_float_type(left.elem_type) || !is_float_type(right.elem_type)) {
            throw std::invalid_argument("op '" + node.op_type + "' requires FLOAT inputs");
        }
        const std::vector<int64_t> dims = broadcast_dims(left.dims, right.dims);
        int a = ensure(node.inputs[0]);
        int b = ensure(node.inputs[1]);
        if (left.dims.size() < right.dims.size()) {
            if (!commutative) {
                throw std::invalid_argument("op '" + node.op_type + "' cannot broadcast a lower-rank left operand");
            }
            std::swap(a, b);
        }
        record(node, result_->graph.add_op(op, {a, b}), 1, dims);
    }

    void build_unary(const Node& node, ggml_graph::Op op) {
        require_count(node, 1, 1);
        const Shape input = shape_of(node.inputs[0]);
        record(node, result_->graph.add_op(op, {ensure(node.inputs[0])}), input.elem_type, input.dims);
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
            output = result_->graph.add_op(ggml_graph::Op::Mul, {output, scalar_f32(weight_name + "#alpha", alpha)});
        }
        return output;
    }

    void build_matmul(const Node& node) {
        require_count(node, 2, 2);
        const Shape a = shape_of(node.inputs[0]);
        const Shape b = shape_of(node.inputs[1]);
        if (a.dims.size() != 2 || b.dims.size() != 2) {
            throw std::invalid_argument("MatMul supports rank-2 operands only");
        }
        if (a.dims[1] != b.dims[0]) {
            throw std::invalid_argument("MatMul inner dimensions do not match");
        }
        const int id = build_linear(node.inputs[0], node.inputs[1], true, "", 1.0f, b.dims[1]);
        record(node, id, 1, {a.dims[0], b.dims[1]});
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
            scaled_bias = ensure(node.inputs[2]);
            if (beta != 1.0f) {
                scaled_bias =
                    result_->graph.add_op(ggml_graph::Op::Mul, {scaled_bias, scalar_f32(node.name + "#beta", beta)});
            }
            fused_bias = alpha == 1.0f;
        }
        bias = fused_bias ? scaled_bias
                          : result_->graph.add_const_f32(node.name + "#zero_bias", {columns},
                                                         std::vector<float>(static_cast<size_t>(columns), 0.0f));
        int output = result_->graph.add_op(ggml_graph::Op::Linear, {x, weight, bias});
        if (alpha != 1.0f) {
            output = result_->graph.add_op(ggml_graph::Op::Mul, {output, scalar_f32(node.name + "#alpha", alpha)});
        }
        if (scaled_bias >= 0 && !fused_bias) {
            output = result_->graph.add_op(ggml_graph::Op::Add, {output, scaled_bias});
        }
        record(node, output, 1, {rows, columns});
    }

    std::vector<int64_t> lookup_i64(const std::string& name) {
        const auto cached = i64_values_.find(name);
        if (cached != i64_values_.end()) {
            return cached->second;
        }
        const auto constant = constants_.find(name);
        if (constant == constants_.end()) {
            throw std::invalid_argument("Reshape/axis input '" + name + "' is not a constant");
        }
        const std::vector<int64_t> values = tensor_i64(constant->second);
        i64_values_[name] = values;
        return values;
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
        if (target.empty() || target.size() > 4) {
            throw std::invalid_argument("Reshape target rank must be 1..4");
        }
        return target;
    }

    void build_reshape(const Node& node) {
        require_count(node, 1, 2);
        const Shape input = shape_of(node.inputs[0]);
        const std::vector<int64_t> target = reshape_target(node, input);
        const std::vector<int64_t> dims = ggml_dims(target);
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
        std::vector<int64_t> perm = attribute_ints(node, "perm");
        if (perm.empty()) {
            for (size_t i = 0; i < input.dims.size(); ++i) {
                perm.push_back(static_cast<int64_t>(input.dims.size() - 1 - i));
            }
        }
        if (perm.size() != input.dims.size()) {
            throw std::invalid_argument("Transpose perm rank does not match the input");
        }
        std::vector<int64_t> dims(perm.size());
        for (size_t i = 0; i < perm.size(); ++i) {
            if (perm[i] < 0 || perm[i] >= static_cast<int64_t>(input.dims.size())) {
                throw std::invalid_argument("Transpose perm is out of range");
            }
            dims[i] = input.dims[static_cast<size_t>(perm[i])];
        }
        record(node, transpose_onnx(ensure(node.inputs[0]), perm), 1, dims);
    }

    void build_concat(const Node& node) {
        require_count(node, 2, static_cast<size_t>(-1));
        const Shape first = shape_of(node.inputs[0]);
        const int64_t rank = static_cast<int64_t>(first.dims.size());
        int64_t axis = attribute_int(node, "axis", 0);
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || axis >= rank) {
            throw std::invalid_argument("Concat axis is out of range");
        }
        std::vector<int64_t> dims = first.dims;
        dims[static_cast<size_t>(axis)] = 0;
        std::vector<int> inputs;
        for (const auto& name : node.inputs) {
            const Shape shape = shape_of(name);
            if (shape.dims.size() != first.dims.size()) {
                throw std::invalid_argument("Concat input ranks do not match at node '" + node.name + "' (input '" +
                                            name + "')");
            }
            for (size_t i = 0; i < shape.dims.size(); ++i) {
                if (i != static_cast<size_t>(axis) && shape.dims[i] != first.dims[i]) {
                    throw std::invalid_argument("Concat input shapes do not match at node '" + node.name + "'");
                }
            }
            dims[static_cast<size_t>(axis)] += shape.dims[static_cast<size_t>(axis)];
            inputs.push_back(ensure(name));
        }
        const int ggml_axis = static_cast<int>(rank - 1 - axis);
        record(node, result_->graph.add_op(ggml_graph::Op::Concat, inputs, {static_cast<float>(ggml_axis)}), 1, dims);
    }

    void build_gather(const Node& node) {
        require_count(node, 2, 2);
        const Shape data = shape_of(node.inputs[0]);
        const Shape indices = shape_of(node.inputs[1]);
        int64_t axis = attribute_int(node, "axis", 0);
        if (axis < 0) {
            axis += static_cast<int64_t>(data.dims.size());
        }
        if (axis != 0 || data.dims.size() != 2 || indices.dims.size() != 1) {
            throw std::invalid_argument("Gather supports rank-2 data with axis=0 and rank-1 indices");
        }
        int index_id = ensure(node.inputs[1]);
        const ggml_graph::DType dtype = id_dtypes_.at(index_id);
        if (dtype == ggml_graph::DType::I64) {
            index_id = result_->graph.add_op(ggml_graph::Op::CastI64ToI32, {index_id});
            id_dtypes_[index_id] = ggml_graph::DType::I32;
        } else if (dtype != ggml_graph::DType::I32) {
            throw std::invalid_argument("Gather indices must be INT32 or INT64");
        }
        record(node, result_->graph.add_op(ggml_graph::Op::Gather, {ensure(node.inputs[0]), index_id}), 1,
               {indices.dims[0], data.dims[1]});
    }

    void build_softmax(const Node& node) {
        require_count(node, 1, 1);
        const Shape input = shape_of(node.inputs[0]);
        int64_t axis = attribute_int(node, "axis", -1);
        if (axis < 0) {
            axis += static_cast<int64_t>(input.dims.size());
        }
        if (axis != static_cast<int64_t>(input.dims.size()) - 1) {
            throw std::invalid_argument("Softmax supports the last axis only");
        }
        record(node, result_->graph.add_op(ggml_graph::Op::Softmax, {ensure(node.inputs[0])}), input.elem_type,
               input.dims);
    }

    void build_layer_norm(const Node& node) {
        require_count(node, 2, 3);
        const Shape input = shape_of(node.inputs[0]);
        int64_t axis = attribute_int(node, "axis", -1);
        if (axis < 0) {
            axis += static_cast<int64_t>(input.dims.size());
        }
        if (axis != static_cast<int64_t>(input.dims.size()) - 1) {
            throw std::invalid_argument("LayerNormalization supports the last axis only");
        }
        int bias = -1;
        if (node.inputs.size() == 3 && !node.inputs[2].empty()) {
            bias = ensure(node.inputs[2]);
        } else {
            bias = result_->graph.add_const_f32(node.name + "#zero_bias", {input.dims.back()},
                                                std::vector<float>(static_cast<size_t>(input.dims.back()), 0.0f));
        }
        const float epsilon = attribute_float(node, "epsilon", 1e-5f);
        record(node,
               result_->graph.add_op(ggml_graph::Op::LayerNorm, {ensure(node.inputs[0]), ensure(node.inputs[1]), bias},
                                     {epsilon}),
               input.elem_type, input.dims);
    }

    void build_cast(const Node& node) {
        require_count(node, 1, 1);
        const Shape input = shape_of(node.inputs[0]);
        const int64_t target = attribute_int(node, "to", 0);
        const int id = ensure(node.inputs[0]);
        const ggml_graph::DType source = id_dtypes_.at(id);
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
        } else if (target == 7 && source == ggml_graph::DType::I32) {
            // Shape arithmetic is carried as I32 inside the graph; keep the
            // ONNX-visible dtype so downstream shape ops accept it.
            record(node, id, 7, input.dims);
            return;
        }
        throw std::invalid_argument("Cast to element type " + std::to_string(target) + " is not supported");
    }

    void build_squeeze(const Node& node, bool unsqueeze) {
        require_count(node, 1, 2);
        const Shape input = shape_of(node.inputs[0]);
        std::vector<int64_t> axes = attribute_ints(node, "axes");
        if (axes.empty() && node.inputs.size() == 2 && !node.inputs[1].empty()) {
            axes = lookup_i64(node.inputs[1]);
        }
        std::vector<int64_t> dims;
        if (unsqueeze) {
            const int64_t output_rank = static_cast<int64_t>(input.dims.size() + axes.size());
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
                    dims.push_back(input.dims[source++]);
                }
            }
        } else {
            std::set<int64_t> normalized;
            for (int64_t axis : axes) {
                if (axis < 0) {
                    axis += static_cast<int64_t>(input.dims.size());
                }
                if (axis < 0 || axis >= static_cast<int64_t>(input.dims.size()) ||
                    input.dims[static_cast<size_t>(axis)] != 1) {
                    throw std::invalid_argument("Squeeze axis is out of range or not one");
                }
                normalized.insert(axis);
            }
            for (size_t i = 0; i < input.dims.size(); ++i) {
                if (normalized.empty() ? input.dims[i] != 1 : normalized.count(static_cast<int64_t>(i)) == 0) {
                    dims.push_back(input.dims[i]);
                }
            }
        }
        if (dims.empty() || dims.size() > 4) {
            throw std::invalid_argument("Squeeze/Unsqueeze result rank must be 1..4");
        }
        const std::vector<int64_t> ggml = ggml_dims(dims);
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
        shapes_[tensor.name] = {tensor.data_type, dims};
    }

    void build_clip(const Node& node) {
        require_count(node, 1, 3);
        const Shape input = shape_of(node.inputs[0]);
        float min_value = -INFINITY;
        float max_value = INFINITY;
        if (node.inputs.size() >= 2 && !node.inputs[1].empty()) {
            const std::vector<float> values = tensor_f32(constants_.at(node.inputs[1]));
            if (values.size() != 1) {
                throw std::invalid_argument("Clip min must be a scalar constant");
            }
            min_value = values[0];
        }
        if (node.inputs.size() >= 3 && !node.inputs[2].empty()) {
            const std::vector<float> values = tensor_f32(constants_.at(node.inputs[2]));
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
        shapes_[tensor.name] = {tensor.data_type, tensor.dims};
    }

    void build_shape(const Node& node) {
        require_count(node, 1, 1);
        const Shape input = shape_of(node.inputs[0]);
        i64_values_[node.outputs[0]] = input.dims;
        std::vector<int32_t> values(input.dims.begin(), input.dims.end());
        const int id = result_->graph.add_const_i32(node.outputs[0], {static_cast<int64_t>(values.size())}, values);
        id_dtypes_[id] = ggml_graph::DType::I32;
        ids_[node.outputs[0]] = id;
        shapes_[node.outputs[0]] = {7, {static_cast<int64_t>(input.dims.size())}};
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
        } else if (op == "Clip") {
            build_clip(node);
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
