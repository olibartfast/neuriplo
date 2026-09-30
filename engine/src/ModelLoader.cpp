// Reads an ONNX model file and builds the engine's graph.
//
// Covers the part of the model format the supported operators need: graph
// nodes, their attributes, float weight and int64 shape-constant initializers,
// and the declared inputs and outputs. The ONNX protobuf field numbers are a
// private detail of this file and the test encoders; no public header depends
// on them.
//
// Everything here fails at load time, never at inference, and rejection
// messages name the offending node and op type.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "WireReader.hpp" // private src-local reader

#include "engine/ModelLoader.hpp"

namespace engine {
namespace {

using Field = WireReader::Field;

// ---------------------------------------------------------------------------
// ONNX protobuf field numbers (private to this file and the test encoders).
// ---------------------------------------------------------------------------
constexpr int kField_Model_OpsetImport = 8;
constexpr int kField_Model_Graph = 7;
constexpr int kField_Opset_Domain = 1;
constexpr int kField_Opset_Version = 2;

constexpr int kField_Graph_Node = 1;
constexpr int kField_Graph_Initializer = 5;
constexpr int kField_Graph_Input = 11;
constexpr int kField_Graph_Output = 12;
constexpr int kField_Graph_ValueInfo = 13;

constexpr int kField_Node_Input = 1;
constexpr int kField_Node_Output = 2;
constexpr int kField_Node_Name = 3;
constexpr int kField_Node_OpType = 4;
constexpr int kField_Node_Attribute = 5;

constexpr int kField_Attr_Name = 1;
constexpr int kField_Attr_F = 2;
constexpr int kField_Attr_I = 3;
constexpr int kField_Attr_S = 4;
constexpr int kField_Attr_T = 5;
constexpr int kField_Attr_Floats = 7;
constexpr int kField_Attr_Ints = 8;
constexpr int kField_Attr_Type = 20;

// AttributeProto::AttributeType values recognized for rejection context.
constexpr int64_t kAttrType_Undefined = 0;
constexpr int64_t kAttrType_Float = 1;
constexpr int64_t kAttrType_Int = 2;
constexpr int64_t kAttrType_String = 3;
constexpr int64_t kAttrType_Tensor = 4;
constexpr int64_t kAttrType_Floats = 6;
constexpr int64_t kAttrType_Ints = 7;

constexpr int kField_Tensor_Dims = 1;
constexpr int kField_Tensor_DataType = 2;
constexpr int kField_Tensor_FloatData = 4;
constexpr int kField_Tensor_Int64Data = 7;
constexpr int kField_Tensor_Name = 8;
constexpr int kField_Tensor_RawData = 9;
constexpr int kField_Tensor_ExternalData = 13;
constexpr int kField_Tensor_DataLocation = 14;
constexpr int kField_Tensor_Ext_Key = 1;
constexpr int kField_Tensor_Ext_Value = 2;

// TensorProto::DataLocation
constexpr int64_t kDataLocation_External = 1;

constexpr int kField_ValueInfo_Name = 1;
constexpr int kField_ValueInfo_Type = 2;

constexpr int kField_TypeProto_TensorType = 1;
constexpr int kField_TensorType_ElemType = 1;
constexpr int kField_TensorType_Shape = 2;
constexpr int kField_Shape_Dim = 1;
constexpr int kField_Dim_Value = 1;
constexpr int kField_Dim_Param = 2;

// ONNX element types. Compute tensors are float32; int64 is accepted for
// embedded constants that carry shape indices and for int64 intermediate
// value-info entries; bool is accepted for boolean intermediate value-info
// entries. Graph inputs and outputs stay float32-only.
constexpr int64_t kElemType_OnnxFloat = 1;
constexpr int64_t kElemType_OnnxInt64 = 7;
constexpr int64_t kElemType_OnnxBool = 9;

// The opset version this loader targets.
constexpr int64_t kRequiredOpsetVersion = 18;
constexpr const char* kOnnxDomain = "ai.onnx";

const char* elem_type_name(int64_t dtype)
{
    switch (dtype) {
    case 0: return "UNDEFINED";
    case 1: return "FLOAT";
    case 2: return "UINT8";
    case 3: return "INT8";
    case 4: return "UINT16";
    case 5: return "INT16";
    case 6: return "INT32";
    case 7: return "INT64";
    case 8: return "STRING";
    case 9: return "BOOL";
    case 10: return "FLOAT16";
    case 11: return "DOUBLE";
    case 12: return "UINT32";
    case 13: return "UINT64";
    case 14: return "COMPLEX64";
    case 15: return "COMPLEX128";
    case 16: return "BFLOAT16";
    default: return "UNKNOWN";
    }
}

std::string copy_string(const uint8_t* data, size_t size)
{
    return std::string(reinterpret_cast<const char*>(data), size);
}

std::string quote(const std::string& s) { return "'" + s + "'"; }

// ---------------------------------------------------------------------------
// Basic read helpers
// ---------------------------------------------------------------------------

std::vector<uint8_t> read_file_bytes(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw ModelLoadException("cannot open model file: " + path);
    }
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
    if (bytes.empty()) {
        throw ModelLoadException("empty model file: " + path);
    }
    return bytes;
}

std::string dirname_of(const std::string& path)
{
    const size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? std::string(".")
                                      : path.substr(0, slash);
}

// A repeated int64 payload: packed (length-delimited varints) or unpacked.
void read_repeated_int64(WireReader& reader, WireType wire_type,
    std::vector<int64_t>& out)
{
    if (wire_type == WireType::LengthDelimited) {
        const auto payload = reader.read_length_delimited();
        WireReader sub(payload.first, payload.second);
        while (!sub.at_end()) {
            out.push_back(static_cast<int64_t>(sub.read_varint()));
        }
    } else {
        out.push_back(static_cast<int64_t>(reader.read_varint()));
    }
}

// A repeated float payload: packed (length-delimited fixed32s) or unpacked.
void read_repeated_float(WireReader& reader, WireType wire_type,
    std::vector<float>& out)
{
    auto push = [](uint32_t bits, std::vector<float>& out) {
        float f = 0.0f;
        std::memcpy(&f, &bits, sizeof(f));
        out.push_back(f);
    };
    if (wire_type == WireType::LengthDelimited) {
        const auto payload = reader.read_length_delimited();
        WireReader sub(payload.first, payload.second);
        while (!sub.at_end()) {
            push(sub.read_fixed32(), out);
        }
    } else {
        push(reader.read_fixed32(), out);
    }
}

std::string ld_string(WireReader& reader)
{
    const auto s = reader.read_length_delimited();
    return copy_string(s.first, s.second);
}

// ---------------------------------------------------------------------------
// TensorProto subset
// ---------------------------------------------------------------------------

struct TensorPatch {
    std::string name;
    int64_t data_type = 0;
    std::vector<int64_t> dims;
    std::vector<float> float_data;
    std::vector<int64_t> int64_data;
    std::string raw_data; // bytes as stored (little-endian elements)
    bool external = false;
    std::string external_location;
    int64_t external_offset = 0;
    int64_t external_length = -1; // -1: full file
};

TensorPatch decode_tensor(const uint8_t* data, size_t size)
{
    TensorPatch t;
    WireReader reader(data, size);
    while (!reader.at_end()) {
        const Field f = reader.read_tag();
        switch (f.field_number) {
        case kField_Tensor_Dims:
            read_repeated_int64(reader, f.wire_type, t.dims);
            break;
        case kField_Tensor_DataType:
            t.data_type = static_cast<int64_t>(reader.read_varint());
            break;
        case kField_Tensor_FloatData:
            read_repeated_float(reader, f.wire_type, t.float_data);
            break;
        case kField_Tensor_Int64Data:
            read_repeated_int64(reader, f.wire_type, t.int64_data);
            break;
        case kField_Tensor_Name:
            t.name = ld_string(reader);
            break;
        case kField_Tensor_RawData:
            if (f.wire_type != WireType::LengthDelimited) {
                throw ModelLoadException("malformed tensor raw_data field");
            }
            t.raw_data = ld_string(reader);
            break;
        case kField_Tensor_ExternalData: {
            if (f.wire_type != WireType::LengthDelimited) {
                throw ModelLoadException("malformed external_data entry");
            }
            const auto entry = reader.read_length_delimited();
            WireReader sub(entry.first, entry.second);
            std::string key;
            std::string value;
            while (!sub.at_end()) {
                const Field ef = sub.read_tag();
                if (ef.field_number == kField_Tensor_Ext_Key) {
                    key = ld_string(sub);
                } else if (ef.field_number == kField_Tensor_Ext_Value) {
                    value = ld_string(sub);
                } else {
                    sub.skip_field(ef.wire_type);
                }
            }
            if (key == "location") {
                t.external_location = value;
            } else if (key == "offset") {
                t.external_offset = std::stoll(value);
            } else if (key == "length") {
                t.external_length = std::stoll(value);
            }
            // Other keys (checksum, base-id) are ignored.
            break;
        }
        case kField_Tensor_DataLocation:
            t.external = reader.read_varint() == kDataLocation_External;
            break;
        default:
            reader.skip_field(f.wire_type);
            break;
        }
    }
    return t;
}

// Resolves the raw payload bytes of a tensor; external files are opened and
// read here and only here. No dtype interpretation happens here.
std::string load_tensor_data(const TensorPatch& t, const std::string& model_dir)
{
    if (!t.external) {
        return t.raw_data;
    }
    if (t.external_location.empty()) {
        throw ModelLoadException("missing external data location for tensor " +
            quote(t.name));
    }
    std::string full = t.external_location;
    if (full.front() != '/') {
        full = model_dir + "/" + full;
    }
    std::ifstream ext(full, std::ios::binary);
    if (!ext) {
        throw ModelLoadException("cannot open external tensor data for tensor " +
            quote(t.name) + " at " + full);
    }
    ext.seekg(t.external_offset);
    if (ext.fail()) {
        throw ModelLoadException("cannot seek external tensor data for tensor " +
            quote(t.name) + " at " + full);
    }
    std::string bytes;
    if (t.external_length < 0) {
        bytes.assign(std::istreambuf_iterator<char>(ext),
            std::istreambuf_iterator<char>());
    } else {
        bytes.resize(static_cast<size_t>(t.external_length));
        ext.read(&bytes[0], t.external_length);
        if (ext.gcount() != t.external_length) {
            throw ModelLoadException(
                "truncated external tensor data for tensor " + quote(t.name) +
                " at " + full);
        }
    }
    return bytes;
}

int64_t tensor_elem_count(const std::vector<int64_t>& dims)
{
    int64_t count = 1;
    for (int64_t d : dims) {
        count *= d;
    }
    return count;
}

DataType initializer_dtype(int64_t onnx_type, const std::string& name)
{
    switch (onnx_type) {
    case kElemType_OnnxFloat:
        return DataType::Float32;
    case kElemType_OnnxInt64:
        return DataType::Int64;
    default:
        throw ModelLoadException("unsupported initializer data type " +
            quote(elem_type_name(onnx_type)) + " for tensor " + quote(name) +
            " (engine accepts FLOAT weights and INT64 shape constants)");
    }
}

std::vector<float> float_initializer_values(const TensorPatch& t,
    const std::string& model_dir, int64_t count)
{
    if (!t.float_data.empty()) {
        if (static_cast<int64_t>(t.float_data.size()) != count) {
            throw ModelLoadException("initializer float_data element count " +
                std::to_string(t.float_data.size()) +
                " does not match shape for tensor " + quote(t.name));
        }
        return t.float_data;
    }
    const std::string bytes = load_tensor_data(t, model_dir);
    if (bytes.empty() && count != 0) {
        throw ModelLoadException("initializer tensor " + quote(t.name) +
            " carries no data");
    }
    if (bytes.size() % sizeof(float) != 0) {
        throw ModelLoadException(
            "initializer raw_data byte size is not a multiple of four for "
            "tensor " +
            quote(t.name));
    }
    const size_t elems = bytes.size() / sizeof(float);
    if (static_cast<int64_t>(elems) != count) {
        throw ModelLoadException("initializer element count " +
            std::to_string(elems) + " does not match shape for tensor " +
            quote(t.name));
    }
    std::vector<float> values(elems);
    if (elems > 0) {
        std::memcpy(values.data(), bytes.data(), bytes.size());
    }
    return values;
}

std::vector<int64_t> int64_initializer_values(const TensorPatch& t,
    const std::string& model_dir, int64_t count)
{
    if (!t.int64_data.empty()) {
        if (static_cast<int64_t>(t.int64_data.size()) != count) {
            throw ModelLoadException("initializer int64_data element count " +
                std::to_string(t.int64_data.size()) +
                " does not match shape for tensor " + quote(t.name));
        }
        return t.int64_data;
    }
    const std::string bytes = load_tensor_data(t, model_dir);
    if (bytes.empty() && count != 0) {
        throw ModelLoadException("initializer tensor " + quote(t.name) +
            " carries no data");
    }
    if (bytes.size() % sizeof(int64_t) != 0) {
        throw ModelLoadException(
            "initializer raw_data byte size is not a multiple of eight for "
            "tensor " +
            quote(t.name));
    }
    const size_t elems = bytes.size() / sizeof(int64_t);
    if (static_cast<int64_t>(elems) != count) {
        throw ModelLoadException("initializer element count " +
            std::to_string(elems) + " does not match shape for tensor " +
            quote(t.name));
    }
    std::vector<int64_t> values(elems);
    if (elems > 0) {
        std::memcpy(values.data(), bytes.data(), bytes.size());
    }
    return values;
}

Initializer materialize_initializer(const TensorPatch& t,
    const std::string& model_dir)
{
    if (t.name.empty()) {
        throw ModelLoadException("initializer tensor is missing its name");
    }
    const DataType dtype = initializer_dtype(t.data_type, t.name);
    const int64_t count = tensor_elem_count(t.dims);

    Initializer init;
    init.dtype = dtype;
    init.dims = t.dims;
    if (dtype == DataType::Float32) {
        init.values = float_initializer_values(t, model_dir, count);
    } else {
        init.values = int64_initializer_values(t, model_dir, count);
    }
    return init;
}

// ---------------------------------------------------------------------------
// Allowed attributes per operator; an empty set means the op takes none.
// ---------------------------------------------------------------------------

enum class AttrSlot { Int, Float, Str, IntList };

const std::unordered_map<std::string, AttrSlot>& conv_attrs()
{
    static const std::unordered_map<std::string, AttrSlot> k = {
        { "pads", AttrSlot::IntList },
        { "strides", AttrSlot::IntList },
        { "dilations", AttrSlot::IntList },
        { "group", AttrSlot::Int },
        { "auto_pad", AttrSlot::Str },
    };
    return k;
}

const std::unordered_map<std::string, AttrSlot>& maxpool_attrs()
{
    static const std::unordered_map<std::string, AttrSlot> k = {
        { "kernel_shape", AttrSlot::IntList },
        { "pads", AttrSlot::IntList },
        { "strides", AttrSlot::IntList },
        { "dilations", AttrSlot::IntList },
        { "ceil_mode", AttrSlot::Int },
        { "storage_order", AttrSlot::Int },
        { "auto_pad", AttrSlot::Str },
    };
    return k;
}

const std::unordered_map<std::string, AttrSlot>& reducemean_attrs()
{
    static const std::unordered_map<std::string, AttrSlot> k = {
        { "keepdims", AttrSlot::Int },
        { "noop_with_empty_axes", AttrSlot::Int },
    };
    return k;
}

const std::unordered_map<std::string, AttrSlot>& reshape_attrs()
{
    static const std::unordered_map<std::string, AttrSlot> k = {
        { "allowzero", AttrSlot::Int },
    };
    return k;
}

const std::unordered_map<std::string, AttrSlot>& gemm_attrs()
{
    static const std::unordered_map<std::string, AttrSlot> k = {
        { "alpha", AttrSlot::Float },
        { "beta", AttrSlot::Float },
        { "transA", AttrSlot::Int },
        { "transB", AttrSlot::Int },
    };
    return k;
}

const std::unordered_map<std::string, AttrSlot>& flatten_attrs()
{
    static const std::unordered_map<std::string, AttrSlot> k = {
        { "axis", AttrSlot::Int },
    };
    return k;
}

static const std::unordered_map<std::string, AttrSlot> kNoAttrs;

const std::unordered_map<std::string,
    const std::unordered_map<std::string, AttrSlot>*>&
allowed_attributes()
{
    static const std::unordered_map<std::string,
        const std::unordered_map<std::string, AttrSlot>*>
        kOps = {
            { "Conv", &conv_attrs() },
            { "MaxPool", &maxpool_attrs() },
            { "ReduceMean", &reducemean_attrs() },
            { "Reshape", &reshape_attrs() },
            { "Gemm", &gemm_attrs() },
            { "Flatten", &flatten_attrs() },
            { "Add", &kNoAttrs },
            { "Relu", &kNoAttrs },
            { "MatMul", &kNoAttrs },
        };
    return kOps;
}

// ---------------------------------------------------------------------------
// AttributeProto subset
// ---------------------------------------------------------------------------

struct AttrPatch {
    std::string name;
    int64_t type = kAttrType_Undefined;
    float f = 0.0f;
    int64_t i = 0;
    std::string s;
    std::vector<float> floats;
    std::vector<int64_t> ints;
};

AttrPatch decode_attr(const std::string& bytes)
{
    AttrPatch a;
    WireReader reader(reinterpret_cast<const uint8_t*>(bytes.data()),
        bytes.size());
    while (!reader.at_end()) {
        const Field f = reader.read_tag();
        switch (f.field_number) {
        case kField_Attr_Name:
            a.name = ld_string(reader);
            break;
        case kField_Attr_F: {
            const uint32_t bits = reader.read_fixed32();
            std::memcpy(&a.f, &bits, sizeof(a.f));
            break;
        }
        case kField_Attr_I:
            a.i = static_cast<int64_t>(reader.read_varint());
            break;
        case kField_Attr_S:
            a.s = ld_string(reader);
            break;
        case kField_Attr_Floats:
            read_repeated_float(reader, f.wire_type, a.floats);
            break;
        case kField_Attr_Ints:
            read_repeated_int64(reader, f.wire_type, a.ints);
            break;
        case kField_Attr_Type:
            a.type = static_cast<int64_t>(reader.read_varint());
            break;
        default:
            reader.skip_field(f.wire_type);
            break;
        }
    }
    return a;
}

std::string attr_msg_prefix(const std::string& node_name,
    const std::string& op_type)
{
    return "unsupported attribute on node" +
        (node_name.empty() ? std::string(" <unnamed>") : " " + quote(node_name)) +
        " op_type " + quote(op_type);
}

Attribute map_attribute(const AttrPatch& a, AttrSlot slot,
    const std::string& node_name, const std::string& op_type)
{
    const std::string prefix = attr_msg_prefix(node_name, op_type);
    auto reject = [&prefix, &a](const std::string& why) {
        throw ModelLoadException(prefix + ": attribute " + quote(a.name) + why);
    };
    switch (a.type) {
    case kAttrType_Int:
        if (slot != AttrSlot::Int) {
            reject(" does not take an int here");
        }
        return Attribute(a.name, a.i);
    case kAttrType_Float:
        if (slot != AttrSlot::Float) {
            reject(" does not take a float here");
        }
        return Attribute(a.name, a.f);
    case kAttrType_String:
        if (slot != AttrSlot::Str) {
            reject(" does not take a string here");
        }
        return Attribute(a.name, a.s);
    case kAttrType_Ints:
        if (slot != AttrSlot::IntList) {
            reject(" does not take an int list here");
        }
        return Attribute(a.name, a.ints);
    case kAttrType_Floats:
        if (slot != AttrSlot::Float) {
            reject(" does not take a float here");
        }
        if (a.floats.size() != 1) {
            reject(" must carry exactly one float here");
        }
        return Attribute(a.name, a.floats.front());
    case kAttrType_Undefined:
    default:
        // Sub-message and byte-list kinds the supported operators do not use.
        reject(" carries an unsupported value kind (" +
            quote("attribute type " + std::to_string(a.type)) + ")");
        break;
    }
    throw ModelLoadException(prefix + ": attribute " + quote(a.name) +
        " could not be decoded");
}

// ---------------------------------------------------------------------------
// NodeProto subset: raw fields first, attributes resolved afterwards (op_type
// and the attribute set cross-reference at that point).
// ---------------------------------------------------------------------------

namespace {

struct NodeRaw {
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<std::string> attr_bytes;
};

// Collects the raw ONNX NodeProto fields (inputs/outputs name strings,
// attribute payloads as raw bytes) without yet resolving attributes —
// op_type and the attribute set cross-reference afterwards.
NodeRaw read_node_raw(const uint8_t* data, size_t size)
{
    NodeRaw n;
    WireReader reader(data, size);
    while (!reader.at_end()) {
        const Field f = reader.read_tag();
        switch (f.field_number) {
        case kField_Node_Input:
            if (f.wire_type != WireType::LengthDelimited) {
                throw ModelLoadException("malformed node input field");
            }
            n.inputs.push_back(ld_string(reader));
            break;
        case kField_Node_Output:
            if (f.wire_type != WireType::LengthDelimited) {
                throw ModelLoadException("malformed node output field");
            }
            n.outputs.push_back(ld_string(reader));
            break;
        case kField_Node_Name:
            n.name = ld_string(reader);
            break;
        case kField_Node_OpType:
            n.op_type = ld_string(reader);
            break;
        case kField_Node_Attribute: {
            if (f.wire_type != WireType::LengthDelimited) {
                throw ModelLoadException("malformed node attribute field");
            }
            const auto payload = reader.read_length_delimited();
            n.attr_bytes.push_back(
                copy_string(payload.first, payload.second));
            break;
        }
        default:
            reader.skip_field(f.wire_type);
            break;
        }
    }
    return n;
}

} // namespace

Node decode_node(const uint8_t* data, size_t size)
{
    const NodeRaw raw = read_node_raw(data, size);

    if (raw.op_type.empty()) {
        throw ModelLoadException("node " +
            (raw.name.empty() ? std::string("<unnamed>") : quote(raw.name)) +
            " is missing op_type");
    }

    const auto& ops = allowed_attributes();
    const auto op_it = ops.find(raw.op_type);
    if (op_it == ops.end()) {
        throw ModelLoadException("unsupported op_type " + quote(raw.op_type) +
            " on node " +
            (raw.name.empty() ? std::string("<unnamed>") : quote(raw.name)) +
            " (engine accepts Conv/Gemm/MatMul/Add/Relu/MaxPool/ReduceMean/"
            "Reshape/Flatten only)");
    }

    const std::unordered_map<std::string, AttrSlot>& allowed = *op_it->second;
    std::set<std::string> seen;

    Node node;
    node.name = raw.name;
    node.op_type = raw.op_type;
    node.inputs = raw.inputs;
    node.outputs = raw.outputs;

    for (const std::string& bytes : raw.attr_bytes) {
        AttrPatch a = decode_attr(bytes);
        if (a.name.empty()) {
            throw ModelLoadException(
                attr_msg_prefix(node.name, node.op_type) +
                ": attribute without a name");
        }
        if (seen.count(a.name) != 0) {
            throw ModelLoadException(
                attr_msg_prefix(node.name, node.op_type) + ": duplicate " +
                quote(a.name));
        }
        seen.insert(a.name);
        const auto slot_it = allowed.find(a.name);
        if (slot_it == allowed.end()) {
            // Empty-allowlist ops (Add/Relu/MatMul) take no attributes at all.
            throw ModelLoadException(
                attr_msg_prefix(node.name, node.op_type) + ": attribute " +
                quote(a.name) + " is not in the accepted set");
        }
        node.attributes.push_back(
            map_attribute(a, slot_it->second, node.name, node.op_type));
    }
    return node;
}

// Folds one Constant node into an initializer entry. The node's single output
// name becomes the initializer key, so consumers already naming that tensor
// rewire to the embedded constant with no edge rewrite. Only float32 and
// int64 payloads fold; any other dtype is a load rejection with node context.
std::pair<std::string, Initializer> fold_constant_node(
    const NodeRaw& raw, const std::string& model_dir)
{
    const std::string who = "node " +
        (raw.name.empty() ? std::string("<unnamed>") : quote(raw.name)) +
        " op_type 'Constant'";
    if (!raw.inputs.empty()) {
        throw ModelLoadException(who + ": Constant must have no inputs");
    }
    if (raw.outputs.size() != 1 || raw.outputs[0].empty()) {
        throw ModelLoadException(
            who + ": Constant must produce exactly one named output");
    }
    if (raw.attr_bytes.size() != 1) {
        throw ModelLoadException(who +
            ": Constant must carry exactly one 'value' attribute");
    }
    const std::string& bytes = raw.attr_bytes.front();
    std::string attr_name;
    int64_t attr_type = kAttrType_Undefined;
    bool has_type = false;
    std::string tensor_bytes;
    bool has_tensor = false;
    WireReader reader(
        reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
    while (!reader.at_end()) {
        const Field f = reader.read_tag();
        switch (f.field_number) {
        case kField_Attr_Name:
            attr_name = ld_string(reader);
            break;
        case kField_Attr_Type:
            attr_type = static_cast<int64_t>(reader.read_varint());
            has_type = true;
            break;
        case kField_Attr_T:
            if (f.wire_type != WireType::LengthDelimited) {
                throw ModelLoadException(
                    who + ": attribute 'value' carries a malformed tensor");
            }
            tensor_bytes = ld_string(reader);
            has_tensor = true;
            break;
        default:
            reader.skip_field(f.wire_type);
            break;
        }
    }
    if (attr_name != "value") {
        throw ModelLoadException(who + ": Constant carries unsupported attribute " +
            quote(attr_name) + " (engine folds 'value' only)");
    }
    if (!has_type || attr_type != kAttrType_Tensor) {
        throw ModelLoadException(who +
            ": attribute 'value' must be a tensor (found attribute type " +
            std::to_string(attr_type) + ")");
    }
    if (!has_tensor || tensor_bytes.empty()) {
        throw ModelLoadException(
            who + ": attribute 'value' carries no tensor payload");
    }
    TensorPatch t = decode_tensor(
        reinterpret_cast<const uint8_t*>(tensor_bytes.data()),
        tensor_bytes.size());
    if (t.data_type != kElemType_OnnxFloat &&
        t.data_type != kElemType_OnnxInt64) {
        throw ModelLoadException("unsupported Constant value data type " +
            quote(elem_type_name(t.data_type)) + " on " + who +
            " (engine folds FLOAT and INT64 Constants only)");
    }
    t.name = raw.outputs[0];
    const int64_t count = tensor_elem_count(t.dims);
    Initializer init;
    init.dims = t.dims;
    if (t.data_type == kElemType_OnnxFloat) {
        init.dtype = DataType::Float32;
        init.values = float_initializer_values(t, model_dir, count);
    } else {
        init.dtype = DataType::Int64;
        init.values = int64_initializer_values(t, model_dir, count);
    }
    return {raw.outputs[0], std::move(init)};
}

// ---------------------------------------------------------------------------
// ValueInfoProto subset (graph inputs / outputs / value_info table).
// ---------------------------------------------------------------------------

std::pair<std::string, TensorInfo> decode_value_info(const uint8_t* data,
    size_t size, const char* kind, bool allow_integer = false,
    bool allow_bool = false)
{
    std::string name;
    std::string element; // payload bytes of the (nested) type sub-messages
    std::string shape;

    WireReader reader(data, size);
    while (!reader.at_end()) {
        const Field f = reader.read_tag();
        switch (f.field_number) {
        case kField_ValueInfo_Name:
            name = ld_string(reader);
            break;
        case kField_ValueInfo_Type:
            if (f.wire_type != WireType::LengthDelimited) {
                throw ModelLoadException("malformed value-info type field");
            }
            element = ld_string(reader);
            break;
        default:
            reader.skip_field(f.wire_type);
            break;
        }
    }

    const auto reject = [&name, kind](const std::string& what) {
        throw ModelLoadException(std::string(kind) + " tensor " +
            (name.empty() ? std::string("<unnamed>") : quote(name)) + " " +
            what);
    };

    if (name.empty()) {
        reject("is missing its name");
    }

    // TypeProto: tensor_type = 1.
    int64_t elem_type = -1;
    std::vector<int64_t> dims;
    if (!element.empty()) {
        WireReader type_reader(
            reinterpret_cast<const uint8_t*>(element.data()), element.size());
        while (!type_reader.at_end()) {
            const Field f = type_reader.read_tag();
            if (f.field_number == kField_TypeProto_TensorType) {
                if (f.wire_type != WireType::LengthDelimited) {
                    throw ModelLoadException(
                        "malformed tensor_type field in value-info " +
                        quote(name));
                }
                const auto tt = type_reader.read_length_delimited();
                WireReader tt_reader(tt.first, tt.second);
                while (!tt_reader.at_end()) {
                    const Field tf = tt_reader.read_tag();
                    switch (tf.field_number) {
                    case kField_TensorType_ElemType:
                        elem_type = static_cast<int64_t>(
                            tt_reader.read_varint());
                        break;
                    case kField_TensorType_Shape:
                        if (tf.wire_type != WireType::LengthDelimited) {
                            throw ModelLoadException(
                                "malformed shape field in value-info " +
                                quote(name));
                        }
                        shape = ld_string(tt_reader);
                        break;
                    default:
                        tt_reader.skip_field(tf.wire_type);
                        break;
                    }
                }
            } else {
                type_reader.skip_field(f.wire_type);
            }
        }
    }

    DataType dtype = DataType::Unknown;
    if (elem_type == kElemType_OnnxFloat) {
        dtype = DataType::Float32;
    } else if (allow_integer && elem_type == kElemType_OnnxInt64) {
        dtype = DataType::Int64;
    } else if (allow_bool && elem_type == kElemType_OnnxBool) {
        dtype = DataType::Bool;
    } else {
        reject(std::string("carries unsupported element type ") +
            quote(elem_type_name(elem_type)) +
            ((allow_integer || allow_bool)
                    ? " (engine accepts FLOAT compute tensors, INT64 "
                      "constants, and BOOL intermediates only)"
                    : " (engine accepts FLOAT only)"));
    }

    // TensorShapeProto: dim = 1 (repeated); each dim carries dim_value = 1 or
    // dim_param = 2. Only static shapes are supported.
    if (!shape.empty()) {
        WireReader shape_reader(
            reinterpret_cast<const uint8_t*>(shape.data()), shape.size());
        while (!shape_reader.at_end()) {
            const Field f = shape_reader.read_tag();
            if (f.field_number != kField_Shape_Dim) {
                shape_reader.skip_field(f.wire_type);
                continue;
            }
            if (f.wire_type != WireType::LengthDelimited) {
                throw ModelLoadException(
                    "malformed dim field in value-info " + quote(name));
            }
            const auto dim = shape_reader.read_length_delimited();
            WireReader dim_reader(dim.first, dim.second);
            bool has_value = false;
            int64_t dim_value = 0;
            std::string dim_param;
            while (!dim_reader.at_end()) {
                const Field df = dim_reader.read_tag();
                switch (df.field_number) {
                case kField_Dim_Value:
                    dim_value = static_cast<int64_t>(dim_reader.read_varint());
                    has_value = true;
                    dim_param.clear();
                    break;
                case kField_Dim_Param:
                    dim_param = ld_string(dim_reader);
                    break;
                default:
                    dim_reader.skip_field(df.wire_type);
                    break;
                }
            }
            if (!has_value || dim_value <= 0) {
                reject("carries a non-fixed dimension" +
                    (dim_param.empty()
                            ? std::string()
                            : " (dim_param " + quote(dim_param) + ")") +
                    " (engine accepts static positive shapes only)");
            }
            dims.push_back(dim_value);
        }
    }

    TensorInfo info;
    info.dtype = dtype;
    info.dims = dims;
    return { name, info };
}

// ---------------------------------------------------------------------------
// Top-level decode: ModelProto -> GraphProto -> Graph IR, then verification.
// ---------------------------------------------------------------------------

namespace {

// Decodes one ModelProto. opset imports are validated here; the graph body
// decode needs the model directory for external initializer data.
Graph decode_model(const std::string& model_dir, const uint8_t* data,
    size_t size)
{
    bool has_required_opset = false;
    std::string graph_bytes;
    std::string opset_bytes; // current repeated entry payload

    WireReader reader(data, size);
    while (!reader.at_end()) {
        const Field f = reader.read_tag();
        switch (f.field_number) {
        case kField_Model_Graph:
            if (f.wire_type != WireType::LengthDelimited) {
                throw ModelLoadException("malformed model graph field");
            }
            graph_bytes = ld_string(reader);
            break;
        case kField_Model_OpsetImport: {
            if (f.wire_type != WireType::LengthDelimited) {
                throw ModelLoadException("malformed opset_import field");
            }
            const auto entry = reader.read_length_delimited();
            WireReader sub(entry.first, entry.second);
            std::string domain;
            int64_t version = -1;
            bool has_version = false;
            while (!sub.at_end()) {
                const Field sf = sub.read_tag();
                switch (sf.field_number) {
                case kField_Opset_Domain:
                    domain = ld_string(sub);
                    break;
                case kField_Opset_Version:
                    version = static_cast<int64_t>(sub.read_varint());
                    has_version = true;
                    break;
                default:
                    sub.skip_field(sf.wire_type);
                    break;
                }
            }
            if (domain.empty() || domain == kOnnxDomain) {
                if (!has_version || version != kRequiredOpsetVersion) {
                    throw ModelLoadException(
                        "model requires ONNX-domain opset " +
                        std::to_string(kRequiredOpsetVersion) +
                        (has_version ? ", found " + std::to_string(version)
                                     : " but declares none"));
                }
                has_required_opset = true;
            }
            // Unknown domains (ai.onnx.training, etc.) are skipped.
            break;
        }
        default:
            reader.skip_field(f.wire_type);
            break;
        }
    }

    (void)opset_bytes;
    if (graph_bytes.empty()) {
        throw ModelLoadException("model carries no graph");
    }
    if (!has_required_opset) {
        throw ModelLoadException("model does not declare ONNX-domain opset " +
            std::to_string(kRequiredOpsetVersion));
    }

    Graph graph;
    std::vector<std::pair<std::string, Initializer>> folded_constants;
    std::vector<std::string> folded_node_names;
    WireReader graph_reader(
        reinterpret_cast<const uint8_t*>(graph_bytes.data()),
        graph_bytes.size());
    while (!graph_reader.at_end()) {
        const Field f = graph_reader.read_tag();
        switch (f.field_number) {
        case kField_Graph_Node:
            if (f.wire_type != WireType::LengthDelimited) {
                throw ModelLoadException("malformed graph node field");
            }
            {
                const auto payload = graph_reader.read_length_delimited();
                const NodeRaw raw =
                    read_node_raw(payload.first, payload.second);
                if (raw.op_type == "Constant") {
                    auto folded =
                        fold_constant_node(raw, model_dir);
                    folded_node_names.push_back(raw.name);
                    folded_constants.push_back(std::move(folded));
                } else {
                    graph.nodes.push_back(
                        decode_node(payload.first, payload.second));
                }
            }
            break;
        case kField_Graph_Initializer: {
            if (f.wire_type != WireType::LengthDelimited) {
                throw ModelLoadException("malformed graph initializer field");
            }
            const auto payload = graph_reader.read_length_delimited();
            const TensorPatch t = decode_tensor(payload.first, payload.second);
            if (graph.initializers.count(t.name) != 0) {
                throw ModelLoadException(
                    "duplicate initializer " + quote(t.name));
            }
            graph.initializers[t.name] =
                materialize_initializer(t, model_dir);
            break;
        }
        case kField_Graph_Input: {
            if (f.wire_type != WireType::LengthDelimited) {
                throw ModelLoadException("malformed graph input field");
            }
            const auto payload = graph_reader.read_length_delimited();
            graph.inputs.push_back(
                decode_value_info(payload.first, payload.second,
                    "graph input"));
            break;
        }
        case kField_Graph_Output: {
            if (f.wire_type != WireType::LengthDelimited) {
                throw ModelLoadException("malformed graph output field");
            }
            const auto payload = graph_reader.read_length_delimited();
            graph.outputs.push_back(
                decode_value_info(payload.first, payload.second,
                    "graph output"));
            break;
        }
        case kField_Graph_ValueInfo: {
            if (f.wire_type != WireType::LengthDelimited) {
                throw ModelLoadException("malformed graph value_info field");
            }
            const auto payload = graph_reader.read_length_delimited();
            const auto entry = decode_value_info(payload.first, payload.second,
                "value_info", /*allow_integer=*/true, /*allow_bool=*/true);
            if (graph.tensors.count(entry.first) != 0) {
                throw ModelLoadException(
                    "duplicate value-info entry " + quote(entry.first));
            }
            graph.tensors[entry.first] = entry.second;
            break;
        }
        default:
            graph_reader.skip_field(f.wire_type);
            break;
        }
    }
    for (size_t i = 0; i < folded_constants.size(); ++i) {
        const std::string& name = folded_constants[i].first;
        const std::string& node_name = folded_node_names[i];
        const std::string who = "node " +
            (node_name.empty() ? std::string("<unnamed>")
                               : quote(node_name)) +
            " op_type 'Constant'";
        if (graph.initializers.count(name) != 0) {
            throw ModelLoadException("duplicate initializer " + quote(name) +
                " folded from " + who);
        }
        graph.initializers[name] = std::move(folded_constants[i].second);
    }
    return graph;
}

// Every node input must be a graph input, an initializer, or the output of an
// earlier node. Rejections name the node and its op type.
void verify_topological_order(const Graph& graph)
{
    std::map<std::string, bool> known;
    for (const auto& io : graph.inputs) {
        known[io.first] = true;
    }
    for (const auto& init : graph.initializers) {
        known[init.first] = true;
    }

    for (const Node& node : graph.nodes) {
        const std::string who =
            node.name.empty() ? quote(node.op_type) + " node <unnamed>"
                              : "node " + quote(node.name) + " op_type " +
                quote(node.op_type);
        for (const std::string& input : node.inputs) {
            // Empty strings model optional/omitted inputs (e.g. Conv B) and
            // are not real edges.
            if (input.empty()) {
                continue;
            }
            if (known.count(input) == 0) {
                throw ModelLoadException(
                    "graph is not topologically sorted: " + who +
                    " consumes tensor " + quote(input) +
                    " before it is produced (no input, initializer, or "
                    "earlier node output provides it)");
            }
        }
        for (const std::string& output : node.outputs) {
            if (!output.empty()) {
                known[output] = true;
            }
        }
    }
}

} // namespace
} // namespace (top-level anonymous)

} // namespace engine

namespace engine {

Graph LoadGraphFromFile(const std::string& path)
{
    const std::vector<uint8_t> bytes = read_file_bytes(path);
    const std::string model_dir = dirname_of(path);
    Graph graph = decode_model(model_dir, bytes.data(), bytes.size());
    verify_topological_order(graph);
    return graph;
}

} // namespace engine
