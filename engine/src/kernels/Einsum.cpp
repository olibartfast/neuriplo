// Einsum: Einstein summation over one or two float32 operands.
//
// The equation attribute lists one subscript per operand and optionally an
// explicit output after "->". Without "->" the output is every label that
// appears exactly once across the inputs, in alphabetical order. Labels absent
// from the output are summed over, and a label repeated within one operand
// selects its diagonal. Ellipsis is rejected. Spaces are ignored. Naive,
// single-threaded; products and sums accumulate in double.

#include "Kernels.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace engine {
namespace kernels {
namespace {

struct Equation {
    std::vector<std::string> inputs;
    std::string output;
};

Equation ParseEquation(const Node& node, const std::string& context) {
    const Attribute* attr = nullptr;
    for (const Attribute& a : node.attributes) {
        if (a.name == "equation") {
            attr = &a;
        }
    }
    if (attr == nullptr) {
        throw InferenceException(context + ": missing 'equation' attribute");
    }
    const std::string* raw = std::get_if<std::string>(&attr->value);
    if (raw == nullptr) {
        throw InferenceException(context + ": attribute 'equation' must be a string");
    }
    std::string text;
    for (char c : *raw) {
        if (c != ' ') {
            text.push_back(c);
        }
    }
    if (text.find('.') != std::string::npos) {
        throw InferenceException(context + ": ellipsis in the equation is not supported");
    }
    const std::size_t arrow = text.find("->");
    const bool explicitOutput = arrow != std::string::npos;
    const std::string lhs = explicitOutput ? text.substr(0, arrow) : text;

    Equation eq;
    std::size_t start = 0;
    while (true) {
        const std::size_t comma = lhs.find(',', start);
        eq.inputs.push_back(lhs.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    std::map<char, int> counts;
    for (const std::string& sub : eq.inputs) {
        for (char c : sub) {
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))) {
                throw InferenceException(context + ": invalid character in the equation");
            }
            ++counts[c];
        }
    }
    if (explicitOutput) {
        eq.output = text.substr(arrow + 2);
        for (std::size_t i = 0; i < eq.output.size(); ++i) {
            const char c = eq.output[i];
            if (counts.find(c) == counts.end()) {
                throw InferenceException(context + ": output label does not appear in the inputs");
            }
            if (eq.output.find(c) != i) {
                throw InferenceException(context + ": output label repeated");
            }
        }
    } else {
        for (const auto& entry : counts) {
            if (entry.second == 1) {
                eq.output.push_back(entry.first);
            }
        }
    }
    return eq;
}

} // namespace

void Einsum(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Einsum (node '" + node.name + "')";
    const Equation eq = ParseEquation(node, context);
    if (eq.inputs.size() != inputs.size() || inputs.empty() || inputs.size() > 2 || outputs.size() != 1) {
        throw InferenceException(context + ": operand count does not match the equation");
    }
    const TensorView& out = outputs[0];
    if (out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }

    std::map<char, int64_t> sizes;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].dtype != DataType::Float32) {
            throw InferenceException(context + ": tensors must be float32");
        }
        if (inputs[i].dims.size() != eq.inputs[i].size()) {
            throw InferenceException(context + ": operand rank does not match its subscript");
        }
        if (inputs[i].data == nullptr) {
            throw InferenceException(context + ": null buffer");
        }
        for (std::size_t j = 0; j < eq.inputs[i].size(); ++j) {
            const int64_t dim = inputs[i].dims[j];
            if (dim <= 0) {
                throw InferenceException(context + ": non-positive dimension");
            }
            const auto inserted = sizes.emplace(eq.inputs[i][j], dim);
            if (!inserted.second && inserted.first->second != dim) {
                throw InferenceException(context + ": label bound to two different sizes");
            }
        }
    }
    std::vector<int64_t> expected;
    for (char c : eq.output) {
        expected.push_back(sizes.at(c));
    }
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the operands");
    }
    if (out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    // Loop labels: every distinct label, with a per-operand and output stride.
    std::vector<char> labels;
    for (const auto& entry : sizes) {
        labels.push_back(entry.first);
    }
    const std::size_t n = labels.size();
    std::vector<std::vector<int64_t>> opStride(inputs.size(), std::vector<int64_t>(n, 0));
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        int64_t stride = 1;
        for (std::size_t j = eq.inputs[i].size(); j-- > 0;) {
            const std::size_t li =
                static_cast<std::size_t>(std::find(labels.begin(), labels.end(), eq.inputs[i][j]) - labels.begin());
            opStride[i][li] += stride;
            stride *= inputs[i].dims[j];
        }
    }
    std::vector<int64_t> outStride(n, 0);
    int64_t outCount = 1;
    {
        int64_t stride = 1;
        for (std::size_t j = eq.output.size(); j-- > 0;) {
            const std::size_t li =
                static_cast<std::size_t>(std::find(labels.begin(), labels.end(), eq.output[j]) - labels.begin());
            outStride[li] = stride;
            stride *= sizes.at(eq.output[j]);
        }
        outCount = stride;
    }

    // Products and sums accumulate in double; each output rounds once to float32.
    float* y = static_cast<float*>(out.data);
    std::vector<double> acc(static_cast<std::size_t>(outCount), 0.0);

    int64_t total = 1;
    for (char c : labels) {
        total *= sizes.at(c);
    }
    std::vector<int64_t> coord(n, 0);
    std::vector<int64_t> offset(inputs.size(), 0);
    int64_t outOffset = 0;
    for (int64_t step = 0; step < total; ++step) {
        double prod = 1.0;
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            prod *= static_cast<double>(static_cast<const float*>(inputs[i].data)[offset[i]]);
        }
        acc[static_cast<std::size_t>(outOffset)] += prod;
        // Advance the odometer, keeping the running offsets in sync.
        for (std::size_t li = n; li-- > 0;) {
            ++coord[li];
            for (std::size_t i = 0; i < inputs.size(); ++i) {
                offset[i] += opStride[i][li];
            }
            outOffset += outStride[li];
            if (coord[li] < sizes.at(labels[li])) {
                break;
            }
            const int64_t size = sizes.at(labels[li]);
            for (std::size_t i = 0; i < inputs.size(); ++i) {
                offset[i] -= opStride[i][li] * size;
            }
            outOffset -= outStride[li] * size;
            coord[li] = 0;
        }
    }
    for (int64_t i = 0; i < outCount; ++i) {
        y[i] = static_cast<float>(acc[static_cast<std::size_t>(i)]);
    }
}

} // namespace kernels
} // namespace engine
