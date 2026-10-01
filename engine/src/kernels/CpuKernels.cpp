// The CPU operator table.
//
// One function-local static maps the exact op-type strings to their kernels;
// unknown op types resolve to null. There is no global mutable registry.

#include "Kernels.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace engine {
namespace kernels {

KernelFn FindCpuKernel(const std::string& op_type) {
    static const std::unordered_map<std::string, KernelFn> table = {
        {"Relu", &Relu},
        {"Add", &Add},
        {"Mul", &Mul},
        {"Div", &Div},
        {"Sub", &Sub},
        {"Sigmoid", &Sigmoid},
        {"Cast", &Cast},
        {"Softmax", &Softmax},
        {"ReduceMax", &ReduceMax},
        {"Mod", &Mod},
        {"Equal", &Equal},
        {"Where", &Where},
        {"Reshape", &Reshape},
        {"ReduceMean", &ReduceMean},
        {"Gemm", &Gemm},
        {"MatMul", &MatMul},
        {"Conv", &Conv},
        {"MaxPool", &MaxPool},
        {"Concat", &Concat},
        {"Split", &Split},
        {"Unsqueeze", &Unsqueeze},
        {"Expand", &Expand},
        {"Transpose", &Transpose},
        {"Slice", &Slice},
        {"Gather", &Gather},
        {"GatherElements", &GatherElements},
        {"Resize", &Resize},
        {"Flatten", &Flatten},
        {"Shape", &Shape},
        {"ConstantOfShape", &ConstantOfShape},
        {"TopK", &TopK},
    };
    const auto it = table.find(op_type);
    return it == table.end() ? nullptr : it->second;
}

} // namespace kernels
} // namespace engine
