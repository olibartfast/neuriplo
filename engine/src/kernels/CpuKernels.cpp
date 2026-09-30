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
        {"Reshape", &Reshape},
        {"ReduceMean", &ReduceMean},
    };
    const auto it = table.find(op_type);
    return it == table.end() ? nullptr : it->second;
}

} // namespace kernels
} // namespace engine
